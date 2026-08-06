"""PomSD3Transformer2DModel: SD3.5 transformer with PoM instead of attention."""
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin, SD3Transformer2DLoadersMixin
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import logging

from .blocks import JointLocalAttnBlock, JointPoMBlock

logger = logging.get_logger(__name__)


class PixelPatchEmbed(nn.Module):
    """Pixel-space patch embedding for JiT-style flow matching (no VAE).

    Replaces Conv2d(in_channels→embed_dim) with a two-stage linear bottleneck
    (patch_dim → bottleneck_dim → embed_dim) that avoids projecting 3072-dim
    patch vectors directly into the 1536-dim transformer space.

    Position embedding: 2D sin-cos at a fixed max grid size, bilinear-interpolated
    for other grid sizes at runtime.
    """

    def __init__(
        self,
        patch_size: int = 32,
        in_channels: int = 3,
        bottleneck_dim: int = 256,
        embed_dim: int = 1536,
        pos_embed_max_size: int = 64,
    ):
        super().__init__()
        self.patch_size = patch_size
        # Stage 1: Conv2d is mathematically identical to a linear map on non-overlapping patches
        # (same kernel size and stride) but operates on (B, C, H, W) input directly.
        self.patch_proj = nn.Conv2d(
            in_channels, bottleneck_dim, kernel_size=patch_size, stride=patch_size
        )
        # Stage 2: expand bottleneck to transformer dim
        self.expand_proj = nn.Linear(bottleneck_dim, embed_dim)
        # Pre-compute sinusoidal pos_embed at max grid size; interpolate for other sizes
        pos = self._build_sincos2d(embed_dim, pos_embed_max_size, pos_embed_max_size)
        self.register_buffer("pos_embed", pos.unsqueeze(0))  # (1, max_size², embed_dim)
        self.pos_embed_max_size = pos_embed_max_size

    @staticmethod
    def _build_sincos2d(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        d = embed_dim // 4
        inv_freq = 1.0 / (10000 ** (torch.arange(d, dtype=torch.float32) / d))
        gy = torch.arange(grid_h, dtype=torch.float32)
        gx = torch.arange(grid_w, dtype=torch.float32)
        ey = torch.outer(gy, inv_freq)  # (h, d)
        ex = torch.outer(gx, inv_freq)  # (w, d)
        emb = torch.cat([
            ey.sin().unsqueeze(1).expand(grid_h, grid_w, d),
            ey.cos().unsqueeze(1).expand(grid_h, grid_w, d),
            ex.sin().unsqueeze(0).expand(grid_h, grid_w, d),
            ex.cos().unsqueeze(0).expand(grid_h, grid_w, d),
        ], dim=-1)  # (h, w, 4d)
        return emb.reshape(grid_h * grid_w, embed_dim)

    def _get_pos_embed(self, h: int, w: int, device, dtype) -> torch.Tensor:
        s = self.pos_embed_max_size
        if h == s and w == s:
            return self.pos_embed.to(device=device, dtype=dtype)
        pos = self.pos_embed.reshape(1, s, s, -1).permute(0, 3, 1, 2)  # (1, D, s, s)
        pos = F.interpolate(pos.float(), size=(h, w), mode="bilinear", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return pos.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) pixel image in [-1, 1]
        _, _, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p
        # Stage 1: patchify + bottleneck → (B, bottleneck_dim, h, w)
        x = self.patch_proj(x.to(self.patch_proj.weight.dtype))
        x = x.flatten(2).transpose(1, 2)   # (B, h*w, bottleneck_dim)
        # Stage 2: expand to transformer dim
        x = self.expand_proj(x)             # (B, h*w, embed_dim)
        return x + self._get_pos_embed(h, w, x.device, x.dtype)


class PomSD3Transformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, SD3Transformer2DLoadersMixin
):
    """SD3.5 transformer with PoM and/or windowed local attention replacing full attention.

    Block assignment is controlled by hybrid_n:
      hybrid_n=1 (default): all blocks are JointPoMBlock (full-PoM model).
          n_pom_blocks can still be used for progressive replacement.
      hybrid_n=0: all blocks are JointLocalAttnBlock (full local-attention model).
      hybrid_n=k (k≥2): block i is PoM if i%k==0, else JointLocalAttnBlock.

    Non-attention weights (patch embed, time/text embeddings, norms, FF layers,
    output projection) are identical to SD3Transformer2DModel and load directly
    from a pretrained SD3.5 checkpoint.  JointLocalAttnBlock also keeps the full
    set of attention projection weights (to_q/k/v etc.) from the checkpoint.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["JointPoMBlock", "JointTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1536,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 384,
        dual_attention_layers: tuple[int, ...] = (),
        qk_norm: str | None = None,
        # PoM-specific hyperparameters
        pom_degree: int = 4,
        pom_expand: int = 2,
        pom_n_groups: int = 1,
        pom_n_sel_heads: int = 24,
        lora_rank: int = 0,
        pom_rope_max_seq_len: int = 8192,
        # Progressive replacement (legacy, used only when hybrid_n==1):
        # last n_pom_blocks blocks are JointPoMBlock,
        # first (num_layers - n_pom_blocks) blocks are JointTransformerBlock.
        n_pom_blocks: int | None = None,
        # Hybrid mode
        hybrid_n: int = 1,
        attention_window_m: int = 4,
        # Explicit per-block layout (overrides hybrid_n when non-empty):
        # pom_layers lists the block indices that should be JointPoMBlock;
        # all other indices become JointTransformerBlock.
        pom_layers: tuple[int, ...] = (),
        # Pixel-space mode: when set, replaces PatchEmbed with PixelPatchEmbed
        # (two-stage linear bottleneck; no VAE required).
        pixel_patch_bottleneck_dim: int | None = None,
    ):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        if pixel_patch_bottleneck_dim is not None:
            self.pos_embed = PixelPatchEmbed(
                patch_size=patch_size,
                in_channels=in_channels,
                bottleneck_dim=pixel_patch_bottleneck_dim,
                embed_dim=self.inner_dim,
                pos_embed_max_size=pos_embed_max_size,
            )
        else:
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=self.inner_dim,
                pos_embed_max_size=pos_embed_max_size,
            )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)

        pom_kwargs = dict(
            pom_degree=pom_degree,
            pom_expand=pom_expand,
            pom_n_groups=pom_n_groups,
            pom_n_sel_heads=pom_n_sel_heads,
            lora_rank=lora_rank,
            pom_rope_max_seq_len=pom_rope_max_seq_len,
        )
        local_kwargs = dict(attention_window_m=attention_window_m)

        pom_layers_set = set(pom_layers)

        def _make_block(i: int) -> nn.Module:
            common = dict(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                context_pre_only=(i == num_layers - 1),
                qk_norm=qk_norm,
                use_dual_attention=(i in dual_attention_layers),
            )
            # Explicit layout takes priority over hybrid_n
            if pom_layers_set:
                if i in pom_layers_set:
                    return JointPoMBlock(**common, **pom_kwargs)
                return JointTransformerBlock(**common)
            if hybrid_n == 1:
                # Legacy behaviour: n_pom_blocks controls which blocks are PoM vs full-attn
                n_pom = num_layers if n_pom_blocks is None else n_pom_blocks
                n_attn = num_layers - n_pom
                if i < n_attn:
                    return JointTransformerBlock(**common)
                return JointPoMBlock(**common, **pom_kwargs)
            elif hybrid_n == 0:
                return JointTransformerBlock(**common)
            else:
                if i % hybrid_n == 0:
                    return JointPoMBlock(**common, **pom_kwargs)
                return JointTransformerBlock(**common)

        self.transformer_blocks = nn.ModuleList([_make_block(i) for i in range(num_layers)])

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True
        )
        if lora_rank > 0:
            out_features = patch_size * patch_size * self.out_channels
            self.proj_out_lora_A = nn.Linear(self.inner_dim, lora_rank, bias=False)
            self.proj_out_lora_B = nn.Linear(lora_rank, out_features, bias=False)
            nn.init.kaiming_uniform_(self.proj_out_lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj_out_lora_B.weight)
        else:
            self.proj_out_lora_A = self.proj_out_lora_B = None
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def merge_lora(self) -> None:
        """Merge LoRA weights into FF down-projections and remove LoRA parameters."""
        for blk in self.transformer_blocks:
            if getattr(blk, 'ff_lora_A', None) is not None:
                blk.ff.net[2].weight.data += blk.ff_lora_B.weight.data @ blk.ff_lora_A.weight.data
                delattr(blk, 'ff_lora_A')
                delattr(blk, 'ff_lora_B')
            if getattr(blk, 'ff_context_lora_A', None) is not None:
                blk.ff_context.net[2].weight.data += (
                    blk.ff_context_lora_B.weight.data @ blk.ff_context_lora_A.weight.data
                )
                delattr(blk, 'ff_context_lora_A')
                delattr(blk, 'ff_context_lora_B')
        if getattr(self, 'proj_out_lora_A', None) is not None:
            self.proj_out.weight.data += self.proj_out_lora_B.weight.data @ self.proj_out_lora_A.weight.data
            delattr(self, 'proj_out_lora_A')
            delattr(self, 'proj_out_lora_B')
        new_cfg = {**self._internal_dict, 'lora_rank': 0}
        object.__setattr__(self, '_internal_dict', self._internal_dict.__class__(new_cfg))

    def enable_forward_chunking(self, chunk_size: int | None = None, dim: int = 0) -> None:
        chunk_size = chunk_size or 1

        def _recurse(module, chunk_size, dim):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
            for child in module.children():
                _recurse(child, chunk_size, dim)

        for module in self.children():
            _recurse(module, chunk_size, dim)

    def disable_forward_chunking(self):
        def _recurse(module, chunk_size, dim):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
            for child in module.children():
                _recurse(child, chunk_size, dim)

        for module in self.children():
            _recurse(module, None, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: list = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        skip_layers: list[int] | None = None,
        return_intermediate: bool = False,
    ) -> torch.Tensor | Transformer2DModelOutput | tuple:
        """
        Same signature as SD3Transformer2DModel.forward with one addition:
            return_intermediate (bool): if True, also return a list of
                (encoder_hidden_states, hidden_states) after each block,
                used for distillation loss computation.
        """
        height, width = hidden_states.shape[-2:]

        # Cast to model dtype so the pipeline can pass float32 latents safely
        hidden_states = hidden_states.to(dtype=self.dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.dtype)
        pooled_projections = pooled_projections.to(dtype=self.dtype)

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        intermediates = [] if return_intermediate else None

        for index_block, block in enumerate(self.transformer_blocks):
            is_skip = skip_layers is not None and index_block in skip_layers

            if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                    use_reentrant=False,
                )
            elif not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if block_controlnet_hidden_states is not None and not block.context_pre_only:
                interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[
                    int(index_block / interval_control)
                ]

            if return_intermediate:
                intermediates.append((
                    encoder_hidden_states if encoder_hidden_states is not None else None,
                    hidden_states,
                ))

        hidden_states = self.norm_out(hidden_states, temb)
        proj_in = hidden_states
        hidden_states = self.proj_out(proj_in)
        proj_lora_A = getattr(self, 'proj_out_lora_A', None)
        if proj_lora_A is not None:
            hidden_states = hidden_states + self.proj_out_lora_B(proj_lora_A(proj_in))

        # Unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size
        )

        if return_intermediate:
            if not return_dict:
                return (output,), intermediates
            return Transformer2DModelOutput(sample=output), intermediates

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
