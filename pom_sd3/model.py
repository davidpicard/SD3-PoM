"""PomSD3Transformer2DModel: SD3.5 transformer with PoM instead of attention."""
import math
from typing import Any

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin, SD3Transformer2DLoadersMixin
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import logging

from .blocks import JointPoMBlock

logger = logging.get_logger(__name__)


class PomSD3Transformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, SD3Transformer2DLoadersMixin
):
    """SD3.5 transformer with all attention layers replaced by PoM.

    Non-attention weights (patch embed, time/text embeddings, norms, FF layers,
    output projection) are identical to SD3Transformer2DModel and can be loaded
    directly from a pretrained SD3.5 checkpoint via `from_sd3_pretrained`.

    The forward signature is identical to SD3Transformer2DModel so this model
    can be plugged into StableDiffusion3Pipeline as a drop-in replacement.
    Pass `return_intermediate=True` during distillation training to also get
    a list of per-block (encoder_hidden_states, hidden_states) outputs.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["JointPoMBlock"]
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
        # Progressive replacement: last n_pom_blocks blocks are JointPoMBlock,
        # first (num_layers - n_pom_blocks) blocks are JointTransformerBlock (frozen attention).
        n_pom_blocks: int | None = None,
    ):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

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

        n_pom = num_layers if n_pom_blocks is None else n_pom_blocks
        n_attn = num_layers - n_pom  # first n_attn blocks stay as JointTransformerBlock

        pom_kwargs = dict(
            pom_degree=pom_degree,
            pom_expand=pom_expand,
            pom_n_groups=pom_n_groups,
            pom_n_sel_heads=pom_n_sel_heads,
            lora_rank=lora_rank,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=(i == num_layers - 1),
                    qk_norm=qk_norm,
                    use_dual_attention=(i in dual_attention_layers),
                ) if i < n_attn else
                JointPoMBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=(i == num_layers - 1),
                    qk_norm=qk_norm,
                    use_dual_attention=(i in dual_attention_layers),
                    **pom_kwargs,
                )
                for i in range(num_layers)
            ]
        )

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

    def merge_lora(self) -> None:
        """Merge LoRA weights into FF down-projections and remove LoRA parameters.

        After this call lora_rank is effectively 0: no extra memory, no extra
        computation. save_pretrained() produces a checkpoint without LoRA keys.
        """
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
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
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
