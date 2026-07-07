"""PoM-based transformer blocks that mirror SD3.5's JointTransformerBlock API."""
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import JointAttnProcessor2_0
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    SD35AdaLayerNormZeroX,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from pom.pom_rope import PoMRoPE

from .masks import build_joint_mask

# ---------------------------------------------------------------------------
# flex_attention fast path for windowed local attention
# ---------------------------------------------------------------------------

try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attn,
        create_block_mask,
    )
    _FLEX_AVAILABLE = True
except ImportError:
    _FLEX_AVAILABLE = False

# BlockMask cache: (n_img, m, device_str) -> BlockMask
_BLOCK_MASK_CACHE: dict = {}
# BLOCK_SIZE=32 divides all typical patch counts: 256, 576, 1024, 2304, 4096
_FLEX_BLOCK_SIZE = 32


def _get_block_mask(n_img: int, m: int, device: torch.device):
    """Return a cached BlockMask for 2D windowed self-attention on image patches."""
    key = (n_img, m, str(device))
    if key in _BLOCK_MASK_CACHE:
        return _BLOCK_MASK_CACHE[key]
    W = int(math.isqrt(n_img))

    def mask_mod(b, h, q_idx, kv_idx):
        return (
            (torch.abs(q_idx // W - kv_idx // W) <= m) &
            (torch.abs(q_idx % W  - kv_idx % W)  <= m)
        )

    bm = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=n_img, KV_LEN=n_img,
        device=device, BLOCK_SIZE=_FLEX_BLOCK_SIZE,
    )
    _BLOCK_MASK_CACHE[key] = bm
    return bm


def _qkv(attn, norm_img, norm_txt, B, n_img, n_txt, H, d, context_pre_only: bool):
    """Project and RMS-norm Q/K/V for image and text modalities."""
    q_img = attn.to_q(norm_img).view(B, n_img, H, d).transpose(1, 2)
    k_img = attn.to_k(norm_img).view(B, n_img, H, d).transpose(1, 2)
    v_img = attn.to_v(norm_img).view(B, n_img, H, d).transpose(1, 2)
    k_txt = attn.add_k_proj(norm_txt).view(B, n_txt, H, d).transpose(1, 2)
    v_txt = attn.add_v_proj(norm_txt).view(B, n_txt, H, d).transpose(1, 2)
    q_txt = (None if context_pre_only
             else attn.add_q_proj(norm_txt).view(B, n_txt, H, d).transpose(1, 2))
    if attn.norm_q is not None:
        q_img = attn.norm_q(q_img)
        if q_txt is not None:
            q_txt = attn.norm_added_q(q_txt)
    if attn.norm_k is not None:
        k_img = attn.norm_k(k_img)
        k_txt = attn.norm_added_k(k_txt)
    return q_img, k_img, v_img, q_txt, k_txt, v_txt


def _merge_lse(out1, lse1, out2, lse2, dtype):
    """Combine two partial attention outputs via the online-softmax LSE formula.

    Equivalent to a single softmax over the union of both key sets.
    lse1, lse2: [B, H, N] float32.  out1, out2: [B, H, N, D].
    """
    m = torch.maximum(lse1, lse2)
    w1 = torch.exp(lse1 - m).unsqueeze(-1)   # [B, H, N, 1]
    w2 = torch.exp(lse2 - m).unsqueeze(-1)
    return ((out1.float() * w1 + out2.float() * w2) / (w1 + w2)).to(dtype)


def _joint_local_flex(attn, norm_img, norm_txt, n_img, n_txt, window_m, device, dtype):
    """Windowed joint attention using flex_attention + SDPA cross-attention.

    Image tokens: windowed img→img (flex_attn) + global img→txt (SDPA), merged via LSE.
    Text tokens : global Q_txt → [img, txt] (SDPA). Omitted when context_pre_only.
    Mathematically equivalent to joint attention with the windowed mask.
    """
    B = norm_img.shape[0]
    H, d = attn.heads, attn.inner_dim // attn.heads
    cpo = attn.context_pre_only

    q_img, k_img, v_img, q_txt, k_txt, v_txt = _qkv(
        attn, norm_img, norm_txt, B, n_img, n_txt, H, d, cpo
    )

    # --- Image → windowed image (flex_attn, O(N·w)) ---
    bm = _get_block_mask(n_img, window_m, device)
    out_local, lse_local = _flex_attn(q_img, k_img, v_img, block_mask=bm, return_lse=True)
    # out_local: [B, H, N_img, D];  lse_local: [B, H, N_img] float32

    # --- Image → all text (SDPA, N_txt=154 is small, no mask needed) ---
    scale = d ** -0.5
    cross = (q_img @ k_txt.transpose(-2, -1)) * scale          # [B, H, N_img, N_txt]
    lse_cross = torch.logsumexp(cross.float(), dim=-1)          # [B, H, N_img] float32
    out_cross = torch.softmax(cross, dim=-1) @ v_txt            # [B, H, N_img, D]

    # Merge: equivalent to softmax over windowed-img ∪ all-txt keys
    out_img = _merge_lse(out_local, lse_local, out_cross, lse_cross, dtype)

    # Output projection for image
    out_img = out_img.transpose(1, 2).reshape(B, n_img, H * d)
    out_img = attn.to_out[0](out_img)
    out_img = attn.to_out[1](out_img)

    if cpo:
        return out_img, None

    # --- Text → all (image + text), global SDPA ---
    k_all = torch.cat([k_img, k_txt], dim=2)   # [B, H, N_img+N_txt, D]
    v_all = torch.cat([v_img, v_txt], dim=2)
    out_txt = F.scaled_dot_product_attention(q_txt, k_all, v_all)
    out_txt = out_txt.transpose(1, 2).reshape(B, n_txt, H * d)
    out_txt = attn.to_add_out(out_txt)

    return out_img, out_txt


def _img_local_flex(attn, norm_img, n_img, window_m, device, dtype):
    """Windowed image-only attention via flex_attention (for dual-attention attn2)."""
    B = norm_img.shape[0]
    H, d = attn.heads, attn.inner_dim // attn.heads

    q = attn.to_q(norm_img).view(B, n_img, H, d).transpose(1, 2)
    k = attn.to_k(norm_img).view(B, n_img, H, d).transpose(1, 2)
    v = attn.to_v(norm_img).view(B, n_img, H, d).transpose(1, 2)
    if attn.norm_q is not None:
        q = attn.norm_q(q)
    if attn.norm_k is not None:
        k = attn.norm_k(k)

    bm = _get_block_mask(n_img, window_m, device)
    out = _flex_attn(q, k, v, block_mask=bm)
    out = out.transpose(1, 2).reshape(B, n_img, H * d)
    out = attn.to_out[0](out)
    out = attn.to_out[1](out)
    return out



@maybe_allow_in_graph
class JointPoMBlock(nn.Module):
    """Drop-in replacement for JointTransformerBlock using PoM instead of attention.

    All norm and FF layers are identical to the original so pretrained weights transfer
    directly. Only the Attention modules are replaced by PoM.

    Joint mixing: image and text tokens are concatenated, mixed by a single PoM, then
    split back — mirroring joint attention's concatenated K/V approach.
    Dual mixing: image-only PoM mirrors the dual self-attention in SD3.5.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: str | None = None,  # kept for API compatibility, unused
        use_dual_attention: bool = False,
        pom_degree: int = 4,
        pom_expand: int = 2,
        pom_n_groups: int = 1,
        pom_n_sel_heads: int = 1,
        lora_rank: int = 0,
        pom_rope_max_seq_len: int = 8192,
    ):
        super().__init__()
        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only

        # --- Norms (identical to JointTransformerBlock) ---
        if use_dual_attention:
            self.norm1 = SD35AdaLayerNormZeroX(dim)
        else:
            self.norm1 = AdaLayerNormZero(dim)

        if context_pre_only:
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
            )
        else:
            self.norm1_context = AdaLayerNormZero(dim)

        # --- PoM: joint image+text mixing (replaces joint Attention) ---
        self.pom = PoMRoPE(
            dim=dim,
            degree=pom_degree,
            expand=pom_expand,
            n_groups=pom_n_groups,
            n_sel_heads=pom_n_sel_heads,
            max_seq_len=pom_rope_max_seq_len,
            rope_2d=False,
        )

        # --- PoM: image-only dual mixing (replaces dual Attention attn2) ---
        if use_dual_attention:
            self.pom2 = PoMRoPE(
                dim=dim,
                degree=pom_degree,
                expand=pom_expand,
                n_groups=pom_n_groups,
                n_sel_heads=pom_n_sel_heads,
                max_seq_len=pom_rope_max_seq_len,
                rope_2d=False,
            )
        else:
            self.pom2 = None

        # --- FF layers (identical to JointTransformerBlock) ---
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None

        # --- LoRA on FF down-projections (applied at x_gate = GELU output, enables exact merge) ---
        if lora_rank > 0:
            inner_dim = self.ff.net[2].in_features  # dim * 4
            self.ff_lora_A = nn.Linear(inner_dim, lora_rank, bias=False)
            self.ff_lora_B = nn.Linear(lora_rank, dim, bias=False)
            nn.init.kaiming_uniform_(self.ff_lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.ff_lora_B.weight)

            if not context_pre_only:
                ctx_inner = self.ff_context.net[2].in_features
                self.ff_context_lora_A = nn.Linear(ctx_inner, lora_rank, bias=False)
                self.ff_context_lora_B = nn.Linear(lora_rank, dim, bias=False)
                nn.init.kaiming_uniform_(self.ff_context_lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.ff_context_lora_B.weight)
            else:
                self.ff_context_lora_A = self.ff_context_lora_B = None
        else:
            self.ff_lora_A = self.ff_lora_B = None
            self.ff_context_lora_A = self.ff_context_lora_B = None

        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: int | None, dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        joint_attention_kwargs: dict[str, Any] | None = None,  # unused, kept for API compat
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_img = hidden_states.shape[1]
        n_txt = encoder_hidden_states.shape[1]
        dev = hidden_states.device

        # 1-D RoPE positions: image tokens at 0..n_img-1 (row-major patch order),
        # text tokens at n_img..n_img+n_txt-1 (no position overlap between modalities)
        img_positions = torch.arange(n_img, device=dev, dtype=torch.int64)
        joint_positions = torch.cat([
            img_positions,
            torch.arange(n_img, n_img + n_txt, device=dev, dtype=torch.int64),
        ])

        # --- Normalize image tokens ---
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = (
                self.norm1(hidden_states, emb=temb)
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, emb=temb
            )

        # --- Normalize text tokens ---
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
                self.norm1_context(encoder_hidden_states, emb=temb)
            )

        # --- Joint PoM: mix image + text tokens together ---
        joint = torch.cat([norm_hidden_states, norm_encoder_hidden_states], dim=1)
        joint_out = self.pom(joint, positions=joint_positions)
        attn_output = joint_out[:, :n_img]
        context_attn_output = joint_out[:, n_img:]

        # --- Image residual from joint mixing ---
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        # --- Dual PoM: image-only secondary mixing ---
        if self.use_dual_attention:
            attn_output2 = self.pom2(norm_hidden_states2, positions=img_positions)
            hidden_states = hidden_states + gate_msa2.unsqueeze(1) * attn_output2

        # --- Image FF ---
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_lora_A = getattr(self, 'ff_lora_A', None)
        if self._chunk_size is not None:
            def _ff_with_lora(chunk):
                x_gate = self.ff.net[1](self.ff.net[0](chunk))
                out = self.ff.net[2](x_gate)
                if ff_lora_A is not None:
                    out = out + self.ff_lora_B(ff_lora_A(x_gate))
                return out
            ff_output = torch.cat(
                [_ff_with_lora(h) for h in norm_hidden_states.split(self._chunk_size, dim=self._chunk_dim)],
                dim=self._chunk_dim,
            )
        else:
            x_gate = self.ff.net[1](self.ff.net[0](norm_hidden_states))
            ff_output = self.ff.net[2](x_gate)
            if ff_lora_A is not None:
                ff_output = ff_output + self.ff_lora_B(ff_lora_A(x_gate))
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        # --- Text residual and FF ---
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = (
                norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            )
            ff_context_lora_A = getattr(self, 'ff_context_lora_A', None)
            if self._chunk_size is not None:
                def _ctx_ff_with_lora(chunk):
                    x_gate = self.ff_context.net[1](self.ff_context.net[0](chunk))
                    out = self.ff_context.net[2](x_gate)
                    if ff_context_lora_A is not None:
                        out = out + self.ff_context_lora_B(ff_context_lora_A(x_gate))
                    return out
                context_ff_output = torch.cat(
                    [_ctx_ff_with_lora(h) for h in norm_encoder_hidden_states.split(self._chunk_size, dim=self._chunk_dim)],
                    dim=self._chunk_dim,
                )
            else:
                x_gate = self.ff_context.net[1](self.ff_context.net[0](norm_encoder_hidden_states))
                context_ff_output = self.ff_context.net[2](x_gate)
                if ff_context_lora_A is not None:
                    context_ff_output = context_ff_output + self.ff_context_lora_B(ff_context_lora_A(x_gate))
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states


# ---------------------------------------------------------------------------
# Local windowed attention
# ---------------------------------------------------------------------------

class JointLocalAttnProcessor:
    """Like JointAttnProcessor2_0 but passes attention_mask to SDPA."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor | tuple[torch.FloatTensor, torch.FloatTensor]:
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key  .view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)
            if not attn.context_pre_only:
                enc_q = attn.add_q_proj(encoder_hidden_states)
                enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                if attn.norm_added_q is not None:
                    enc_q = attn.norm_added_q(enc_q)
                query = torch.cat([query, enc_q], dim=2)
            key   = torch.cat([key,   enc_k], dim=2)
            value = torch.cat([value, enc_v], dim=2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


@maybe_allow_in_graph
class JointLocalAttnBlock(nn.Module):
    """Drop-in replacement for JointTransformerBlock with 2D-windowed local attention.

    Submodule names are identical to JointTransformerBlock so SD3.5 pretrained
    weights (including QKV projections) load without key remapping.

    Image tokens attend only to neighbours within a (2m+1)×(2m+1) patch window
    (row-major 1-D sequence assumed to be a square grid).  Text tokens attend
    globally to all tokens.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: str | None = None,
        use_dual_attention: bool = False,
        attention_window_m: int = 4,
    ):
        super().__init__()
        self.context_pre_only = context_pre_only
        self.use_dual_attention = use_dual_attention
        self.window_m = attention_window_m

        _proc = JointLocalAttnProcessor()

        # --- Norms (identical to JointTransformerBlock) ---
        if use_dual_attention:
            self.norm1 = SD35AdaLayerNormZeroX(dim)
        else:
            self.norm1 = AdaLayerNormZero(dim)

        if context_pre_only:
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
            )
        else:
            self.norm1_context = AdaLayerNormZero(dim)

        # --- Attention (same Attention class, custom processor) ---
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=_proc,
            qk_norm=qk_norm,
            eps=1e-6,
        )

        if use_dual_attention:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                processor=_proc,
                qk_norm=qk_norm,
                eps=1e-6,
            )
        else:
            self.attn2 = None

        # --- FF layers (identical to JointTransformerBlock) ---
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None

        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: int | None, dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def _get_masks(
        self, n_img: int, n_txt: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return build_joint_mask(n_img, n_txt, self.window_m, device, dtype)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_img = hidden_states.shape[1]
        n_txt = encoder_hidden_states.shape[1]
        device, dtype = hidden_states.device, hidden_states.dtype

        # Masks only needed for the SDPA fallback path.
        if not _FLEX_AVAILABLE:
            joint_mask, img_mask = self._get_masks(n_img, n_txt, device, dtype)
            # When context_pre_only only image tokens are queries → slice mask rows.
            if self.context_pre_only:
                joint_mask = joint_mask[:, :, :n_img, :]

        # --- Normalize ---
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = (
                self.norm1(hidden_states, emb=temb)
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, emb=temb
            )

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
                self.norm1_context(encoder_hidden_states, emb=temb)
            )

        # --- Joint local attention ---
        if _FLEX_AVAILABLE:
            attn_output, context_attn_output = _joint_local_flex(
                self.attn, norm_hidden_states, norm_encoder_hidden_states,
                n_img, n_txt, self.window_m, device, dtype,
            )
        else:
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                attention_mask=joint_mask,
            )
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        # --- Dual image-only local attention ---
        if self.use_dual_attention:
            if _FLEX_AVAILABLE:
                attn_output2 = _img_local_flex(
                    self.attn2, norm_hidden_states2, n_img, self.window_m, device, dtype,
                )
            else:
                attn_output2 = self.attn2(
                    hidden_states=norm_hidden_states2,
                    attention_mask=img_mask,
                )
            hidden_states = hidden_states + gate_msa2.unsqueeze(1) * attn_output2

        # --- Image FF ---
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            from diffusers.models.attention import _chunked_feed_forward
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        # --- Text residual and FF ---
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = (
                norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            )
            if self._chunk_size is not None:
                from diffusers.models.attention import _chunked_feed_forward
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states
