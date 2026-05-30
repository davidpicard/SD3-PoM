"""PoM-based transformer blocks that mirror SD3.5's JointTransformerBlock API."""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import FeedForward
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    SD35AdaLayerNormZeroX,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from pom import PoM


def _chunked_ff(ff: nn.Module, x: torch.Tensor, chunk_size: int, dim: int) -> torch.Tensor:
    return torch.cat([ff(hid) for hid in x.split(chunk_size, dim=dim)], dim=dim)


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
        self.pom = PoM(
            dim=dim,
            degree=pom_degree,
            expand=pom_expand,
            n_groups=pom_n_groups,
            n_sel_heads=pom_n_sel_heads,
        )

        # --- PoM: image-only dual mixing (replaces dual Attention attn2) ---
        if use_dual_attention:
            self.pom2 = PoM(
                dim=dim,
                degree=pom_degree,
                expand=pom_expand,
                n_groups=pom_n_groups,
                n_sel_heads=pom_n_sel_heads,
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
        joint_out = self.pom(joint)
        attn_output = joint_out[:, :n_img]
        context_attn_output = joint_out[:, n_img:]

        # --- Image residual from joint mixing ---
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        # --- Dual PoM: image-only secondary mixing ---
        if self.use_dual_attention:
            attn_output2 = self.pom2(norm_hidden_states2)
            hidden_states = hidden_states + gate_msa2.unsqueeze(1) * attn_output2

        # --- Image FF ---
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            ff_output = _chunked_ff(self.ff, norm_hidden_states, self._chunk_size, self._chunk_dim)
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
                context_ff_output = _chunked_ff(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_size, self._chunk_dim
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states
