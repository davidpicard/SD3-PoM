"""Utilities for initializing PomSD3Transformer2DModel from a pretrained SD3.5 checkpoint."""
import re

import torch
from diffusers import SD3Transformer2DModel
from diffusers.models.attention import JointTransformerBlock

from .blocks import JointPoMBlock
from .model import PomSD3Transformer2DModel

# SD3.5 Medium config
SD35_MEDIUM_CONFIG = dict(
    sample_size=128,
    patch_size=2,
    in_channels=16,
    num_layers=24,
    attention_head_dim=64,
    num_attention_heads=24,
    joint_attention_dim=4096,
    caption_projection_dim=1536,
    pooled_projection_dim=2048,
    out_channels=16,
    pos_embed_max_size=384,
    dual_attention_layers=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    qk_norm="rms_norm",
)

_ATTENTION_KEY_RE = re.compile(r"transformer_blocks\.(\d+)\.attn2?\.")
_BLOCK_IDX_RE = re.compile(r"transformer_blocks\.(\d+)\.")


def _is_attention_key(key: str) -> bool:
    return bool(_ATTENTION_KEY_RE.search(key))


def _is_pom_block_attention_key(key: str, n_pom_blocks: int, num_layers: int) -> bool:
    """True only for attention keys that belong to a PoM block (should be skipped on load)."""
    m = _BLOCK_IDX_RE.match(key)
    if m is None:
        return False
    block_idx = int(m.group(1))
    is_pom_block = block_idx >= num_layers - n_pom_blocks
    return is_pom_block and bool(_ATTENTION_KEY_RE.search(key))


def load_sd3_weights_into_pom(
    student: PomSD3Transformer2DModel,
    teacher_state_dict: dict[str, torch.Tensor],
    strict_non_attention: bool = True,
) -> tuple[list[str], list[str]]:
    """Copy weights from teacher_state_dict into student.

    For PoM blocks: attention keys are skipped (PoM has no attention).
    For JointTransformerBlock blocks (n_pom_blocks < num_layers): all keys including
    attention are loaded so those blocks are exact copies of the teacher.

    Returns (missing_keys, unexpected_keys).
    """
    n_pom = getattr(student.config, "n_pom_blocks", None)
    num_layers = student.config.num_layers
    if n_pom is None:
        n_pom = num_layers  # all blocks are PoM (legacy behaviour)

    student_dict = student.state_dict()
    transfer = {}
    shape_mismatches = []

    for key, val in teacher_state_dict.items():
        if _is_pom_block_attention_key(key, n_pom, num_layers):
            continue  # PoM blocks have no attention weights
        if key not in student_dict:
            continue
        if student_dict[key].shape != val.shape:
            shape_mismatches.append((key, student_dict[key].shape, val.shape))
            continue
        transfer[key] = val

    if shape_mismatches:
        raise RuntimeError(
            f"Shape mismatches for non-attention keys (check model configs match):\n"
            + "\n".join(f"  {k}: student={s} teacher={t}" for k, s, t in shape_mismatches)
        )

    missing, unexpected = student.load_state_dict(transfer, strict=False)

    # Filter: missing keys that are NOT PoM-specific, LoRA, or attention blocks are a problem
    pom_key_fragments = (".pom.", ".pom2.", ".ff_lora_", ".ff_context_lora_", "proj_out_lora_")
    non_pom_missing = [
        k for k in missing
        if not any(f in k for f in pom_key_fragments)
        and not _is_pom_block_attention_key(k, n_pom, num_layers)
    ]
    if strict_non_attention and non_pom_missing:
        raise RuntimeError(
            f"Non-PoM keys missing after weight transfer (suggests config mismatch):\n"
            + "\n".join(f"  {k}" for k in non_pom_missing)
        )

    return missing, unexpected


def build_from_sd3_pretrained(
    model_id: str = "stabilityai/stable-diffusion-3.5-medium",
    pom_degree: int = 4,
    pom_expand: int = 2,
    pom_n_groups: int = 1,
    pom_n_sel_heads: int = 24,
    lora_rank: int = 16,
    n_pom_blocks: int | None = None,
    pom_rope_max_seq_len: int = 8192,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> PomSD3Transformer2DModel:
    """Load SD3.5 transformer weights into a fresh PomSD3Transformer2DModel.

    When n_pom_blocks < num_layers, the first (num_layers - n_pom_blocks) blocks are
    JointTransformerBlock with all weights (including attention) loaded from the pretrained
    checkpoint. The last n_pom_blocks blocks are JointPoMBlock with attention weights skipped.
    """
    from pathlib import Path
    local_files_only = Path(model_id).exists()

    print(f"Loading teacher transformer from {model_id} ...")
    teacher = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    teacher_sd = teacher.state_dict()
    del teacher

    num_layers = SD35_MEDIUM_CONFIG["num_layers"]
    n_pom = num_layers if n_pom_blocks is None else n_pom_blocks

    pom_config = dict(
        pom_degree=pom_degree,
        pom_expand=pom_expand,
        pom_n_groups=pom_n_groups,
        pom_n_sel_heads=pom_n_sel_heads,
        lora_rank=lora_rank,
        n_pom_blocks=n_pom,
        pom_rope_max_seq_len=pom_rope_max_seq_len,
    )
    student = PomSD3Transformer2DModel(**SD35_MEDIUM_CONFIG, **pom_config)
    student = student.to(dtype=torch_dtype)

    print(f"Transferring weights ({n_pom}/{num_layers} PoM blocks, {num_layers - n_pom} attention blocks) ...")
    missing, _ = load_sd3_weights_into_pom(student, teacher_sd)
    pom_keys = [k for k in missing if any(f in k for f in (".pom.", ".pom2."))]
    print(f"  PoM-specific params randomly initialized: {len(pom_keys)} keys")

    return student.to(device)


def replace_next_attention_block(model: PomSD3Transformer2DModel) -> JointPoMBlock:
    """Replace the next attention block (from the end) with a JointPoMBlock.

    Updates model.transformer_blocks in-place, transfers FF/norm weights from the
    outgoing JointTransformerBlock, and bumps n_pom_blocks in the config.

    Returns the new block so the caller can register its params with the optimizer.
    """
    n_pom = model.config.n_pom_blocks if model.config.n_pom_blocks is not None else model.config.num_layers
    num_layers = model.config.num_layers
    if n_pom >= num_layers:
        raise ValueError("All blocks are already PoM — nothing to replace")

    block_idx = num_layers - n_pom - 1
    old_block = model.transformer_blocks[block_idx]
    if not isinstance(old_block, JointTransformerBlock):
        raise TypeError(f"Block {block_idx} is already a JointPoMBlock")

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    new_block = JointPoMBlock(
        dim=model.inner_dim,
        num_attention_heads=model.config.num_attention_heads,
        attention_head_dim=model.config.attention_head_dim,
        context_pre_only=(block_idx == num_layers - 1),
        qk_norm=model.config.qk_norm,
        use_dual_attention=(block_idx in model.config.dual_attention_layers),
        pom_degree=model.config.pom_degree,
        pom_expand=model.config.pom_expand,
        pom_n_groups=model.config.pom_n_groups,
        pom_n_sel_heads=model.config.pom_n_sel_heads,
        lora_rank=model.config.lora_rank,
        pom_rope_max_seq_len=model.config.pom_rope_max_seq_len,
    ).to(dtype=dtype, device=device)

    # Transfer FF and norm weights; PoM layers stay randomly initialized
    for name in ("norm1", "norm1_context", "ff", "ff_context", "norm2", "norm2_context"):
        if hasattr(old_block, name) and hasattr(new_block, name):
            getattr(new_block, name).load_state_dict(
                getattr(old_block, name).state_dict(), strict=False
            )

    model.transformer_blocks[block_idx] = new_block

    new_cfg = {**model._internal_dict, "n_pom_blocks": n_pom + 1}
    object.__setattr__(model, "_internal_dict", model._internal_dict.__class__(new_cfg))

    return new_block
