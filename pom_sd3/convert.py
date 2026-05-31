"""Utilities for initializing PomSD3Transformer2DModel from a pretrained SD3.5 checkpoint."""
import re

import torch
from diffusers import SD3Transformer2DModel

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

_ATTENTION_KEY_RE = re.compile(r"transformer_blocks\.\d+\.attn2?\.")


def _is_attention_key(key: str) -> bool:
    return bool(_ATTENTION_KEY_RE.search(key))


def load_sd3_weights_into_pom(
    student: PomSD3Transformer2DModel,
    teacher_state_dict: dict[str, torch.Tensor],
    strict_non_attention: bool = True,
) -> tuple[list[str], list[str]]:
    """Copy all non-attention weights from teacher_state_dict into student.

    Returns (missing_keys, unexpected_keys) — only PoM-specific keys should
    appear as missing from the teacher, and attention keys should be the only
    unexpected keys not present in the student.
    """
    student_dict = student.state_dict()
    transfer = {}
    skipped_attention = []
    shape_mismatches = []

    for key, val in teacher_state_dict.items():
        if _is_attention_key(key):
            skipped_attention.append(key)
            continue
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

    # Filter: missing keys that are NOT PoM-specific or LoRA are a problem
    pom_key_fragments = (".pom.", ".pom2.", ".ff_lora_", ".ff_context_lora_", "proj_out_lora_")
    non_pom_missing = [k for k in missing if not any(f in k for f in pom_key_fragments)]
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
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> PomSD3Transformer2DModel:
    """Load SD3.5 transformer weights into a fresh PomSD3Transformer2DModel.

    Non-attention weights are copied from the pretrained checkpoint; PoM
    layers are randomly initialized.
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
    del teacher  # free memory immediately

    pom_config = dict(
        pom_degree=pom_degree,
        pom_expand=pom_expand,
        pom_n_groups=pom_n_groups,
        pom_n_sel_heads=pom_n_sel_heads,
    )
    student = PomSD3Transformer2DModel(**SD35_MEDIUM_CONFIG, **pom_config)
    student = student.to(dtype=torch_dtype)

    print("Transferring non-attention weights ...")
    missing, _ = load_sd3_weights_into_pom(student, teacher_sd)
    pom_keys = [k for k in missing if any(f in k for f in (".pom.", ".pom2."))]
    print(f"  Transferred weights — PoM-specific params randomly initialized: {len(pom_keys)} keys")

    return student.to(device)
