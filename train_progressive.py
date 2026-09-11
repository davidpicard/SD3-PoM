"""Progressive PoM replacement: start from pretrained SD3.5, activate one
block at a time from the end, keeping every 4th block as attention.

Architecture (24 blocks):
  att(0), pom(1..3), att(4), pom(5..7), att(8), pom(9..11),
  att(12), pom(13..15), att(16), pom(17..19), att(20), pom(21..23)

Activation order (end → front): 23, 22, 21, 20, 19, ..., 1, 0
Every --phase_steps training steps, the next block is unfrozen:
  - PoM block: randomly initialised; starts contributing when activated.
  - Att block: pretrained SD3.5 weights; adapts at --pretrained_lr_scale × lr.

Launch:
    torchrun --nproc_per_node=4 --nnodes=2 ... train_progressive.py ...
"""
import argparse
import collections
import contextlib
import functools
import json
import math
import os
import re
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from safetensors.torch import save_file as safetensors_save_file

import wandb
from diffusers import AutoencoderKL, SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.models.attention import JointTransformerBlock
from torchvision import transforms

from pom_sd3 import PomSD3Transformer2DModel
from pom_sd3.blocks import JointPoMBlock

# Re-use helpers and data loading from train_scratch.py
from train_scratch import (
    _silence_encoding_noise,
    fast_encode_prompt,
    setup_ddp,
    cleanup_ddp,
    is_main,
    GPicDataset,
    gpic_collate,
    generate_samples,
    load_val_cache,
    run_validation,
    find_latest_checkpoint,
    save_checkpoint,
    load_optimizer_fsdp,
    load_checkpoint_optimizer,
    wrap_model_fsdp,
    lr_schedule,
    print_model_summary,
    print_model_layers,
    _VAL_SIGMAS,
    _VAL_T_INTS,
    SAMPLE_PROMPTS,
)


# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------

ATT_KEEP = frozenset(range(0, 24, 4))          # {0, 4, 8, 12, 16, 20}
POM_LAYERS = tuple(i for i in range(24) if i not in ATT_KEEP)
# Activation order: unfreeze from the end toward the front
ACTIVATION_ORDER = list(range(23, -1, -1))      # [23, 22, 21, ..., 0]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_progressive_model(
    model_id: str,
    pom_degree: int,
    pom_expand: int,
    pom_n_groups: int,
    pom_n_sel_heads: int,
    pom_rope_max_seq_len: int,
    torch_dtype: torch.dtype,
    device,
) -> PomSD3Transformer2DModel:
    """Build interleaved att+PoM model, weights from pretrained SD3.5.

    Att blocks (0,4,8,12,16,20): all weights loaded from checkpoint.
    PoM blocks: FF/norm weights loaded; attention replaced with random PoM.
    """
    _local = Path(model_id).exists()
    model = PomSD3Transformer2DModel(
        sample_size=128, patch_size=2, in_channels=16, num_layers=24,
        attention_head_dim=64, num_attention_heads=24,
        joint_attention_dim=4096, caption_projection_dim=1536,
        pooled_projection_dim=2048, out_channels=16, pos_embed_max_size=384,
        dual_attention_layers=tuple(range(13)),   # 0..12, same as SD3.5 Medium
        pom_layers=POM_LAYERS,
        qk_norm="rms_norm",
        pom_degree=pom_degree,
        pom_expand=pom_expand,
        pom_n_groups=pom_n_groups,
        pom_n_sel_heads=pom_n_sel_heads,
        pom_rope_max_seq_len=pom_rope_max_seq_len,
        lora_rank=0,
    ).to(dtype=torch_dtype)

    if is_main():
        print(f"Loading pretrained SD3.5 weights from {model_id} ...")
    sd = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer",
        torch_dtype=torch_dtype, local_files_only=_local,
    ).state_dict()

    student_sd = model.state_dict()
    to_load: dict = {}
    attn_re = re.compile(r'transformer_blocks\.(\d+)\.attn')

    for key, val in sd.items():
        m = re.match(r'transformer_blocks\.(\d+)\.', key)
        if m:
            blk = int(m.group(1))
            if blk not in ATT_KEEP and attn_re.search(key):
                continue   # PoM block: skip attention weights
        if key in student_sd and student_sd[key].shape == val.shape:
            to_load[key] = val

    missing, unexpected = model.load_state_dict(to_load, strict=False)
    if is_main():
        pom_missing = [k for k in missing if 'pom' not in k and 'attn' not in k]
        if pom_missing:
            print(f"  WARNING: unexpected non-PoM missing keys: {pom_missing[:5]}")
        n_loaded = sum(v.numel() for k, v in to_load.items())
        n_total  = sum(p.numel() for p in model.parameters())
        print(f"  Loaded {n_loaded/1e6:.0f}M / {n_total/1e6:.0f}M params from pretrained.")

    # AdaLN-Zero for all PoM blocks: zero norm1/norm1_context linear projections
    # so each block starts as an identity map (gate=0 → zero residual contribution).
    # Frozen blocks receive no optimizer updates, so their gates stay near-zero until
    # activated. This eliminates the random-signal spike that otherwise corrupts the
    # residual stream when a new block first enters the optimizer.
    import torch.nn as nn
    for i, blk in enumerate(model.transformer_blocks):
        if i not in ATT_KEEP:
            for attr in ("norm1", "norm1_context"):
                norm = getattr(blk, attr, None)
                if norm is not None and hasattr(norm, "linear"):
                    nn.init.zeros_(norm.linear.weight)
                    if norm.linear.bias is not None:
                        nn.init.zeros_(norm.linear.bias)

    return model.to(device)


# ---------------------------------------------------------------------------
# Progressive activation
# ---------------------------------------------------------------------------

def block_params(model: torch.nn.Module, block_idx: int) -> list:
    """Return parameters of transformer_blocks[block_idx] (handles FSDP wrapper)."""
    inner = getattr(model, '_fsdp_wrapped_module', model)
    return list(inner.transformer_blocks[block_idx].parameters())


def activate_block(model, optimizer, block_idx: int, lr: float, is_att: bool,
                   pretrained_lr_scale: float, activated_at: int = 0) -> None:
    """Add one block's params to the optimizer (all params already have requires_grad=True)."""
    params = block_params(model, block_idx)
    group_lr = lr * pretrained_lr_scale if is_att else lr
    optimizer.add_param_group({
        "params": params,
        "lr": group_lr,
        "block_idx": block_idx,
        "is_att": is_att,
        "activated_at": activated_at,
    })
    kind = f"att (lr×{pretrained_lr_scale})" if is_att else "PoM"
    n = sum(p.numel() for p in params)
    if is_main():
        print(f"  Activated block {block_idx:2d} ({kind}): {n/1e6:.1f}M params "
              f"@ lr={group_lr:.2e}")


def replay_activations(model, optimizer, phases_done: int, lr: float,
                       pretrained_lr_scale: float, phase_steps: int = 0) -> None:
    """Re-apply activation history when resuming from checkpoint.

    On resume we don't know exact activation times, so set activated_at so that
    the per-block warmup is already fully ramped (use negative offsets).
    """
    for i in range(phases_done):
        block_idx = ACTIVATION_ORDER[i]
        is_att = block_idx in ATT_KEEP
        # Estimate each block was activated at i * phase_steps; mark fully warmed
        # by setting activated_at far in the past relative to current step.
        estimated_at = i * phase_steps
        activate_block(model, optimizer, block_idx, lr, is_att, pretrained_lr_scale,
                       activated_at=estimated_at)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--init_from", default=None,
                   help="Load weights from checkpoint, fresh optimizer/step")

    # Dataset
    p.add_argument("--dataset_name", default="stanford-vision-lab/gpic")
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--caption_type", default="all")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)

    # PoM architecture
    p.add_argument("--pom_degree", type=int, default=4)
    p.add_argument("--pom_expand", type=int, default=2)
    p.add_argument("--pom_n_groups", type=int, default=1)
    p.add_argument("--pom_n_sel_heads", type=int, default=24)
    p.add_argument("--pom_rope_max_seq_len", type=int, default=8192)

    # Progressive replacement
    p.add_argument("--phase_steps", type=int, default=10_000,
                   help="Maximum steps per phase; also used as minimum when plateau detection fires")
    p.add_argument("--pretrained_lr_scale", type=float, default=0.1,
                   help="LR multiplier for unfrozen att blocks (smaller = gentler adaptation)")
    p.add_argument("--block_warmup_steps", type=int, default=1_000,
                   help="Per-block LR warmup steps after activation (ramps 0→full LR)")
    p.add_argument("--min_phase_steps", type=int, default=None,
                   help="Minimum steps before plateau check triggers early phase advance "
                        "(default: phase_steps // 2)")
    p.add_argument("--plateau_threshold", type=float, default=0.005,
                   help="Relative loss improvement below which a phase is considered converged")
    p.add_argument("--plateau_window", type=int, default=2_000,
                   help="Steps over which to measure loss improvement for plateau detection")
    p.add_argument("--consolidation_lr_scale", type=float, default=0.5,
                   help="LR multiplier applied to ALL groups once all blocks are activated")

    # Training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--caption_dropout", type=float, default=0.1)
    p.add_argument("--max_sequence_length", type=int, default=77)
    p.add_argument("--crop_str_dropout", type=float, default=0.1)
    p.add_argument("--logit_normal_mean", type=float, default=0.0)
    p.add_argument("--logit_normal_std", type=float, default=0.8)
    p.add_argument("--max_steps", type=int, default=500_000)
    p.add_argument("--warmup_steps", type=int, default=2_000)

    # FSDP
    p.add_argument("--gpus_per_node", type=int, default=None)

    # Logging / checkpointing
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--sample_every", type=int, default=20_000)
    p.add_argument("--val_every", type=int, default=10_000)
    p.add_argument("--n_val_images", type=int, default=256)
    p.add_argument("--num_sample_prompts", type=int, default=25)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--snr_gamma", type=float, default=5.0,
                   help="Min-SNR loss weighting gamma (0 = disabled)")
    p.add_argument("--ema_decay", type=float, default=0.9999,
                   help="EMA decay for model weights used in sampling/validation (0 = disabled)")
    p.add_argument("--wandb_project", default="sd3-pom-progressive")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--smoke_test", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    out_dir = Path(args.output_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            mode="offline" if args.wandb_offline else "online",
            settings=wandb.Settings(console="off"),
        )

    # --- Resolve checkpoint paths ---
    resume_dir: Path | None = None
    init_dir: Path | None = None
    _resume_last_phase_step: int = 0
    if args.resume_from:
        resume_dir = Path(args.resume_from)
    elif args.resume:
        resume_dir = find_latest_checkpoint(out_dir)
        if resume_dir is None and args.init_from:
            init_dir = Path(args.init_from)
    elif args.init_from:
        init_dir = Path(args.init_from)

    # --- VAE ---
    if not args.smoke_test:
        print(f"[rank {rank}] Loading VAE ...")
        _local = Path(args.model_id).exists()
        vae = AutoencoderKL.from_pretrained(
            args.model_id, subfolder="vae",
            torch_dtype=torch.bfloat16, local_files_only=_local,
        ).to(device)
        for p in vae.parameters():
            p.requires_grad_(False)
        vae.eval()
        vae = torch.compile(vae, dynamic=False)
    else:
        vae = None

    # --- Text encoders ---
    if not args.smoke_test:
        print(f"[rank {rank}] Loading text encoders ...")
        text_pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model_id, transformer=None, vae=None,
            torch_dtype=torch.bfloat16, local_files_only=_local,
        ).to(device)
        for enc in (text_pipe.text_encoder, text_pipe.text_encoder_2, text_pipe.text_encoder_3):
            if enc is not None:
                enc.requires_grad_(False)
        if text_pipe.text_encoder is not None:
            text_pipe.text_encoder = torch.compile(text_pipe.text_encoder, dynamic=True)
        if text_pipe.text_encoder_2 is not None:
            text_pipe.text_encoder_2 = torch.compile(text_pipe.text_encoder_2, dynamic=True)
        if text_pipe.text_encoder_3 is not None:
            text_pipe.text_encoder_3 = torch.compile(text_pipe.text_encoder_3, dynamic=True)
        if args.caption_dropout > 0:
            with _silence_encoding_noise():
                null_enc_hs, null_pooled = fast_encode_prompt(
                    text_pipe, [""], args.max_sequence_length, device,
                )
        else:
            null_enc_hs = null_pooled = None
    else:
        text_pipe = None
        null_enc_hs = null_pooled = None

    # --- Model ---
    phases_done = 0   # number of blocks already activated (restored from checkpoint)

    if resume_dir is not None:
        print(f"[rank {rank}] Resuming from {resume_dir} ...")
        model = PomSD3Transformer2DModel.from_pretrained(resume_dir).to(
            device=device, dtype=torch.bfloat16,
        )
        state_path = resume_dir / "train_state.json"
        if state_path.exists():
            phases_done = json.loads(state_path.read_text()).get("phases_done", 0)
    elif init_dir is not None:
        print(f"[rank {rank}] Init weights from {init_dir}, fresh training state ...")
        model = PomSD3Transformer2DModel.from_pretrained(init_dir).to(
            device=device, dtype=torch.bfloat16,
        )
        state_path = init_dir / "train_state.json"
        if state_path.exists():
            phases_done = json.loads(state_path.read_text()).get("phases_done", 0)
    elif not args.smoke_test:
        model = build_progressive_model(
            args.model_id,
            pom_degree=args.pom_degree,
            pom_expand=args.pom_expand,
            pom_n_groups=args.pom_n_groups,
            pom_n_sel_heads=args.pom_n_sel_heads,
            pom_rope_max_seq_len=args.pom_rope_max_seq_len,
            torch_dtype=torch.bfloat16,
            device=device,
        )
    else:
        # Tiny smoke-test model (2 blocks: 1 att + 1 PoM)
        model = PomSD3Transformer2DModel(
            sample_size=32, patch_size=2, in_channels=16, num_layers=2,
            attention_head_dim=16, num_attention_heads=4,
            joint_attention_dim=4096, caption_projection_dim=64,
            pooled_projection_dim=2048, out_channels=16,
            pos_embed_max_size=32, dual_attention_layers=(0,),
            pom_layers=(1,), qk_norm="rms_norm",
            pom_degree=2, pom_expand=2, pom_n_groups=1, pom_n_sel_heads=1,
            pom_rope_max_seq_len=256, lora_rank=0,
        ).to(device=device, dtype=torch.bfloat16)
        for p in model.parameters():
            p.requires_grad_(False)

    if is_main():
        print_model_summary(model, label="progressive (initially frozen)")
        print_model_layers(model)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # FSDP wrap — requires_grad=False params are not reduced (FSDP + use_orig_params=True)
    model = wrap_model_fsdp(model, local_rank, gpus_per_node=args.gpus_per_node)
    model.train()

    # --- EMA shadow (fp32, per-rank shards match FSDP sharding) ---
    # Updated after every optimizer step; used for sampling and validation.
    ema_params = (
        [p.data.clone().float() for p in model.parameters()]
        if args.ema_decay > 0 else None
    )

    @contextlib.contextmanager
    def _use_ema():
        """Temporarily swap EMA weights into the model for inference."""
        if ema_params is None:
            yield
            return
        saved = [p.data.clone() for p in model.parameters()]
        for ema_p, p in zip(ema_params, model.parameters()):
            p.data.copy_(ema_p.to(p.dtype))
        try:
            yield
        finally:
            for sv, p in zip(saved, model.parameters()):
                p.data.copy_(sv)

    def _group_grad_norms() -> dict:
        """Per optimizer-group gradient norms. All ranks participate (all-reduce)."""
        sq_list = [
            sum(
                (p.grad.detach().float().pow(2).sum() for p in pg["params"] if p.grad is not None),
                torch.tensor(0.0, device=device),
            )
            for pg in optimizer.param_groups
        ]
        sq_t = torch.stack(sq_list)
        if dist.is_initialized():
            dist.all_reduce(sq_t, op=dist.ReduceOp.SUM)
        return {
            pg.get("block_idx", -99): sq_t[i].item() ** 0.5
            for i, pg in enumerate(optimizer.param_groups)
        }

    # --- Optimizer (initially empty; blocks added as they're activated) ---
    # Overhead params are split into two groups:
    #   input-side  (pos_embed, time_text_embed, context_embedder): pretrained weights worth
    #               preserving → pretrained_lr_scale × lr
    #   output-side (norm_out, proj_out): pretrained only for SD3.5 att block outputs, so
    #               their pretrained calibration becomes meaningless as PoM blocks take over;
    #               train at full lr so they can adapt quickly.
    inner = getattr(model, '_fsdp_wrapped_module', model)
    block_param_ids = {id(p) for i in range(len(inner.transformer_blocks))
                       for p in inner.transformer_blocks[i].parameters()}

    output_modules = (inner.norm_out, inner.proj_out)
    output_param_ids = {id(p) for m in output_modules for p in m.parameters()}

    input_overhead  = [p for p in model.parameters()
                       if id(p) not in block_param_ids and id(p) not in output_param_ids]
    output_overhead = [p for p in model.parameters() if id(p) in output_param_ids]

    for p in input_overhead + output_overhead:
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [
            {"params": input_overhead,  "lr": args.lr * args.pretrained_lr_scale,
             "block_idx": -1, "is_att": True,  "activated_at": 0},
            {"params": output_overhead, "lr": args.lr,
             "block_idx": -2, "is_att": False, "activated_at": 0},
        ],
        weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )

    # Replay activations for resumed runs (adds param groups to optimizer)
    replay_activations(model, optimizer, phases_done, args.lr, args.pretrained_lr_scale,
                       phase_steps=args.phase_steps)

    # --- Restore optimizer state on full resume ---
    step = 0
    if resume_dir is not None:
        opt_path = resume_dir / "optimizer.pt"
        if opt_path.exists():
            if isinstance(model, FSDP):
                load_optimizer_fsdp(model, optimizer, resume_dir)
            else:
                load_checkpoint_optimizer(
                    torch.load(opt_path, map_location="cpu"), optimizer, model, device
                )
        state_path = resume_dir / "train_state.json"
        _resume_last_phase_step = step
        if state_path.exists():
            d = json.loads(state_path.read_text())
            step = d.get("step", 0) + 1
            phases_done = d.get("phases_done", phases_done)
            _resume_last_phase_step = d.get("last_phase_step", 0)
            if is_main():
                print(f"  Resumed at step {step}, phases_done={phases_done}")

    # --- Load EMA state if resuming ---
    if ema_params is not None and resume_dir is not None:
        ema_path = resume_dir / f"ema_rank{rank}.pt"
        if ema_path.exists():
            saved_ema = torch.load(ema_path, map_location="cpu", weights_only=True)
            for ema_p, sv in zip(ema_params, saved_ema):
                ema_p.copy_(sv.float())
            if is_main():
                print(f"  Loaded EMA from {ema_path.name}")
        else:
            if is_main():
                print("  No EMA checkpoint found — EMA initialised from model weights")

    # Activate the first block right away (at step 0) if nothing has been activated yet.
    # This ensures we have at least one trainable block from the very first step.
    if phases_done == 0 and not args.smoke_test:
        block_idx = ACTIVATION_ORDER[0]   # = 23
        activate_block(model, optimizer, block_idx, args.lr,
                       is_att=(block_idx in ATT_KEEP), pretrained_lr_scale=args.pretrained_lr_scale,
                       activated_at=0)
        phases_done = 1
    elif args.smoke_test and phases_done == 0:
        # smoke: activate both blocks immediately
        for idx in [1, 0]:
            activate_block(model, optimizer, idx, args.lr,
                           is_att=(idx in ATT_KEEP), pretrained_lr_scale=args.pretrained_lr_scale)
        phases_done = 2

    # --- Dataset ---
    latent_size = args.image_size // 8

    if args.smoke_test:
        class _SmokeStream(torch.utils.data.IterableDataset):
            def __iter__(self):
                while True:
                    yield {"pixel_values": torch.randn(3, 64, 64), "caption": "a test image"}
        dataset = _SmokeStream()
    else:
        dataset = GPicDataset(
            dataset_name=args.dataset_name, split=args.dataset_split,
            image_size=args.image_size, rank=rank, world_size=world_size,
            caption_type=args.caption_type, dataset_dir=args.dataset_dir,
        )

    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        num_workers=0 if args.smoke_test else args.num_workers,
        pin_memory=not args.smoke_test, collate_fn=gpic_collate,
    )

    # --- Validation cache ---
    val_cache = None
    if not args.smoke_test and text_pipe is not None and vae is not None and args.val_every > 0:
        val_cache = load_val_cache(
            args.dataset_dir, args.n_val_images, args.image_size,
            text_pipe, args.max_sequence_length, device, vae,
        )

    # --- Training loop setup ---
    min_phase_steps = args.min_phase_steps if args.min_phase_steps is not None \
                      else args.phase_steps // 2
    last_phase_step = _resume_last_phase_step if resume_dir is not None else step
    loss_ema = None                 # exponential moving average of the loss
    loss_ema_decay = 0.99
    loss_history = collections.deque(maxlen=args.plateau_window)  # for plateau detection
    in_consolidation = (phases_done == len(ACTIVATION_ORDER))

    start_step = step
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    batch_iter = iter(loader)

    def _try_advance_phase(step):
        """Activate the next block if the phase has converged or hit phase_steps."""
        nonlocal phases_done, last_phase_step, loss_ema, in_consolidation
        if phases_done >= len(ACTIVATION_ORDER) or args.smoke_test:
            return
        steps_in_phase = step - last_phase_step
        if steps_in_phase < min_phase_steps:
            return

        # Check: hit hard cap OR loss has plateaued.
        plateau = False
        if len(loss_history) == args.plateau_window:
            old_ema = loss_history[0]
            cur_ema = loss_history[-1]
            relative_improvement = (old_ema - cur_ema) / (old_ema + 1e-8)
            plateau = relative_improvement < args.plateau_threshold
        forced = steps_in_phase >= args.phase_steps

        if not (plateau or forced):
            return

        reason = "plateau" if plateau and not forced else "max steps"

        # Log samples just before activating the next block — clean baseline
        # uncontaminated by the freshly activated block (logged at step-1).
        model.eval()
        with _use_ema():
            generate_samples(model, vae, text_pipe, step - 1, device,
                             args.num_sample_prompts, resolution=args.image_size)
        model.train()

        block_idx = ACTIVATION_ORDER[phases_done]
        is_att_block = block_idx in ATT_KEEP
        activate_block(model, optimizer, block_idx, args.lr,
                       is_att=is_att_block,
                       pretrained_lr_scale=args.pretrained_lr_scale,
                       activated_at=step)
        phases_done += 1
        last_phase_step = step

        # Bump downstream att blocks to full LR when a PoM is inserted upstream.
        if not is_att_block:
            for pg in optimizer.param_groups:
                bid = pg.get("block_idx", -1)
                if bid >= 0 and bid > block_idx and pg.get("is_att", False):
                    pg["is_att"] = False
                    if is_main():
                        print(f"  Bumped block {bid} to full LR "
                              f"(upstream PoM inserted at block {block_idx})")

        if is_main():
            total_trainable = sum(
                p.numel() for g in optimizer.param_groups for p in g["params"]
                if p.requires_grad
            )
            print(f"step={step}  phases_done={phases_done}/{len(ACTIVATION_ORDER)}"
                  f"  trainable={total_trainable/1e6:.0f}M params  reason={reason}")

        if phases_done == len(ACTIVATION_ORDER):
            in_consolidation = True
            if is_main():
                print(f"step={step}  All blocks activated → consolidation phase "
                      f"(lr_scale={args.consolidation_lr_scale})")

    while step < args.max_steps:
        # --- Progressive activation (adaptive: plateau or phase_steps cap) ---
        if step > 0:
            _try_advance_phase(step)

        # --- LR schedule + per-block warmup + consolidation ---
        base_lr = lr_schedule(step, args.warmup_steps, args.max_steps, args.lr)
        for pg in optimizer.param_groups:
            # Per-block warmup: ramp from 0 to target LR over block_warmup_steps.
            local_step = step - pg.get("activated_at", 0)
            warmup = min(1.0, local_step / max(1, args.block_warmup_steps))
            if in_consolidation:
                # Consolidation: uniform lr scale for everything.
                pg["lr"] = base_lr * args.consolidation_lr_scale * warmup
            else:
                scale = args.pretrained_lr_scale if pg.get("is_att", False) else 1.0
                pg["lr"] = base_lr * scale * warmup

        # --- Batch ---
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)

        pixel_values = batch["pixel_values"]
        captions = batch["caption"]
        B = pixel_values.shape[0]

        crop_strs = batch.get("crop_str")
        if crop_strs is not None and args.crop_str_dropout < 1.0:
            captions = [
                cap + " " + cs if random.random() > args.crop_str_dropout else cap
                for cap, cs in zip(captions, crop_strs)
            ]

        # --- VAE encode ---
        if vae is not None:
            with torch.no_grad():
                latents = vae.encode(
                    pixel_values.to(device=device, dtype=torch.bfloat16)
                ).latent_dist.sample()
                x_0 = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            x_0 = torch.randn(B, 16, latent_size, latent_size,
                               device=device, dtype=torch.bfloat16)

        # --- Text encode ---
        if text_pipe is not None:
            with _silence_encoding_noise():
                enc_hs, pooled = fast_encode_prompt(
                    text_pipe, captions, args.max_sequence_length, device,
                )
            if args.caption_dropout > 0 and null_enc_hs is not None:
                drop = torch.rand(B, device=device) < args.caption_dropout
                if drop.any():
                    n_drop = int(drop.sum())
                    enc_hs[drop] = null_enc_hs.expand(n_drop, -1, -1)
                    pooled[drop] = null_pooled.expand(n_drop, -1)
        else:
            enc_hs = torch.randn(B, 8, 4096, device=device, dtype=torch.bfloat16)
            pooled = torch.randn(B, 2048, device=device, dtype=torch.bfloat16)

        # --- Flow matching loss (SD3 v-prediction) ---
        u = torch.sigmoid(
            torch.randn(B, device=device) * args.logit_normal_std + args.logit_normal_mean
        )
        t = (u * 999).clamp(1, 999).long()
        sigma = (t.float() / 1000).view(B, 1, 1, 1)
        eps = torch.randn_like(x_0)
        x_t = ((1 - sigma) * x_0 + sigma * eps).to(x_0.dtype)

        is_last_accum = (step + 1) % args.grad_accum_steps == 0
        sync_ctx = (
            contextlib.nullcontext()
            if not isinstance(model, FSDP) or is_last_accum
            else model.no_sync()
        )
        with sync_ctx:
            v_pred = model(
                hidden_states=x_t,
                encoder_hidden_states=enc_hs,
                pooled_projections=pooled,
                timestep=t,
            ).sample
            v_target = (eps - x_0).to(v_pred.dtype)
            loss_per = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=(1, 2, 3))
            # Min-SNR loss weighting: down-weights very-low-noise timesteps
            # (σ≪1, SNR→∞) which otherwise dominate and slow convergence.
            # SNR = (1-σ)²/σ²; weight = min(SNR, γ)/SNR ∈ (0,1].
            if args.snr_gamma > 0:
                snr = (1.0 - sigma.view(B)) ** 2 / (sigma.view(B) ** 2).clamp(min=1e-6)
                snr_weight = (snr.clamp(max=args.snr_gamma) / snr).to(loss_per.dtype)
                loss = (loss_per * snr_weight).mean()
            else:
                loss = loss_per.mean()
            (loss / args.grad_accum_steps).backward()

        _block_gnorms: dict | None = None
        if is_last_accum:
            if isinstance(model, FSDP):
                model.clip_grad_norm_(1.0)
            else:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], 1.0
                )
            # Capture per-block grad norms before zero_grad (all ranks participate)
            if step % args.log_every == 0:
                _block_gnorms = _group_grad_norms()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # EMA update: lerp each shard toward the current weights.
            # Adaptive decay ramps from ~0 at step 0 to args.ema_decay
            # asymptotically, so early samples aren't biased toward the
            # randomly-initialised PoM weights.
            if ema_params is not None:
                ema_decay_t = min(args.ema_decay, (1.0 + step) / (10.0 + step))
                with torch.no_grad():
                    for ema_p, p in zip(ema_params, model.parameters()):
                        ema_p.lerp_(p.data.float(), 1.0 - ema_decay_t)

        # --- Update loss EMA and history (used for plateau detection) ---
        loss_val = loss.item()
        # All-reduce the scalar loss so every rank has the same EMA value.
        # Plateau detection uses loss_history; if ranks diverge they advance
        # the phase at different steps → NCCL deadlock on the FSDP forward
        # inside generate_samples.
        if dist.is_initialized():
            _lv = torch.tensor(loss_val, device=device)
            dist.all_reduce(_lv, op=dist.ReduceOp.AVG)
            loss_val = _lv.item()
        if loss_ema is None:
            loss_ema = loss_val
        else:
            loss_ema = loss_ema_decay * loss_ema + (1 - loss_ema_decay) * loss_val
        loss_history.append(loss_ema)

        # --- Logging ---
        if is_main() and step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = (step + 1 - start_step) * args.batch_size * world_size / elapsed
            t_cpu = t.cpu().float()
            lp = loss_per.detach().cpu()
            low  = t_cpu < 334
            mid  = (t_cpu >= 334) & (t_cpu < 667)
            high = t_cpu >= 667
            log = {
                "loss":           loss_val,
                "loss_ema":       loss_ema,
                "loss_low_t":     lp[low].mean().item()  if low.any()  else float("nan"),
                "loss_mid_t":     lp[mid].mean().item()  if mid.any()  else float("nan"),
                "loss_high_t":    lp[high].mean().item() if high.any() else float("nan"),
                "lr":             base_lr,
                "phases_done":    phases_done,
                "in_consolidation": int(in_consolidation),
                "steps_in_phase": step - last_phase_step,
                "step":           step,
                "samples_per_sec": sps,
            }
            if _block_gnorms is not None:
                for bid, gnorm in _block_gnorms.items():
                    key = "grad_norm/output" if bid == -2 else \
                          "grad_norm/input"  if bid == -1 else \
                          f"grad_norm/block_{bid:02d}"
                    log[key] = gnorm
            wandb.log(log, step=step)
            print(f"step={step:7d}  loss={loss_val:.4f}  ema={loss_ema:.4f}  lr={base_lr:.2e}"
                  f"  phases={phases_done}/{len(ACTIVATION_ORDER)}  {sps:.1f} samp/s")

        # h_scale histograms: summon_full_params is a collective — all ranks
        # must enter it, so this block is outside is_main().
        if step % args.log_every == 0:
            with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
                if is_main():
                    _inner = getattr(model, '_fsdp_wrapped_module', model)
                    hscale_log = {}
                    for i, blk in enumerate(_inner.transformer_blocks):
                        hs = getattr(getattr(blk, 'pom', None), 'h_scale', None)
                        if hs is not None:
                            hscale_log[f"h_scale/block_{i:02d}"] = wandb.Histogram(hs.detach().float().cpu().numpy())
                        hs2 = getattr(getattr(blk, 'pom2', None), 'h_scale', None)
                        if hs2 is not None:
                            hscale_log[f"h_scale2/block_{i:02d}"] = wandb.Histogram(hs2.detach().float().cpu().numpy())
                    wandb.log(hscale_log, step=step)

        # --- Checkpointing ---
        if step > 0 and step % args.save_every == 0:
            ckpt_dir = out_dir / f"step_{step:07d}"
            save_checkpoint(model, optimizer, step, ckpt_dir)
            # Append phases_done to the train_state so resume restores correctly
            if is_main():
                state = json.loads((ckpt_dir / "train_state.json").read_text())
                state["phases_done"] = phases_done
                state["last_phase_step"] = last_phase_step
                (ckpt_dir / "train_state.json").write_text(json.dumps(state))
                print(f"Saved checkpoint to {ckpt_dir}")
            if ema_params is not None:
                torch.save([p.cpu() for p in ema_params], ckpt_dir / f"ema_rank{rank}.pt")
            if dist.is_initialized():
                dist.barrier()

        # --- Validation ---
        if step > 0 and args.val_every > 0 and step % args.val_every == 0 and not args.smoke_test:
            with _use_ema():
                run_validation(model, val_cache, step, device)

        # --- Sample generation ---
        if step > 0 and step % args.sample_every == 0 and not args.smoke_test:
            model.eval()
            with _use_ema():
                generate_samples(model, vae, text_pipe, step, device,
                                 args.num_sample_prompts, resolution=args.image_size)
            model.train()

        step += 1

        # --- Wall-time signal (Slurm requeue) ---
        if (out_dir / ".save_and_exit").exists():
            ckpt_dir = out_dir / f"step_{step:07d}"
            save_checkpoint(model, optimizer, step, ckpt_dir)
            if ema_params is not None:
                torch.save([p.cpu() for p in ema_params], ckpt_dir / f"ema_rank{rank}.pt")
            if is_main():
                state = json.loads((ckpt_dir / "train_state.json").read_text())
                state["phases_done"] = phases_done
                state["last_phase_step"] = last_phase_step
                (ckpt_dir / "train_state.json").write_text(json.dumps(state))
                (out_dir / ".save_and_exit").unlink(missing_ok=True)
                print(f"Wall-time signal — saved to {ckpt_dir}. Exiting.")
            if dist.is_initialized():
                dist.barrier()
            cleanup_ddp()
            sys.exit(0)

        if args.smoke_test and step >= 5:
            print("Smoke test passed.")
            cleanup_ddp()
            return

    # --- Final save ---
    final_dir = out_dir / "final"
    if isinstance(model, FSDP):
        fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_cfg):
            full_sd = model.state_dict()
        _prefix = "_fsdp_wrapped_module."
        full_sd = {(k[len(_prefix):] if k.startswith(_prefix) else k): v
                   for k, v in full_sd.items()}
        if is_main():
            final_dir.mkdir(parents=True, exist_ok=True)
            safetensors_save_file(full_sd, final_dir / "diffusion_pytorch_model.safetensors")
            getattr(model, "_fsdp_wrapped_module", model).save_config(final_dir)
        if dist.is_initialized():
            dist.barrier()
    else:
        if is_main():
            model.save_pretrained(final_dir)

    if ema_params is not None:
        torch.save([p.cpu() for p in ema_params], final_dir / f"ema_rank{rank}.pt")

    if is_main():
        print(f"Training complete. Model saved to {final_dir}")
        wandb.finish()
    cleanup_ddp()


if __name__ == "__main__":
    main()
