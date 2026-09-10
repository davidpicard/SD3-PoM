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

    # All params start with requires_grad=True so FSDP gradient hooks are set up
    # correctly for all blocks. Frozen blocks are excluded from the optimizer
    # instead — gradients are computed (needed for backprop through frozen layers)
    # but not applied. Blocks enter the optimizer progressively via activate_block().

    return model.to(device)


# ---------------------------------------------------------------------------
# Progressive activation
# ---------------------------------------------------------------------------

def block_params(model: torch.nn.Module, block_idx: int) -> list:
    """Return parameters of transformer_blocks[block_idx] (handles FSDP wrapper)."""
    inner = getattr(model, '_fsdp_wrapped_module', model)
    return list(inner.transformer_blocks[block_idx].parameters())


def activate_block(model, optimizer, block_idx: int, lr: float, is_att: bool,
                   pretrained_lr_scale: float) -> None:
    """Add one block's params to the optimizer (all params already have requires_grad=True)."""
    params = block_params(model, block_idx)
    group_lr = lr * pretrained_lr_scale if is_att else lr
    optimizer.add_param_group({
        "params": params,
        "lr": group_lr,
        "block_idx": block_idx,
        "is_att": is_att,
    })
    kind = f"att (lr×{pretrained_lr_scale})" if is_att else "PoM"
    n = sum(p.numel() for p in params)
    if is_main():
        print(f"  Activated block {block_idx:2d} ({kind}): {n/1e6:.1f}M params "
              f"@ lr={group_lr:.2e}")


def replay_activations(model, optimizer, phases_done: int, lr: float,
                       pretrained_lr_scale: float) -> None:
    """Re-apply activation history when resuming from checkpoint."""
    for i in range(phases_done):
        block_idx = ACTIVATION_ORDER[i]
        is_att = block_idx in ATT_KEEP
        activate_block(model, optimizer, block_idx, lr, is_att, pretrained_lr_scale)


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
                   help="Training steps between block activations")
    p.add_argument("--pretrained_lr_scale", type=float, default=0.1,
                   help="LR multiplier for unfrozen att blocks (smaller = gentler adaptation)")

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
             "block_idx": -1, "is_att": True},
            {"params": output_overhead, "lr": args.lr,
             "block_idx": -2, "is_att": False},
        ],
        weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )

    # Replay activations for resumed runs (adds param groups to optimizer)
    replay_activations(model, optimizer, phases_done, args.lr, args.pretrained_lr_scale)

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
        if state_path.exists():
            d = json.loads(state_path.read_text())
            step = d.get("step", 0) + 1
            phases_done = d.get("phases_done", phases_done)
            if is_main():
                print(f"  Resumed at step {step}, phases_done={phases_done}")

    # Activate the first block right away (at step 0) if nothing has been activated yet.
    # This ensures we have at least one trainable block from the very first step.
    if phases_done == 0 and not args.smoke_test:
        block_idx = ACTIVATION_ORDER[0]   # = 23
        activate_block(model, optimizer, block_idx, args.lr,
                       is_att=(block_idx in ATT_KEEP), pretrained_lr_scale=args.pretrained_lr_scale)
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

    # --- Training loop ---
    start_step = step
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    batch_iter = iter(loader)

    while step < args.max_steps:
        # --- Progressive activation ---
        # At each phase boundary (after the first, which happened before the loop),
        # activate the next block from ACTIVATION_ORDER.
        if (step > 0 and step % args.phase_steps == 0
                and phases_done < len(ACTIVATION_ORDER) and not args.smoke_test):
            # Log samples just before replacing the next block so we have a
            # clean baseline uncontaminated by the freshly activated block.
            # Logged at step-1 so it appears just before the phase boundary in wandb.
            model.eval()
            generate_samples(model, vae, text_pipe, step - 1, device,
                             args.num_sample_prompts, resolution=args.image_size)
            model.train()

            block_idx = ACTIVATION_ORDER[phases_done]
            is_att_block = block_idx in ATT_KEEP
            activate_block(model, optimizer, block_idx, args.lr,
                           is_att=is_att_block,
                           pretrained_lr_scale=args.pretrained_lr_scale)
            phases_done += 1

            # When inserting a PoM block at position k, every downstream block
            # (block_idx > k) now receives a changed activation distribution.
            # Att blocks among them were frozen at pretrained_lr_scale to gently
            # awaken; but the upstream PoM invalidates their pretrained input
            # statistics, so bump them to full LR now.
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
                      f"  trainable={total_trainable/1e6:.0f}M params")

        # --- LR schedule ---
        lr = lr_schedule(step, args.warmup_steps, args.max_steps, args.lr)
        for pg in optimizer.param_groups:
            scale = args.pretrained_lr_scale if pg.get("is_att", False) else 1.0
            pg["lr"] = lr * scale

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
            loss = loss_per.mean()
            (loss / args.grad_accum_steps).backward()

        if is_last_accum:
            if isinstance(model, FSDP):
                model.clip_grad_norm_(1.0)
            else:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], 1.0
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

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
                "loss":        loss.item(),
                "loss_low_t":  lp[low].mean().item()  if low.any()  else float("nan"),
                "loss_mid_t":  lp[mid].mean().item()  if mid.any()  else float("nan"),
                "loss_high_t": lp[high].mean().item() if high.any() else float("nan"),
                "lr": lr,
                "phases_done": phases_done,
                "step": step,
                "samples_per_sec": sps,
            }
            wandb.log(log, step=step)
            print(f"step={step:7d}  loss={log['loss']:.4f}  lr={lr:.2e}"
                  f"  phases={phases_done}/{len(ACTIVATION_ORDER)}  {sps:.1f} samp/s")

        # --- Checkpointing ---
        if step > 0 and step % args.save_every == 0:
            ckpt_dir = out_dir / f"step_{step:07d}"
            save_checkpoint(model, optimizer, step, ckpt_dir)
            # Append phases_done to the train_state so resume restores correctly
            if is_main():
                state = json.loads((ckpt_dir / "train_state.json").read_text())
                state["phases_done"] = phases_done
                (ckpt_dir / "train_state.json").write_text(json.dumps(state))
                print(f"Saved checkpoint to {ckpt_dir}")
            if dist.is_initialized():
                dist.barrier()

        # --- Validation ---
        if step > 0 and args.val_every > 0 and step % args.val_every == 0 and not args.smoke_test:
            run_validation(model, val_cache, step, device)

        # --- Sample generation ---
        if step > 0 and step % args.sample_every == 0 and not args.smoke_test:
            model.eval()
            generate_samples(model, vae, text_pipe, step, device,
                             args.num_sample_prompts, resolution=args.image_size)
            model.train()

        step += 1

        # --- Wall-time signal (Slurm requeue) ---
        if (out_dir / ".save_and_exit").exists():
            ckpt_dir = out_dir / f"step_{step:07d}"
            save_checkpoint(model, optimizer, step, ckpt_dir)
            if is_main():
                state = json.loads((ckpt_dir / "train_state.json").read_text())
                state["phases_done"] = phases_done
                (ckpt_dir / "train_state.json").write_text(json.dumps(state))
                (out_dir / ".save_and_exit").unlink(missing_ok=True)
                print(f"Wall-time signal — saved to {ckpt_dir}. Exiting.")
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

    if is_main():
        print(f"Training complete. Model saved to {final_dir}")
        wandb.finish()
    cleanup_ddp()


if __name__ == "__main__":
    main()
