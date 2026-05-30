"""Distillation training: PomSD3 student learns to match SD3.5 teacher block-by-block.

Launch with torchrun (SLURM handles multi-node):
    torchrun --nproc_per_node=4 --nnodes=4 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
        train.py --config config.yaml

Or run locally for a smoke test:
    python train.py --embeddings_dir ./embeddings --output_dir ./output --smoke_test
"""
import argparse
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import wandb
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline

from dataset import EmbeddingDataset
from pom_sd3 import PomSD3Transformer2DModel, SD35_MEDIUM_CONFIG, load_sd3_weights_into_pom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium",
                   help="HF hub ID or local path (use local path on compute nodes without internet)")
    p.add_argument("--embeddings_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from", default=None, help="Path to student checkpoint to resume from")

    # PoM arch
    p.add_argument("--pom_degree", type=int, default=4)
    p.add_argument("--pom_expand", type=int, default=2)
    p.add_argument("--pom_n_groups", type=int, default=1)
    p.add_argument("--pom_n_sel_heads", type=int, default=1)

    # Training
    p.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size")
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--warmup_steps", type=int, default=1_000)
    p.add_argument("--block_loss_weight", type=float, default=0.1,
                   help="Weight for intermediate block losses vs final output loss")
    p.add_argument("--latent_height", type=int, default=64)
    p.add_argument("--latent_width", type=int, default=64)

    # Logging / checkpointing
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2_000)
    p.add_argument("--sample_every", type=int, default=2_000)
    p.add_argument("--num_sample_prompts", type=int, default=4)
    p.add_argument("--wandb_project", default="sd3-pom")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_offline", action="store_true",
                   help="Run wandb in offline mode (no internet required). Sync later with: wandb sync <run_dir>")

    p.add_argument("--smoke_test", action="store_true",
                   help="Run 5 steps on random data then exit (no HF hub needed)")
    return p.parse_args()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def lr_schedule(step: int, warmup_steps: int, max_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

def distillation_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    student_intermediates: list[tuple],
    teacher_intermediates: list[tuple],
    block_weight: float,
) -> dict[str, torch.Tensor]:
    """MSE + MAE between student and teacher at each block and at final output."""
    def mse_mae(a, b):
        return F.mse_loss(a, b) + F.l1_loss(a, b)

    final_loss = mse_mae(student_out, teacher_out)

    block_loss = torch.tensor(0.0, device=student_out.device)
    for (s_enc, s_img), (t_enc, t_img) in zip(student_intermediates, teacher_intermediates):
        block_loss = block_loss + mse_mae(s_img, t_img)
        if s_enc is not None and t_enc is not None:
            block_loss = block_loss + mse_mae(s_enc, t_enc)

    total = final_loss + block_weight * block_loss
    return {"loss": total, "final_loss": final_loss.detach(), "block_loss": block_loss.detach()}


# ---------------------------------------------------------------------------
# Sample image generation (main process only)
# ---------------------------------------------------------------------------

SAMPLE_PROMPTS = [
    "a serene mountain landscape at sunrise, photorealistic",
    "a cyberpunk city at night, neon lights reflecting on wet streets",
    "a portrait of a fox in a business suit, oil painting",
    "abstract colorful geometric shapes, vibrant, high contrast",
    "an astronaut floating in space, Earth visible in the background",
    "a cozy library with warm lighting and shelves full of books",
    "a dragon perched on a medieval castle tower, fantasy art",
    "a bowl of ramen with steam rising, food photography",
    "a watercolor painting of a Venice canal at dusk",
    "a robot tending to a flower garden, whimsical illustration",
    "dense rainforest with rays of sunlight piercing the canopy",
    "a black and white portrait of an elderly woman, cinematic",
    "a futuristic space station interior, hard sci-fi concept art",
    "cherry blossom trees along a river in spring, Japan",
    "a close-up of a honeybee on a sunflower, macro photography",
    "a surrealist painting of melting clocks in a desert landscape",
    "a Viking longship on a stormy sea, dramatic lighting",
    "a bustling street market in Marrakech, golden hour",
    "an Art Deco poster of a luxury ocean liner",
    "a snowy owl in flight against a pale winter sky",
    "an underwater coral reef teeming with colorful fish",
    "a steampunk airship over a Victorian city, detailed illustration",
    "a minimalist ink drawing of a mountain range",
    "a wolf howling at the full moon in a pine forest, night",
    "a child blowing dandelion seeds in a summer meadow, soft focus",
]


@torch.no_grad()
def generate_samples(student: PomSD3Transformer2DModel, model_id: str, step: int, device):
    """Generate images with the student and return wandb.Image list."""
    print(f"Generating sample images at step {step} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=student,
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    images = []
    for prompt in SAMPLE_PROMPTS:
        img = pipe(prompt, num_inference_steps=28, guidance_scale=4.5).images[0]
        images.append(wandb.Image(img, caption=prompt))

    del pipe
    torch.cuda.empty_cache()
    return images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- Wandb ---
    if is_main():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            mode="offline" if args.wandb_offline else "online",
        )

    # --- Dataset ---
    if args.smoke_test:
        # Tiny synthetic dataset for smoke testing without real embeddings
        from torch.utils.data import TensorDataset
        B = 16
        fake_enc = torch.randn(B, 8, 4096)
        fake_pooled = torch.randn(B, 2048)
        fake_noise = torch.randn(B, 16, 32, 32)
        fake_t = torch.randint(0, 1000, (B,))

        class SmokeDset(torch.utils.data.Dataset):
            def __len__(self): return B
            def __getitem__(self, i):
                return {
                    "hidden_states": fake_noise[i],
                    "encoder_hidden_states": fake_enc[i],
                    "pooled_projections": fake_pooled[i],
                    "timestep": fake_t[i].item(),
                }
        dataset = SmokeDset()
    else:
        dataset = EmbeddingDataset(
            args.embeddings_dir,
            latent_height=args.latent_height,
            latent_width=args.latent_width,
        )

    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # --- Teacher (frozen) ---
    if not args.smoke_test:
        print(f"[rank {local_rank}] Loading teacher ...")
        from pathlib import Path as _Path
        _local = _Path(args.model_id).exists()
        teacher = SD3Transformer2DModel.from_pretrained(
            args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
            local_files_only=_local,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    else:
        # Lightweight fake teacher for smoke test
        teacher = None

    # --- Student ---
    if args.resume_from:
        print(f"[rank {local_rank}] Resuming student from {args.resume_from} ...")
        student = PomSD3Transformer2DModel.from_pretrained(args.resume_from).to(device=device, dtype=torch.bfloat16)
    elif not args.smoke_test:
        print(f"[rank {local_rank}] Initializing student from teacher weights ...")
        student = PomSD3Transformer2DModel(
            **SD35_MEDIUM_CONFIG,
            pom_degree=args.pom_degree,
            pom_expand=args.pom_expand,
            pom_n_groups=args.pom_n_groups,
            pom_n_sel_heads=args.pom_n_sel_heads,
        ).to(dtype=torch.bfloat16)
        teacher_sd = {k: v.cpu() for k, v in teacher.state_dict().items()}
        load_sd3_weights_into_pom(student, teacher_sd)
        del teacher_sd
        student = student.to(device)
    else:
        student = PomSD3Transformer2DModel(
            sample_size=32, patch_size=2, in_channels=16, num_layers=2,
            attention_head_dim=16, num_attention_heads=4,
            joint_attention_dim=4096, caption_projection_dim=64,
            pooled_projection_dim=2048, out_channels=16,
            pos_embed_max_size=32, dual_attention_layers=(0,),
            pom_degree=2, pom_expand=2, pom_n_groups=1, pom_n_sel_heads=1,
        ).to(device=device, dtype=torch.bfloat16)

    student.train()
    if dist.is_initialized():
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=False)

    raw_student = student.module if isinstance(student, DDP) else student

    # --- Freeze everything except PoM layers ---
    pom_fragments = (".pom.", ".pom2.")
    for name, param in raw_student.named_parameters():
        param.requires_grad_(any(f in name for f in pom_fragments))

    pom_params = [p for p in raw_student.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in pom_params)
    n_total = sum(p.numel() for p in raw_student.parameters())
    if is_main():
        print(f"Trainable (PoM) params: {n_trainable / 1e6:.1f}M / {n_total / 1e6:.1f}M total")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        pom_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # --- Training loop ---
    step = 0
    epoch = 0
    t0 = time.time()

    while step < args.max_steps:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in loader:
            if step >= args.max_steps:
                break

            # Update LR
            lr = lr_schedule(step, args.warmup_steps, args.max_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            hidden_states = batch["hidden_states"].to(device=device, dtype=torch.bfloat16)
            enc_hs = batch["encoder_hidden_states"].to(device=device, dtype=torch.bfloat16)
            pooled = batch["pooled_projections"].to(device=device, dtype=torch.bfloat16)
            timestep = batch["timestep"].to(device=device)

            # --- Teacher forward (no grad) ---
            if teacher is not None:
                with torch.no_grad():
                    teacher_intermediates = []
                    teacher_hooks = []

                    def _make_teacher_hook(lst):
                        def hook(m, inp, out):
                            lst.append((
                                out[0].detach() if out[0] is not None else None,
                                out[1].detach(),
                            ))
                        return hook

                    for blk in teacher.transformer_blocks:
                        teacher_hooks.append(blk.register_forward_hook(_make_teacher_hook(teacher_intermediates)))

                    teacher_out_dict = teacher(
                        hidden_states=hidden_states,
                        encoder_hidden_states=enc_hs,
                        pooled_projections=pooled,
                        timestep=timestep,
                    )
                    teacher_out = teacher_out_dict.sample.detach()

                    for h in teacher_hooks:
                        h.remove()
            else:
                # Smoke test: use random targets
                teacher_out = torch.randn_like(hidden_states)
                n_tokens = (32 * 32) // (2 * 2)  # patch_size=2
                teacher_intermediates = [
                    (None, torch.randn(hidden_states.shape[0], n_tokens, 64, device=device, dtype=torch.bfloat16))
                    for _ in range(2)
                ]

            # --- Student forward ---
            student_out_dict, student_intermediates = raw_student(
                hidden_states=hidden_states,
                encoder_hidden_states=enc_hs,
                pooled_projections=pooled,
                timestep=timestep,
                return_intermediate=True,
            ) if not isinstance(student, DDP) else student.module(
                hidden_states=hidden_states,
                encoder_hidden_states=enc_hs,
                pooled_projections=pooled,
                timestep=timestep,
                return_intermediate=True,
            )
            student_out = student_out_dict.sample

            # --- Loss ---
            losses = distillation_loss(
                student_out,
                teacher_out,
                student_intermediates,
                teacher_intermediates,
                block_weight=args.block_loss_weight,
            )
            loss = losses["loss"] / args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # --- Logging ---
            if is_main() and step % args.log_every == 0:
                elapsed = time.time() - t0
                log = {
                    "loss": losses["loss"].item(),
                    "final_loss": losses["final_loss"].item(),
                    "block_loss": losses["block_loss"].item(),
                    "lr": lr,
                    "step": step,
                    "samples_per_sec": (step + 1) * args.batch_size * (dist.get_world_size() if dist.is_initialized() else 1) / elapsed,
                }
                wandb.log(log, step=step)
                print(
                    f"step={step:6d}  loss={log['loss']:.4f}  "
                    f"final={log['final_loss']:.4f}  block={log['block_loss']:.4f}  "
                    f"lr={lr:.2e}  {log['samples_per_sec']:.1f} samp/s"
                )

            # --- Checkpointing ---
            if is_main() and step > 0 and step % args.save_every == 0:
                ckpt_dir = out_dir / f"step_{step:07d}"
                raw_student.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint to {ckpt_dir}")

            # --- Image samples ---
            if is_main() and step > 0 and step % args.sample_every == 0 and not args.smoke_test:
                raw_student.eval()
                images = generate_samples(raw_student, args.model_id, step, device)
                wandb.log({"samples": images}, step=step)
                raw_student.train()

            step += 1

            if args.smoke_test and step >= 5:
                print("Smoke test passed — 5 steps completed successfully.")
                cleanup_ddp()
                return

    # --- Final save ---
    if is_main():
        final_dir = out_dir / "final"
        raw_student.save_pretrained(final_dir)
        print(f"Training complete. Final model saved to {final_dir}")
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
