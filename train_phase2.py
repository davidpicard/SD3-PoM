"""Phase-2 distillation: all PomSD3 parameters are trained (not just PoM layers).

Start from a phase-1 checkpoint produced by train.py. The same distillation
loss (MSE + MAE vs the frozen SD3.5 teacher) is used; the only difference is
that every student parameter receives gradients.

Launch with torchrun (SLURM handles multi-node):
    torchrun --nproc_per_node=4 --nnodes=4 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
        train_phase2.py \
            --resume_from /path/to/phase1/final \
            --model_id /path/to/models/sd3.5-medium \
            --embeddings_dir /path/to/embeddings \
            --output_dir /path/to/output_phase2

Or run locally for a smoke test (no checkpoint needed):
    python train_phase2.py --embeddings_dir ./embeddings --output_dir ./output_p2 --smoke_test
"""
import argparse
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
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
                   help="HF hub ID or local path to SD3.5 medium (teacher + pipeline)")
    p.add_argument("--embeddings_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from", default=None,
                   help="Phase-1 checkpoint to start from (required unless --smoke_test)")

    # PoM arch — must match the phase-1 checkpoint
    p.add_argument("--pom_degree", type=int, default=4)
    p.add_argument("--pom_expand", type=int, default=2)
    p.add_argument("--pom_n_groups", type=int, default=1)
    p.add_argument("--pom_n_sel_heads", type=int, default=24)

    # Training
    p.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size")
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Lower default than phase 1 — fine-tuning all weights")
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=50_000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--block_loss_weight", type=float, default=0.1)
    p.add_argument("--lora_rank", type=int, default=16,
                   help="LoRA rank on FF layers in each block (0 to disable)")
    p.add_argument("--pure_block_steps", type=int, default=0,
                   help="For this many steps, train on block_loss only (no student forward / final_loss).")
    p.add_argument("--latent_height", type=int, default=64)
    p.add_argument("--latent_width", type=int, default=64)

    # Logging / checkpointing
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2_000)
    p.add_argument("--sample_every", type=int, default=2_000)
    p.add_argument("--num_sample_prompts", type=int, default=25)
    p.add_argument("--wandb_project", default="sd3-pom")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_offline", action="store_true",
                   help="Run wandb in offline mode. Sync later with: wandb sync <run_dir>")

    p.add_argument("--smoke_test", action="store_true",
                   help="Run 5 steps on random data then exit (no checkpoint or HF hub needed)")
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

def _mse_mae(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b) + F.l1_loss(a, b)


def _mse_mae_block(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    scale = b.detach().std().clamp(min=1e-8)
    return F.mse_loss(a / scale, b / scale) + F.l1_loss(a / scale, b / scale)


def teacher_forced_block_loss(
    raw_student,
    teacher_block_data: list[dict],
    device: torch.device,
) -> torch.Tensor:
    """Block-level teacher forcing: run the full student block on teacher's pre-block
    states and compare the full block output to the teacher's block output."""
    block_loss = torch.tensor(0.0, device=device)
    for s_blk, cap in zip(raw_student.transformer_blocks, teacher_block_data):
        hs_in = cap.get('hs_in')
        if hs_in is None:
            continue
        enc_hs_pred, hs_pred = s_blk(
            hidden_states=hs_in,
            encoder_hidden_states=cap.get('enc_hs_in'),
            temb=cap['temb_in'],
        )
        block_loss = block_loss + _mse_mae_block(hs_pred, cap['hs_out'])
        if enc_hs_pred is not None and cap.get('enc_hs_out') is not None:
            block_loss = block_loss + _mse_mae_block(enc_hs_pred, cap['enc_hs_out'])
    return block_loss


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
def generate_samples(
    student: PomSD3Transformer2DModel,
    model_id: str,
    step: int,
    device,
    num_prompts: int = 4,
):
    model_path = Path(model_id)
    if not (model_path / "vae").exists():
        print(
            f"  Skipping samples at step {step}: VAE not found in {model_id}. "
            "Re-download without --skip_vae to enable sample generation."
        )
        return

    print(f"Generating sample images at step {step} ...")
    local = model_path.exists()
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=None,
        local_files_only=local,
    ).to(device=device, dtype=torch.bfloat16)
    pipe.transformer = student
    pipe.set_progress_bar_config(disable=True)

    tmpdir = tempfile.mkdtemp()
    try:
        images = []
        for i, prompt in enumerate(SAMPLE_PROMPTS[:num_prompts]):
            img = pipe(prompt, num_inference_steps=28, guidance_scale=4.5).images[0]
            path = os.path.join(tmpdir, f"{i:03d}.jpg")
            img.save(path, format="JPEG", quality=85)
            images.append(wandb.Image(path, caption=prompt))

        del pipe
        torch.cuda.empty_cache()

        wandb.log({"samples": images}, step=step)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.smoke_test and not args.resume_from:
        raise ValueError("--resume_from is required for phase-2 training (pass a phase-1 checkpoint path)")

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
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # --- Teacher (frozen) ---
    if not args.smoke_test:
        print(f"[rank {local_rank}] Loading teacher ...")
        _local = Path(args.model_id).exists()
        teacher = SD3Transformer2DModel.from_pretrained(
            args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
            local_files_only=_local,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    else:
        teacher = None

    # --- Student ---
    if args.resume_from:
        print(f"[rank {local_rank}] Loading phase-1 checkpoint from {args.resume_from} ...")
        student = PomSD3Transformer2DModel.from_pretrained(args.resume_from).to(device=device, dtype=torch.bfloat16)
    else:
        # Smoke test only — tiny model, no checkpoint
        student = PomSD3Transformer2DModel(
            sample_size=32, patch_size=2, in_channels=16, num_layers=2,
            attention_head_dim=16, num_attention_heads=4,
            joint_attention_dim=4096, caption_projection_dim=64,
            pooled_projection_dim=2048, out_channels=16,
            pos_embed_max_size=32, dual_attention_layers=(0,),
            pom_degree=2, pom_expand=2, pom_n_groups=1, pom_n_sel_heads=1,
            lora_rank=4,
        ).to(device=device, dtype=torch.bfloat16)

    student.train()
    raw_student = student

    # --- All parameters are trainable in phase 2 ---
    for param in raw_student.parameters():
        param.requires_grad_(True)

    n_total = sum(p.numel() for p in raw_student.parameters())
    if is_main():
        print(f"Trainable params: {n_total / 1e6:.1f}M (all)")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        raw_student.parameters(),
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

            lr = lr_schedule(step, args.warmup_steps, args.max_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            hidden_states = batch["hidden_states"].to(device=device, dtype=torch.bfloat16)
            enc_hs = batch["encoder_hidden_states"].to(device=device, dtype=torch.bfloat16)
            pooled = batch["pooled_projections"].to(device=device, dtype=torch.bfloat16)
            timestep = batch["timestep"].to(device=device)

            # --- Teacher forward (no grad): capture block input/output per block ---
            if teacher is not None:
                teacher_block_data = [{} for _ in range(len(teacher.transformer_blocks))]
                teacher_hooks = []

                for _i, _blk in enumerate(teacher.transformer_blocks):
                    _cap = teacher_block_data[_i]

                    def _make_pre(_c):
                        def _hook(_m, _args, _kwargs):
                            _c['hs_in']     = _kwargs.get('hidden_states')
                            _c['enc_hs_in'] = _kwargs.get('encoder_hidden_states')
                            _c['temb_in']   = _kwargs.get('temb')
                        return _hook

                    def _make_post(_c):
                        def _hook(_m, _inp, _out):
                            _c['enc_hs_out'] = _out[0].detach() if _out[0] is not None else None
                            _c['hs_out']     = _out[1].detach()
                        return _hook

                    teacher_hooks.append(_blk.register_forward_pre_hook(_make_pre(_cap), with_kwargs=True))
                    teacher_hooks.append(_blk.register_forward_hook(_make_post(_cap)))

                with torch.no_grad():
                    teacher_out = teacher(
                        hidden_states=hidden_states,
                        encoder_hidden_states=enc_hs,
                        pooled_projections=pooled,
                        timestep=timestep,
                    ).sample.detach()

                for _h in teacher_hooks:
                    _h.remove()
            else:
                # Smoke test: no teacher, skip block loss
                teacher_out = torch.randn_like(hidden_states)
                teacher_block_data = []

            # --- Block loss: full student block on teacher's pre-block states (with grad) ---
            block_loss = (
                teacher_forced_block_loss(raw_student, teacher_block_data, device)
                if teacher_block_data else torch.tensor(0.0, device=device)
            )

            # --- Student forward for final loss ---
            pure_block_phase = step < args.pure_block_steps
            if pure_block_phase:
                final_loss = torch.tensor(0.0, device=device)
                total_loss = block_loss
            else:
                student_out = raw_student(
                    hidden_states=hidden_states,
                    encoder_hidden_states=enc_hs,
                    pooled_projections=pooled,
                    timestep=timestep,
                ).sample
                final_loss = _mse_mae(student_out, teacher_out)
                total_loss = final_loss + args.block_loss_weight * block_loss
            (total_loss / args.grad_accum_steps).backward()

            if (step + 1) % args.grad_accum_steps == 0:
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    for _p in raw_student.parameters():
                        if _p.grad is not None:
                            dist.all_reduce(_p.grad, op=dist.ReduceOp.SUM)
                            _p.grad.div_(world_size)
                torch.nn.utils.clip_grad_norm_(raw_student.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # --- Logging ---
            if is_main() and step % args.log_every == 0:
                elapsed = time.time() - t0
                phase_tag = "[block-only]" if pure_block_phase else "[joint]    "
                log = {
                    "loss": total_loss.item(),
                    "final_loss": final_loss.item(),
                    "block_loss": block_loss.item(),
                    "lr": lr,
                    "step": step,
                    "samples_per_sec": (step + 1) * args.batch_size * (dist.get_world_size() if dist.is_initialized() else 1) / elapsed,
                }
                wandb.log(log, step=step)
                print(
                    f"step={step:6d} {phase_tag}  loss={log['loss']:.4f}  "
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
                generate_samples(raw_student, args.model_id, step, device, num_prompts=args.num_sample_prompts)
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
