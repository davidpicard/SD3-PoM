"""Progressive PoM replacement training.

Starts with SD3.5 attention model and replaces one block at a time (from the output end)
with JointPoMBlock+LoRA. At each stage the hybrid model keeps its denoising capability
(frozen attention blocks carry the load) while the newly-inserted PoM block is trained
with a clean gradient signal.

Schedule (default --replacement_step_schedule 1000):
  step 0      : 1 PoM block (block 23) + proj_out LoRA
  step 1000   : 2 PoM blocks (blocks 22-23)
  ...
  step 23000  : 24 PoM blocks (fully PoM) — hand off to train_finetune.py

Launch with torchrun (SLURM handles multi-node):
    torchrun --nproc_per_node=4 --nnodes=4 \\
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \\
        train_progressive.py --model_id /path/to/sd3.5-medium ...

Or run locally for a smoke test:
    python train_progressive.py --embeddings_dir ./embeddings --output_dir ./output \\
        --smoke_test --n_pom_blocks_start 1 --replacement_step_schedule 2
"""
import argparse
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

import wandb
from diffusers import (
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
)

from dataset import EmbeddingDataset
from pom_sd3 import (
    PomSD3Transformer2DModel,
    build_from_sd3_pretrained,
    replace_next_attention_block,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--embeddings_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from", default=None,
                   help="Resume a progressive-training checkpoint (already has n_pom_blocks set)")

    # PoM arch
    p.add_argument("--pom_degree", type=int, default=4)
    p.add_argument("--pom_expand", type=int, default=2)
    p.add_argument("--pom_n_groups", type=int, default=1)
    p.add_argument("--pom_n_sel_heads", type=int, default=24)
    p.add_argument("--lora_rank", type=int, default=16)

    # Progressive replacement
    p.add_argument("--n_pom_blocks_start", type=int, default=1,
                   help="Number of PoM blocks at initialization (last N blocks of the 24)")
    p.add_argument("--replacement_step_schedule", type=int, default=1000,
                   help="Replace one more attention block every this many steps")

    # Training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=30_000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--block_loss_weight", type=float, default=0.1)
    p.add_argument("--latent_height", type=int, default=64)
    p.add_argument("--latent_width", type=int, default=64)
    p.add_argument("--teacher_steps_max", type=int, default=28,
                   help="Max Euler steps for teacher x_0 generation; scales down at high t")
    p.add_argument("--amortize_k", type=int, default=2,
                   help="Independent (t -> x_0 -> x_t) passes per optimizer step")
    p.add_argument("--uncond_prob", type=float, default=0.1)

    # Logging / checkpointing
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2_000)
    p.add_argument("--sample_every", type=int, default=2_000)
    p.add_argument("--num_sample_prompts", type=int, default=4)
    p.add_argument("--wandb_project", default="sd3-pom")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--smoke_test", action="store_true",
                   help="Run 5 steps on random data then exit (no HF hub needed)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

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
# Teacher x_0 generation (shared with train.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def teacher_generate_x0(
    teacher: SD3Transformer2DModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    enc_hs: torch.Tensor,
    pooled: torch.Tensor,
    latent_h: int,
    latent_w: int,
    device: torch.device,
    n_steps: int,
) -> torch.Tensor:
    B = enc_hs.shape[0]
    x = torch.randn(B, 16, latent_h, latent_w, device=device, dtype=torch.bfloat16)
    scheduler.set_timesteps(n_steps, device=device)
    for t in scheduler.timesteps:
        v = teacher(x, enc_hs, pooled, t.expand(B)).sample
        x = scheduler.step(v, t, x).prev_sample
    return x


# ---------------------------------------------------------------------------
# Distillation losses (same as train.py)
# ---------------------------------------------------------------------------

def _mse_mae(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b) + F.l1_loss(a, b)


def _mse_mae_block(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    scale = b.detach().std().clamp(min=1e-8)
    return F.mse_loss(a / scale, b / scale) + F.l1_loss(a / scale, b / scale)


def teacher_forced_block_loss(
    raw_student: PomSD3Transformer2DModel,
    teacher_block_data: list[dict],
    device: torch.device,
) -> torch.Tensor:
    block_loss = torch.tensor(0.0, device=device)
    for s_blk, cap in zip(raw_student.transformer_blocks, teacher_block_data):
        hs_in = cap.get("hs_in")
        if hs_in is None:
            continue
        enc_hs_pred, hs_pred = s_blk(
            hidden_states=hs_in,
            encoder_hidden_states=cap.get("enc_hs_in"),
            temb=cap["temb_in"],
        )
        block_loss = block_loss + _mse_mae_block(hs_pred, cap["hs_out"])
        if enc_hs_pred is not None and cap.get("enc_hs_out") is not None:
            block_loss = block_loss + _mse_mae_block(enc_hs_pred, cap["enc_hs_out"])
    return block_loss


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

SAMPLE_PROMPTS = [
    "a serene mountain landscape at sunrise, photorealistic",
    "a cyberpunk city at night, neon lights reflecting on wet streets",
    "a portrait of a fox in a business suit, oil painting",
    "abstract colorful geometric shapes, vibrant, high contrast",
]


@torch.no_grad()
def generate_samples(student, model_id, step, device, num_prompts=4):
    model_path = Path(model_id)
    if not (model_path / "vae").exists():
        print(f"  Skipping samples at step {step}: VAE not found in {model_id}")
        return
    print(f"Generating sample images at step {step} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, transformer=None, local_files_only=model_path.exists(),
    ).to(device=device, dtype=torch.bfloat16)
    pipe.transformer = student
    pipe.set_progress_bar_config(disable=True)
    tmpdir = tempfile.mkdtemp()
    try:
        images = []
        for i, prompt in enumerate(SAMPLE_PROMPTS[:num_prompts]):
            img = pipe(prompt, num_inference_steps=28, guidance_scale=1.0).images[0]
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
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    if is_main():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            mode="offline" if args.wandb_offline else "online",
        )

    # --- Dataset ---
    if args.smoke_test:
        class _SmokeDset(torch.utils.data.Dataset):
            def __len__(self): return 16
            def __getitem__(self, i):
                return {
                    "encoder_hidden_states": torch.randn(8, 4096),
                    "pooled_projections": torch.randn(2048),
                }
        dataset = _SmokeDset()
    else:
        dataset = EmbeddingDataset(args.embeddings_dir, args.latent_height, args.latent_width)

    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=2, pin_memory=True, drop_last=True,
    )

    # --- Teacher (frozen, for x_0 generation + block-state capture) ---
    if not args.smoke_test:
        print(f"[rank {local_rank}] Loading teacher ...")
        _local = Path(args.model_id).exists()
        teacher = SD3Transformer2DModel.from_pretrained(
            args.model_id, subfolder="transformer",
            torch_dtype=torch.bfloat16, local_files_only=_local,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.model_id, subfolder="scheduler",
        )
    else:
        teacher = None
        scheduler = None

    # --- Student (hybrid: attention + PoM) ---
    num_layers = 24  # SD3.5 Medium
    if args.resume_from:
        print(f"[rank {local_rank}] Resuming from {args.resume_from} ...")
        raw_student = PomSD3Transformer2DModel.from_pretrained(args.resume_from).to(
            device=device, dtype=torch.bfloat16
        )
    elif not args.smoke_test:
        print(f"[rank {local_rank}] Building hybrid student ({args.n_pom_blocks_start} PoM blocks) ...")
        raw_student = build_from_sd3_pretrained(
            args.model_id,
            pom_degree=args.pom_degree,
            pom_expand=args.pom_expand,
            pom_n_groups=args.pom_n_groups,
            pom_n_sel_heads=args.pom_n_sel_heads,
            lora_rank=args.lora_rank,
            n_pom_blocks=args.n_pom_blocks_start,
            device=device,
        )
    else:
        raw_student = PomSD3Transformer2DModel(
            sample_size=32, patch_size=2, in_channels=16, num_layers=4,
            attention_head_dim=16, num_attention_heads=4,
            joint_attention_dim=4096, caption_projection_dim=64,
            pooled_projection_dim=2048, out_channels=16,
            pos_embed_max_size=32, dual_attention_layers=(0,),
            pom_degree=2, pom_expand=2, pom_n_groups=1, pom_n_sel_heads=1,
            lora_rank=4, n_pom_blocks=args.n_pom_blocks_start,
        ).to(device=device, dtype=torch.bfloat16)
        num_layers = 4

    raw_student.train()

    # Freeze attention blocks; only PoM+LoRA params are trainable
    pom_fragments = (".pom.", ".pom2.", ".ff_lora_", ".ff_context_lora_", "proj_out_lora_", "norm_out.")
    for name, param in raw_student.named_parameters():
        param.requires_grad_(any(f in name for f in pom_fragments))

    pom_params = [p for p in raw_student.parameters() if p.requires_grad]
    if is_main():
        n_trainable = sum(p.numel() for p in pom_params)
        n_total = sum(p.numel() for p in raw_student.parameters())
        n_pom_now = raw_student.config.n_pom_blocks or num_layers
        print(f"Trainable params: {n_trainable / 1e6:.1f}M / {n_total / 1e6:.1f}M "
              f"({n_pom_now}/{num_layers} PoM blocks)")

    optimizer = torch.optim.AdamW(
        pom_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
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

            enc_hs = batch["encoder_hidden_states"].to(device=device, dtype=torch.bfloat16)
            pooled = batch["pooled_projections"].to(device=device, dtype=torch.bfloat16)
            B = enc_hs.shape[0]

            if random.random() < args.uncond_prob:
                enc_hs = torch.zeros_like(enc_hs)
                pooled = torch.zeros_like(pooled)

            K = args.amortize_k
            total_loss = torch.tensor(0.0, device=device)
            final_loss = torch.tensor(0.0, device=device)
            block_loss = torch.tensor(0.0, device=device)
            teacher_steps_sum = 0

            for _ in range(K):
                timestep = torch.randint(1, 999, (B,), device=device)

                if teacher is not None:
                    t_frac = 1.0 - timestep.float().mean().item() / 1000.0
                    n_steps = max(1, round(args.teacher_steps_max * t_frac))
                    teacher_steps_sum += n_steps
                    x_0 = teacher_generate_x0(
                        teacher, scheduler, enc_hs, pooled,
                        args.latent_height, args.latent_width,
                        device, n_steps=n_steps,
                    )
                else:
                    x_0 = torch.randn(B, 16, 32, 32, device=device, dtype=torch.bfloat16)

                sigma = (timestep.float() / 1000).view(B, 1, 1, 1)
                eps = torch.randn_like(x_0)
                x_t = ((1 - sigma) * x_0 + sigma * eps).to(x_0.dtype)

                # Teacher forward with hooks to capture block states
                if teacher is not None:
                    teacher_block_data = [{} for _ in range(len(teacher.transformer_blocks))]
                    teacher_hooks = []

                    for _i, _blk in enumerate(teacher.transformer_blocks):
                        _cap = teacher_block_data[_i]

                        def _make_pre(_c):
                            def _hook(_m, _args, _kwargs):
                                _c["hs_in"]     = _kwargs.get("hidden_states")
                                _c["enc_hs_in"] = _kwargs.get("encoder_hidden_states")
                                _c["temb_in"]   = _kwargs.get("temb")
                            return _hook

                        def _make_post(_c):
                            def _hook(_m, _inp, _out):
                                _c["enc_hs_out"] = _out[0].detach() if _out[0] is not None else None
                                _c["hs_out"]     = _out[1].detach()
                            return _hook

                        teacher_hooks.append(_blk.register_forward_pre_hook(_make_pre(_cap), with_kwargs=True))
                        teacher_hooks.append(_blk.register_forward_hook(_make_post(_cap)))

                    with torch.no_grad():
                        teacher_out = teacher(
                            hidden_states=x_t,
                            encoder_hidden_states=enc_hs,
                            pooled_projections=pooled,
                            timestep=timestep,
                        ).sample.detach()

                    for _h in teacher_hooks:
                        _h.remove()
                else:
                    teacher_out = torch.randn_like(x_0)
                    teacher_block_data = []

                blk_loss_k = (
                    teacher_forced_block_loss(raw_student, teacher_block_data, device)
                    if teacher_block_data else torch.tensor(0.0, device=device)
                )
                block_loss = block_loss + blk_loss_k.detach() / K

                student_out = raw_student(
                    hidden_states=x_t,
                    encoder_hidden_states=enc_hs,
                    pooled_projections=pooled,
                    timestep=timestep,
                ).sample
                fin_loss_k = _mse_mae(student_out, teacher_out)
                final_loss = final_loss + fin_loss_k.detach() / K

                step_loss = fin_loss_k + args.block_loss_weight * blk_loss_k
                (step_loss / (K * args.grad_accum_steps)).backward()
                total_loss = total_loss + step_loss.detach() / K

            if (step + 1) % args.grad_accum_steps == 0:
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    for _p in pom_params:
                        if _p.grad is not None:
                            dist.all_reduce(_p.grad, op=dist.ReduceOp.SUM)
                            _p.grad.div_(world_size)
                torch.nn.utils.clip_grad_norm_(pom_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # --- Progressive replacement ---
            n_pom_now = raw_student.config.n_pom_blocks if raw_student.config.n_pom_blocks is not None else num_layers
            if step > 0 and step % args.replacement_step_schedule == 0 and n_pom_now < num_layers:
                new_block = replace_next_attention_block(raw_student)
                # Mark new block's PoM+LoRA params as trainable and register with optimizer
                new_pom_params = []
                for name, param in new_block.named_parameters():
                    if any(f in name for f in pom_fragments):
                        param.requires_grad_(True)
                        new_pom_params.append(param)
                        pom_params.append(param)
                if new_pom_params:
                    optimizer.add_param_group({"params": new_pom_params})
                n_pom_now = raw_student.config.n_pom_blocks
                if is_main():
                    print(f"  → Replaced block {num_layers - n_pom_now} | "
                          f"now {n_pom_now}/{num_layers} PoM blocks")
                    wandb.log({"n_pom_blocks": n_pom_now}, step=step)

            # --- Logging ---
            if is_main() and step % args.log_every == 0:
                elapsed = time.time() - t0
                n_world = dist.get_world_size() if dist.is_initialized() else 1
                log = {
                    "loss": total_loss.item(),
                    "final_loss": final_loss.item(),
                    "block_loss": block_loss.item(),
                    "teacher_steps_avg": teacher_steps_sum / K if teacher is not None else 0,
                    "n_pom_blocks": n_pom_now,
                    "lr": lr,
                    "step": step,
                    "samples_per_sec": (step + 1) * args.batch_size * n_world / elapsed,
                }
                wandb.log(log, step=step)
                print(
                    f"step={step:6d}  loss={log['loss']:.4f}  "
                    f"final={log['final_loss']:.4f}  block={log['block_loss']:.4f}  "
                    f"pom={n_pom_now}/{num_layers}  lr={lr:.2e}  {log['samples_per_sec']:.1f} samp/s"
                )

            # --- Checkpointing ---
            if is_main() and step > 0 and step % args.save_every == 0:
                ckpt_dir = out_dir / f"step_{step:07d}"
                raw_student.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint to {ckpt_dir}")

            # --- Image samples ---
            if is_main() and step > 0 and step % args.sample_every == 0 and not args.smoke_test:
                raw_student.eval()
                generate_samples(raw_student, args.model_id, step, device, args.num_sample_prompts)
                raw_student.train()

            step += 1

            if args.smoke_test and step >= 5:
                print("Smoke test passed — 5 steps completed successfully.")
                cleanup_ddp()
                return

    # --- Final save ---
    if is_main():
        raw_student.merge_lora()
        final_dir = out_dir / "final"
        raw_student.save_pretrained(final_dir)
        print(f"Training complete. Final model (LoRA merged) saved to {final_dir}")
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
