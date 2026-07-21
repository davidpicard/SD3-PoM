#!/usr/bin/env python3
"""
Per-component speed benchmark for the SD3-PoM training loop.

Single GPU:    python speed_test.py --model_id /path/to/sd3.5-medium
4-GPU node:    torchrun --nproc_per_node=4 speed_test.py --model_id /path/to/sd3.5-medium

Warms up for --n_warmup steps (lets torch.compile finish), then times --n_steps steps
and prints a breakdown table showing where time actually goes.
"""
import argparse
import contextlib
import functools
import os
import statistics
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from diffusers import AutoencoderKL, StableDiffusion3Pipeline
from pom_sd3.model import PomSD3Transformer2DModel
from pom_sd3.convert import SD35_MEDIUM_CONFIG


# ---------------------------------------------------------------------------
# Distributed helpers (mirrors train_scratch.py)
# ---------------------------------------------------------------------------

def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def wrap_model_fsdp(model, local_rank):
    if not dist.is_initialized():
        return model
    from diffusers.models.attention import JointTransformerBlock
    from pom_sd3.blocks import JointPoMBlock
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={JointPoMBlock, JointTransformerBlock},
    )
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        device_id=local_rank,
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class StepTimer:
    """Accumulates per-step CUDA-event timings. Call .start()/.stop() around
    each timed region; call .collect() once per step after cuda.synchronize()."""

    def __init__(self):
        self.samples: list[float] = []
        self._start_ev = None
        self._stop_ev = None

    def start(self):
        self._start_ev = torch.cuda.Event(enable_timing=True)
        self._start_ev.record()

    def stop(self):
        self._stop_ev = torch.cuda.Event(enable_timing=True)
        self._stop_ev.record()

    def collect(self):
        if self._start_ev is not None and self._stop_ev is not None:
            self.samples.append(self._start_ev.elapsed_time(self._stop_ev))
            self._start_ev = self._stop_ev = None

    @property
    def mean_ms(self):
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def std_ms(self):
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0


# ---------------------------------------------------------------------------
# Dummy caption tokenisation for per-encoder breakdown
# ---------------------------------------------------------------------------

def make_clip_tokens(tokenizer, captions, device):
    toks = tokenizer(
        captions,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in toks.items()}


def make_t5_tokens(tokenizer, captions, max_seq_len, device):
    toks = tokenizer(
        captions,
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in toks.items() if k in ("input_ids", "attention_mask")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hybrid_n", type=int, default=2)
    p.add_argument("--pom_degree", type=int, default=4)
    p.add_argument("--pom_expand", type=int, default=2)
    p.add_argument("--pom_n_groups", type=int, default=1)
    p.add_argument("--pom_n_sel_heads", type=int, default=24)
    p.add_argument("--max_sequence_length", type=int, default=77,
                   help="T5 token budget (SD3 default=256; 77 is the CLIP budget)")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--no_compile", action="store_true",
                   help="Skip torch.compile on VAE and text encoders")
    p.add_argument("--n_warmup", type=int, default=3,
                   help="Steps before timing starts (lets torch.compile JIT finish)")
    p.add_argument("--n_steps", type=int, default=10,
                   help="Steps to time")
    return p.parse_args()


def main():
    args = parse_args()
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    ws = world_size()

    _local = os.path.exists(args.model_id)

    # ------------------------------------------------------------------ VAE
    if is_main():
        print("Loading VAE ...")
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae",
        torch_dtype=torch.bfloat16, local_files_only=_local,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()
    if not args.no_compile:
        vae = torch.compile(vae, dynamic=True)

    # ------------------------------------------------------------------ Text encoders
    if is_main():
        print("Loading text encoders ...")
    text_pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, transformer=None, vae=None,
        torch_dtype=torch.bfloat16, local_files_only=_local,
    ).to(device)
    for enc in (text_pipe.text_encoder, text_pipe.text_encoder_2, text_pipe.text_encoder_3):
        if enc is not None:
            enc.requires_grad_(False)
    if not args.no_compile:
        if text_pipe.text_encoder is not None:
            text_pipe.text_encoder = torch.compile(text_pipe.text_encoder, dynamic=True)
        if text_pipe.text_encoder_2 is not None:
            text_pipe.text_encoder_2 = torch.compile(text_pipe.text_encoder_2, dynamic=True)
        if text_pipe.text_encoder_3 is not None:
            text_pipe.text_encoder_3 = torch.compile(text_pipe.text_encoder_3, dynamic=True)

    # ------------------------------------------------------------------ Transformer
    if is_main():
        print("Loading transformer ...")
    model = PomSD3Transformer2DModel(
        **SD35_MEDIUM_CONFIG,
        n_pom_blocks=24,
        pom_degree=args.pom_degree,
        pom_expand=args.pom_expand,
        pom_n_groups=args.pom_n_groups,
        pom_n_sel_heads=args.pom_n_sel_heads,
        lora_rank=0,
        hybrid_n=args.hybrid_n,
        attention_window_m=4,
    ).to(device=device, dtype=torch.bfloat16)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    model = wrap_model_fsdp(model, local_rank)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ------------------------------------------------------------------ Dummy data
    B = args.batch_size
    H = W = args.image_size
    latent_size = H // 8
    dummy_pixels = torch.randn(B, 3, H, W, device=device, dtype=torch.bfloat16)
    dummy_captions = ["a photo of a cat sitting on a couch"] * B

    # Pre-tokenise for the per-encoder breakdown
    clip_toks_1 = make_clip_tokens(text_pipe.tokenizer, dummy_captions, device)
    clip_toks_2 = make_clip_tokens(text_pipe.tokenizer_2, dummy_captions, device)
    t5_toks = make_t5_tokens(text_pipe.tokenizer_3, dummy_captions, args.max_sequence_length, device)

    # ------------------------------------------------------------------ Timers
    timers = {
        "vae_encode":   StepTimer(),
        "clip_l":       StepTimer(),
        "clip_g":       StepTimer(),
        "t5":           StepTimer(),
        "encode_total": StepTimer(),
        "fwd":          StepTimer(),
        "bwd":          StepTimer(),
        "opt_step":     StepTimer(),
    }

    # ------------------------------------------------------------------ Benchmark loop
    total_steps = args.n_warmup + args.n_steps
    if is_main():
        ckpt_str = " +grad_ckpt" if args.gradient_checkpointing else ""
        cmp_str = " no_compile" if args.no_compile else " +compile"
        print(f"\nRunning {args.n_warmup} warmup + {args.n_steps} timed steps "
              f"(bs={B}, img={H}px, T5_seq={args.max_sequence_length}, "
              f"ws={ws}{ckpt_str}{cmp_str}) ...\n")

    optimizer.zero_grad(set_to_none=True)

    for step in range(total_steps):
        timing = step >= args.n_warmup

        # VAE encode
        t = timers["vae_encode"]
        if timing:
            t.start()
        with torch.no_grad():
            latents = vae.encode(dummy_pixels).latent_dist.sample()
            x_0 = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        if timing:
            t.stop()

        # Per-encoder breakdown (runs the raw encoder, not encode_prompt)
        with torch.no_grad():
            t = timers["clip_l"]
            if timing:
                t.start()
            _ = text_pipe.text_encoder(**clip_toks_1)
            if timing:
                t.stop()

            t = timers["clip_g"]
            if timing:
                t.start()
            _ = text_pipe.text_encoder_2(**clip_toks_2)
            if timing:
                t.stop()

            t = timers["t5"]
            if timing:
                t.start()
            _ = text_pipe.text_encoder_3(**t5_toks)
            if timing:
                t.stop()

        # Full encode_prompt (includes tokenization overhead + all three encoders)
        t = timers["encode_total"]
        if timing:
            t.start()
        with torch.no_grad():
            enc_hs, _, pooled, _ = text_pipe.encode_prompt(
                prompt=dummy_captions,
                prompt_2=dummy_captions,
                prompt_3=dummy_captions,
                max_sequence_length=args.max_sequence_length,
            )
        enc_hs = enc_hs.to(dtype=torch.bfloat16)
        pooled = pooled.to(dtype=torch.bfloat16)
        if timing:
            t.stop()

        # Flow-matching noise
        u = torch.sigmoid(torch.randn(B, device=device))
        ts = (u * 999).clamp(1, 999).long()
        sigma = (ts.float() / 1000).view(B, 1, 1, 1)
        eps = torch.randn_like(x_0)
        x_t = ((1 - sigma) * x_0 + sigma * eps).to(x_0.dtype)

        # Transformer forward
        t = timers["fwd"]
        if timing:
            t.start()
        v_pred = model(
            hidden_states=x_t,
            encoder_hidden_states=enc_hs,
            pooled_projections=pooled,
            timestep=ts,
        ).sample
        if timing:
            t.stop()

        # Loss + backward
        v_target = (eps - x_0).to(v_pred.dtype)
        loss = F.mse_loss(v_pred, v_target)
        t = timers["bwd"]
        if timing:
            t.start()
        loss.backward()
        if timing:
            t.stop()

        # Optimizer step
        t = timers["opt_step"]
        if timing:
            t.start()
        if isinstance(model, FSDP):
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if timing:
            t.stop()

        # Synchronise once per step, then collect all timers
        if timing:
            torch.cuda.synchronize()
            for tmr in timers.values():
                tmr.collect()

        if is_main() and not timing:
            print(f"  warmup {step + 1}/{args.n_warmup} done")

    # ------------------------------------------------------------------ Report
    if not is_main():
        dist.barrier() if dist.is_initialized() else None
        return

    # Build rows
    rows = [
        ("VAE encode",       timers["vae_encode"]),
        ("CLIP-L",           timers["clip_l"]),
        ("CLIP-G",           timers["clip_g"]),
        (f"T5 (seq={args.max_sequence_length})", timers["t5"]),
        ("encode_prompt total", timers["encode_total"]),
        ("Transformer fwd",  timers["fwd"]),
        ("Transformer bwd",  timers["bwd"]),
        ("Optimizer step",   timers["opt_step"]),
    ]

    # Step total: encode_total + fwd + bwd + vae_encode + opt_step
    step_ms = (timers["vae_encode"].mean_ms
               + timers["encode_total"].mean_ms
               + timers["fwd"].mean_ms
               + timers["bwd"].mean_ms
               + timers["opt_step"].mean_ms)

    print()
    print("=" * 72)
    ckpt_str = " +gradient_checkpointing" if args.gradient_checkpointing else ""
    cmp_str  = " (no torch.compile)" if args.no_compile else " (+torch.compile)"
    print(f"  Speed breakdown — bs={B}×{ws} GPUs, {H}px, "
          f"T5_seq={args.max_sequence_length}{ckpt_str}{cmp_str}")
    print("=" * 72)
    print(f"{'Component':<30} {'Mean (ms)':>10} {'±Std':>8} {'% of step':>10}")
    print("-" * 72)
    for label, tmr in rows:
        pct = 100.0 * tmr.mean_ms / step_ms if step_ms > 0 else 0.0
        print(f"  {label:<28} {tmr.mean_ms:>10.1f} {tmr.std_ms:>8.1f} {pct:>9.1f}%")
    print("-" * 72)
    print(f"  {'Step total (estimated)':<28} {step_ms:>10.1f}")
    sps = B * ws / (step_ms / 1000.0)
    print(f"  {'Throughput':<28} {sps:>10.0f} samp/s")
    print("=" * 72)

    # Highlight the biggest component
    biggest = max(rows, key=lambda r: r[1].mean_ms)
    pct_biggest = 100.0 * biggest[1].mean_ms / step_ms
    print(f"\n  Bottleneck: {biggest[0]} ({pct_biggest:.0f}% of step)")

    note_enc = timers["encode_total"].mean_ms
    note_t5  = timers["t5"].mean_ms
    note_fwd = timers["fwd"].mean_ms
    note_bwd = timers["bwd"].mean_ms
    if note_enc > note_fwd + note_bwd:
        print("  → Caching VAE latents and/or text embeddings would give the biggest speedup.")
        if note_t5 > 0.5 * note_enc:
            print(f"  → T5 alone is {100*note_t5/note_enc:.0f}% of encoding cost; "
                  f"try --max_sequence_length 77 if using 256.")
    else:
        print("  → Transformer is the bottleneck; encoding optimisations will have limited effect.")
        if args.gradient_checkpointing:
            fwd_frac = note_fwd / (note_fwd + note_bwd) if note_fwd + note_bwd > 0 else 0
            print(f"  → Backward is {100*(1-fwd_frac):.0f}% of fwd+bwd — "
                  f"gradient checkpointing overhead is visible.")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
