#!/usr/bin/env python3
"""
Per-component speed benchmark for the pixel-space SD3-PoM training loop.

Single GPU:
    python speed_test.py --model_id /path/to/sd3.5-medium

4-GPU node:
    torchrun --nproc_per_node=4 speed_test.py --model_id /path/to/sd3.5-medium

Measures text encoding, transformer fwd, transformer bwd, and optimizer step.
No VAE — the pixel model operates directly in pixel space.

Flags to A/B test (all off by default so each adds cleanly over the baseline):
  --forward_prefetch       FSDP forward all-gather prefetch (overlaps comm + compute)
  --no_limit_all_gathers   remove FSDP cap on concurrent in-flight all-gathers
  --use_orig_params        FSDP use_orig_params=True (required for --compile_model)
  --compile_model          torch.compile the transformer (needs --use_orig_params)
  --cudnn_benchmark        cuDNN autotuning for Conv2d (PixelPatchEmbed)
  --no_compile_text        skip torch.compile on text encoders
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

from diffusers import StableDiffusion3Pipeline
from pom_sd3.convert import build_pixel_grouped


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def ws():
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier():
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------

def wrap_fsdp(model, local_rank, args):
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
    fsdp_kwargs = dict(
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        device_id=local_rank,
        forward_prefetch=args.forward_prefetch,
        limit_all_gathers=not args.no_limit_all_gathers,
        use_orig_params=args.use_orig_params,
    )

    gpus_per_node = args.gpus_per_node
    world = ws()
    if gpus_per_node is not None and gpus_per_node < world:
        rank      = dist.get_rank()
        num_nodes = world // gpus_per_node
        node_idx  = rank // gpus_per_node
        all_intra = [dist.new_group(list(range(n * gpus_per_node, (n + 1) * gpus_per_node)))
                     for n in range(num_nodes)]
        all_inter = [dist.new_group(list(range(g, world, gpus_per_node)))
                     for g in range(gpus_per_node)]
        intra_group = all_intra[node_idx]
        inter_group = all_inter[rank % gpus_per_node]
        if is_main():
            print(f"HYBRID_SHARD: {gpus_per_node} GPUs/node × {num_nodes} nodes")
        return FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                    process_group=(intra_group, inter_group), **fsdp_kwargs)

    return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, **fsdp_kwargs)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class StepTimer:
    def __init__(self):
        self.samples: list[float] = []
        self._s = self._e = None

    def start(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._s.record()

    def stop(self):
        self._e = torch.cuda.Event(enable_timing=True)
        self._e.record()

    def collect(self):
        if self._s is not None and self._e is not None:
            self.samples.append(self._s.elapsed_time(self._e))
            self._s = self._e = None

    @property
    def mean_ms(self):
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def std_ms(self):
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True,
                   help="SD3.5-medium path (provides text encoders only; no VAE used)")
    # Pixel model
    p.add_argument("--image_size",    type=int, default=512)
    p.add_argument("--patch_size",    type=int, default=32)
    p.add_argument("--bottleneck_dim",type=int, default=256)
    p.add_argument("--pom_degree",    type=int, default=4)
    p.add_argument("--pom_expand",    type=int, default=2)
    p.add_argument("--pom_n_groups",  type=int, default=1)
    p.add_argument("--pom_n_sel_heads", type=int, default=24)
    # Training dims
    p.add_argument("--batch_size",          type=int, default=32)
    p.add_argument("--max_sequence_length", type=int, default=77)
    # FSDP flags
    p.add_argument("--gpus_per_node",       type=int, default=None)
    p.add_argument("--forward_prefetch",    action="store_true",
                   help="FSDP forward all-gather prefetch")
    p.add_argument("--no_limit_all_gathers", action="store_true",
                   help="Remove FSDP limit on concurrent all-gathers")
    p.add_argument("--use_orig_params",     action="store_true",
                   help="FSDP use_orig_params (required for --compile_model)")
    # Compile
    p.add_argument("--compile_model",  action="store_true",
                   help="torch.compile the transformer (needs --use_orig_params)")
    p.add_argument("--no_compile_text", action="store_true",
                   help="Skip torch.compile on text encoders")
    # Other
    p.add_argument("--cudnn_benchmark", action="store_true",
                   help="cuDNN autotuning (helps PixelPatchEmbed Conv2d)")
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_steps",  type=int, default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    _local = os.path.exists(args.model_id)

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
            enc.eval()
    if not args.no_compile_text:
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            enc = getattr(text_pipe, attr, None)
            if enc is not None:
                setattr(text_pipe, attr, torch.compile(enc, dynamic=True))

    # ------------------------------------------------------------------ Pixel transformer
    if is_main():
        print("Building pixel transformer ...")
    model = build_pixel_grouped(
        patch_size=args.patch_size,
        bottleneck_dim=args.bottleneck_dim,
        pom_degree=args.pom_degree,
        pom_expand=args.pom_expand,
        pom_n_groups=args.pom_n_groups,
        pom_n_sel_heads=args.pom_n_sel_heads,
        torch_dtype=torch.bfloat16,
        device=device,
    )
    model = wrap_fsdp(model, local_rank, args)
    if args.compile_model:
        if not args.use_orig_params:
            if is_main():
                print("WARNING: --compile_model without --use_orig_params may not work correctly.")
        model = torch.compile(model, dynamic=False)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ------------------------------------------------------------------ Dummy data
    B = args.batch_size
    H = W = args.image_size
    dummy_pixels  = torch.randn(B, 3, H, W, device=device, dtype=torch.bfloat16)
    dummy_captions = ["a photo of a cat sitting on a couch"] * B

    # Pre-tokenise per encoder for the breakdown timers
    def make_tokens(tokenizer, captions, max_len):
        toks = tokenizer(captions, padding="max_length", max_length=max_len,
                         truncation=True, return_tensors="pt")
        return {k: v.to(device) for k, v in toks.items()
                if k in ("input_ids", "attention_mask")}

    clip_toks_1 = make_tokens(text_pipe.tokenizer,   dummy_captions, 77)
    clip_toks_2 = make_tokens(text_pipe.tokenizer_2,  dummy_captions, 77)
    t5_toks     = make_tokens(text_pipe.tokenizer_3,  dummy_captions, args.max_sequence_length)

    # ------------------------------------------------------------------ Timers
    timers = {
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

    flags = []
    if args.forward_prefetch:      flags.append("forward_prefetch")
    if args.no_limit_all_gathers:  flags.append("no_limit_all_gathers")
    if args.use_orig_params:       flags.append("use_orig_params")
    if args.compile_model:         flags.append("compile_model")
    if args.cudnn_benchmark:       flags.append("cudnn_benchmark")
    if args.no_compile_text:       flags.append("no_compile_text")
    flag_str = ("+[" + ", ".join(flags) + "]") if flags else "baseline"

    if is_main():
        print(f"\nRunning {args.n_warmup} warmup + {args.n_steps} timed steps "
              f"(bs={B}×{ws()} GPUs, {H}px, patch={args.patch_size}, "
              f"T5_seq={args.max_sequence_length}, {flag_str}) ...\n")

    optimizer.zero_grad(set_to_none=True)

    for step in range(total_steps):
        timing = step >= args.n_warmup

        # Per-encoder breakdown (raw encoder forward, not full encode_prompt)
        with torch.no_grad():
            t = timers["clip_l"]
            if timing: t.start()
            _ = text_pipe.text_encoder(**clip_toks_1)
            if timing: t.stop()

            t = timers["clip_g"]
            if timing: t.start()
            _ = text_pipe.text_encoder_2(**clip_toks_2)
            if timing: t.stop()

            t = timers["t5"]
            if timing: t.start()
            _ = text_pipe.text_encoder_3(**t5_toks)
            if timing: t.stop()

        # Full encode_prompt (tokenisation + all three encoders)
        t = timers["encode_total"]
        if timing: t.start()
        with torch.no_grad():
            enc_hs, _, pooled, _ = text_pipe.encode_prompt(
                prompt=dummy_captions,
                prompt_2=dummy_captions,
                prompt_3=dummy_captions,
                max_sequence_length=args.max_sequence_length,
            )
        enc_hs = enc_hs.to(dtype=torch.bfloat16)
        pooled = pooled.to(dtype=torch.bfloat16)
        if timing: t.stop()

        # JiT flow matching: t=0 noise, t=1 clean
        # t_cont ~ LogitNormal(-0.8, 0.8) as in training
        t_cont = torch.sigmoid(
            torch.randn(B, device=device) * 0.8 + (-0.8)
        )
        t_int = ((1.0 - t_cont) * 999).long().clamp(0, 999)
        t_view = t_cont.view(B, 1, 1, 1)
        x_0 = dummy_pixels
        eps = torch.randn_like(x_0)
        z_t = t_view * x_0 + (1.0 - t_view) * eps

        # Transformer forward
        t = timers["fwd"]
        if timing: t.start()
        x_pred = model(
            hidden_states=z_t,
            encoder_hidden_states=enc_hs,
            pooled_projections=pooled,
            timestep=t_int,
        ).sample
        if timing: t.stop()

        # x-prediction loss with (1-t)^2 weighting (JiT paper, clamp 5e-2)
        one_minus_t = (1.0 - t_cont).clamp(min=5e-2)
        loss_per = F.mse_loss(x_pred, x_0, reduction="none").mean(dim=(1, 2, 3))
        loss = (loss_per / one_minus_t.pow(2)).mean()

        t = timers["bwd"]
        if timing: t.start()
        loss.backward()
        if timing: t.stop()

        # Optimizer step
        t = timers["opt_step"]
        if timing: t.start()
        if isinstance(model, FSDP):
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if timing: t.stop()

        if timing:
            torch.cuda.synchronize()
            for tmr in timers.values():
                tmr.collect()

        if is_main() and not timing:
            print(f"  warmup {step + 1}/{args.n_warmup} done")

    # ------------------------------------------------------------------ Report
    if not is_main():
        barrier()
        return

    rows = [
        ("CLIP-L",                timers["clip_l"]),
        ("CLIP-G",                timers["clip_g"]),
        (f"T5 (seq={args.max_sequence_length})", timers["t5"]),
        ("encode_prompt total",   timers["encode_total"]),
        ("Transformer fwd",       timers["fwd"]),
        ("Transformer bwd",       timers["bwd"]),
        ("Optimizer step",        timers["opt_step"]),
    ]

    step_ms = (timers["encode_total"].mean_ms
               + timers["fwd"].mean_ms
               + timers["bwd"].mean_ms
               + timers["opt_step"].mean_ms)

    print()
    print("=" * 76)
    print(f"  Pixel-space speed breakdown — {flag_str}")
    print(f"  bs={B}×{ws()} GPUs · {H}px · patch={args.patch_size} · "
          f"T5_seq={args.max_sequence_length}")
    print("=" * 76)
    print(f"  {'Component':<32} {'Mean (ms)':>10} {'±Std':>8} {'% of step':>10}")
    print("-" * 76)
    for label, tmr in rows:
        pct = 100.0 * tmr.mean_ms / step_ms if step_ms > 0 else 0.0
        print(f"  {label:<32} {tmr.mean_ms:>10.1f} {tmr.std_ms:>8.1f} {pct:>9.1f}%")
    print("-" * 76)
    print(f"  {'Step total (estimated)':<32} {step_ms:>10.1f}")
    sps = B * ws() / (step_ms / 1000.0)
    print(f"  {'Throughput':<32} {sps:>10.0f} samp/s")
    print("=" * 76)

    biggest = max(rows, key=lambda r: r[1].mean_ms)
    pct_biggest = 100.0 * biggest[1].mean_ms / step_ms
    print(f"\n  Bottleneck: {biggest[0]} ({pct_biggest:.0f}% of step)")

    enc_ms  = timers["encode_total"].mean_ms
    fwdbwd  = timers["fwd"].mean_ms + timers["bwd"].mean_ms
    t5_ms   = timers["t5"].mean_ms
    if enc_ms > fwdbwd:
        print("  → Text encoding dominates. Caching embeddings would give the biggest gain.")
        if t5_ms > 0.5 * enc_ms:
            print(f"  → T5 is {100*t5_ms/enc_ms:.0f}% of encoding cost — "
                  f"rank-0-only encoding or pre-computation is the key lever.")
    else:
        print("  → Transformer fwd+bwd dominates. "
              "FSDP overlap and torch.compile are the main levers.")

    barrier()


if __name__ == "__main__":
    main()
