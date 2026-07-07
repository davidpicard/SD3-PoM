#!/usr/bin/env python3
"""Benchmark speed and peak memory: PomSD3 hybrid vs SD3.5 baseline.

Measures per batch size:
  - Single transformer forward pass (synthetic inputs, ms/image)
  - Full denoising pipeline (s/image)
  - Peak GPU memory (GB total)

Outputs a LaTeX table to stdout and a TikZ pgfplots figure to <output>.tex.

Usage:
    python benchmark.py \\
        --checkpoint /path/to/step_XXXXXXX \\
        --model_id /path/to/sd3.5-medium \\
        [--image_size 512] [--steps 28] [--output benchmark]
"""
import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKL, SD3Transformer2DModel, StableDiffusion3Pipeline

from pom_sd3 import PomSD3Transformer2DModel


@dataclass
class Row:
    batch_size: int
    fwd_ms: float    # ms per image (synthetic inputs, no pipeline overhead)
    gen_s: float     # seconds per image (full denoising pipeline)
    mem_gb: float    # peak GPU memory in GB (total, includes model weights)


def _reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _fwd_inputs(bs: int, image_size: int, device, dtype):
    """Minimal synthetic inputs matching the transformer's forward signature."""
    L = image_size // 8
    return dict(
        hidden_states=torch.randn(bs, 16, L, L, device=device, dtype=dtype),
        encoder_hidden_states=torch.randn(bs, 154, 4096, device=device, dtype=dtype),
        pooled_projections=torch.randn(bs, 2048, device=device, dtype=dtype),
        timestep=torch.randint(0, 1000, (bs,), device=device),
    )


def _time_fwd(model, inputs, warmup: int, repeats: int) -> float:
    """Return mean forward-pass time in ms using CUDA events."""
    for _ in range(warmup):
        with torch.no_grad():
            model(**inputs)
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(repeats)]
    for s, e in pairs:
        s.record()
        with torch.no_grad():
            model(**inputs)
        e.record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in pairs) / repeats


def run_bench(label, transformer, pipe, batch_sizes, image_size, steps, device, dtype,
              warmup, repeats):
    print(f"\n--- {label} ---")
    rows = []
    for bs in batch_sizes:
        print(f"  bs={bs:2d} ...", end="", flush=True)

        # Forward pass with synthetic inputs
        _reset()
        inputs = _fwd_inputs(bs, image_size, device, dtype)
        fwd_ms_total = _time_fwd(transformer, inputs, warmup, repeats)

        # Full denoising pipeline (text encoders + denoising loop, no VAE decode)
        _reset()
        pipe.transformer = transformer
        t0 = time.perf_counter()
        with torch.no_grad():
            pipe(
                ["a photo of a cat"] * bs,
                num_inference_steps=steps,
                guidance_scale=4.0,
                height=image_size,
                width=image_size,
                output_type="latent",   # skip VAE decode — same for both models
            )
        torch.cuda.synchronize()
        gen_s_total = time.perf_counter() - t0
        mem_gb = torch.cuda.max_memory_allocated() / 1e9

        row = Row(bs, fwd_ms_total / bs, gen_s_total / bs, mem_gb)
        rows.append(row)
        print(f"  fwd={row.fwd_ms:7.1f} ms/img  "
              f"gen={row.gen_s:.3f} s/img  "
              f"mem={row.mem_gb:.2f} GB")

    pipe.transformer = None
    return rows


# ---------------------------------------------------------------------------
# Output: LaTeX table
# ---------------------------------------------------------------------------

def latex_table(sd35: list[Row], ours: list[Row], image_size: int, steps: int) -> str:
    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        (r"\caption{Speed and peak memory at "
         rf"{image_size}px, {steps} denoising steps. "
         r"Forward pass uses synthetic inputs. "
         r"Generation times the full denoising loop (VAE decode excluded). "
         r"Memory is total peak GPU memory including model weights "
         r"and shared components (text encoders). "
         r"$\times$ = SD3.5 / Ours (higher $= $ faster / smaller for Ours).}"),
        r"\label{tab:benchmark}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        (r" & \multicolumn{3}{c}{Forward pass (ms/img)}"
         r" & \multicolumn{3}{c}{Generation (s/img)}"
         r" & \multicolumn{2}{c}{Peak mem (GB)} \\"),
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
        r"Batch & SD3.5 & Ours & $\times$"
        r" & SD3.5 & Ours & $\times$"
        r" & SD3.5 & Ours \\",
        r"\midrule",
    ]
    for b, o in zip(sd35, ours):
        fwd_x = b.fwd_ms / o.fwd_ms
        gen_x = b.gen_s / o.gen_s
        lines.append(
            rf"{b.batch_size}"
            rf" & {b.fwd_ms:.1f} & {o.fwd_ms:.1f} & {fwd_x:.2f}$\times$"
            rf" & {b.gen_s:.3f} & {o.gen_s:.3f} & {gen_x:.2f}$\times$"
            rf" & {b.mem_gb:.2f} & {o.mem_gb:.2f} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output: TikZ / pgfplots figure
# ---------------------------------------------------------------------------

def _coords(rows: list[Row], attr: str) -> str:
    return " ".join(f"({r.batch_size},{getattr(r, attr):.5g})" for r in rows)


def tikz_figure(sd35: list[Row], ours: list[Row], image_size: int, steps: int) -> str:
    batch_ticks = ",".join(str(r.batch_size) for r in sd35)

    def axis(title, ylabel, attr):
        return (
            rf"\nextgroupplot[title={{{title}}}, ylabel={{{ylabel}}}]" + "\n"
            rf"\addplot+[mark=o,thick] coordinates {{{_coords(sd35, attr)}}};" + "\n"
            rf"\addplot+[mark=square*,thick] coordinates {{{_coords(ours, attr)}}};" + "\n"
        )

    return rf"""% Requires: \usepackage{{pgfplots}}, \usepgfplotslibrary{{groupplots}}
% Add to preamble: \pgfplotsset{{compat=1.18}}
\begin{{figure}}[htb]
\centering
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{
        group size=3 by 1,
        horizontal sep=2.2cm,
    }},
    width=5.2cm, height=4.8cm,
    xtick={{{batch_ticks}}},
    xticklabels={{{batch_ticks}}},
    xlabel={{Batch size}},
    legend style={{font=\small, at={{(0.5,1.05)}}, anchor=south}},
    legend columns=2,
    cycle list name=color list,
    mark options={{solid}},
    grid=major, grid style={{dashed,gray!40}},
]
{axis("Forward pass (ms/img)", r"ms / img", "fwd_ms")}
\legend{{SD3.5, Ours}}

{axis("Generation (s/img)", r"s / img", "gen_s")}

{axis("Peak memory (GB)", r"GB", "mem_gb")}

\end{{groupplot}}
\end{{tikzpicture}}
\caption{{Speed and memory at {image_size}px, {steps} steps.
  \textcolor{{blue}}{{$\circ$}} SD3.5 baseline,
  \textcolor{{red}}{{$\blacksquare$}} Ours (hybrid PoM).}}
\label{{fig:benchmark}}
\end{{figure}}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="PomSD3Transformer2DModel checkpoint directory")
    p.add_argument("--model_id", required=True,
                   help="SD3.5 base model path (VAE, text encoders, and baseline transformer)")
    p.add_argument("--image_size", type=int, default=512,
                   help="Generation resolution in pixels (default 512)")
    p.add_argument("--steps", type=int, default=28,
                   help="Number of denoising steps (default 28)")
    p.add_argument("--batch_sizes", default="1,2,4,8,16",
                   help="Comma-separated batch sizes to benchmark")
    p.add_argument("--warmup", type=int, default=2,
                   help="Warmup forward passes before timing (default 2)")
    p.add_argument("--repeats", type=int, default=5,
                   help="Timed repetitions for forward-pass average (default 5)")
    p.add_argument("--output", default="benchmark",
                   help="Output file prefix; writes <prefix>.tex (default: benchmark)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    local = Path(args.model_id).exists()

    # ---- Shared pipeline components (text encoders; no transformer or VAE yet) ----
    print("Loading text encoders ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, transformer=None, vae=None,
        torch_dtype=dtype, local_files_only=local,
    )
    for enc in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        if enc is not None:
            enc.to(device).eval()
    pipe.set_progress_bar_config(disable=True)

    # ---- Benchmark SD3.5 baseline ----
    print("\nLoading SD3.5 baseline transformer ...")
    baseline = SD3Transformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer",
        torch_dtype=dtype, local_files_only=local,
    ).to(device).eval()
    sd35_rows = run_bench("SD3.5 baseline", baseline, pipe,
                          batch_sizes, args.image_size, args.steps,
                          device, dtype, args.warmup, args.repeats)
    del baseline
    torch.cuda.empty_cache()

    # ---- Benchmark PomSD3 ----
    print("\nLoading PomSD3 transformer ...")
    pom = PomSD3Transformer2DModel.from_pretrained(args.checkpoint).to(device=device, dtype=dtype).eval()
    pom_rows = run_bench("PomSD3 hybrid", pom, pipe,
                         batch_sizes, args.image_size, args.steps,
                         device, dtype, args.warmup, args.repeats)
    del pom
    torch.cuda.empty_cache()

    # ---- Output ----
    table = latex_table(sd35_rows, pom_rows, args.image_size, args.steps)
    figure = tikz_figure(sd35_rows, pom_rows, args.image_size, args.steps)

    print("\n" + "=" * 70)
    print(table)

    out_path = f"{args.output}.tex"
    Path(out_path).write_text(table + "\n\n" + figure)
    print(f"\nLaTeX table + TikZ figure written to {out_path}")


if __name__ == "__main__":
    main()
