"""Interactive image generation from a PomSD3Transformer2DModel checkpoint.

Usage:
    python generate.py --checkpoint /path/to/step_0005000 \
                       --model_id /path/to/sd3.5-medium
"""
import argparse
import re
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusion3Pipeline

from pom_sd3 import PomSD3Transformer2DModel


def _increment_filename(name: str) -> str:
    """Return name with a trailing _NNN counter incremented (or appended as _001)."""
    stem, _, ext = name.rpartition(".")
    if not stem:          # no dot — treat whole name as stem, no extension
        stem, ext = name, ""
    else:
        ext = "." + ext

    m = re.search(r"^(.*?)_(\d+)$", stem)
    if m:
        base, n = m.group(1), int(m.group(2))
        width = len(m.group(2))
        return f"{base}_{n + 1:0{width}d}{ext}"
    return f"{stem}_001{ext}"


def _prompt(text: str, default: str | None) -> str:
    """Read a line from stdin, returning default on empty input.  Raises EOFError on Ctrl-D."""
    if default is not None:
        display = f"{text} [{default}]: "
    else:
        display = f"{text}: "
    line = input(display).strip()
    if not line:
        if default is None:
            raise ValueError(f"{text} cannot be empty and has no default")
        return default
    return line


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to a PomSD3Transformer2DModel checkpoint directory "
                        "(e.g. output_dir/step_0005000)")
    p.add_argument("--model_id", required=True,
                   help="Path (or HF repo) to the base SD3.5-medium model "
                        "(provides VAE and text encoders)")
    p.add_argument("--image_size", type=int, default=None,
                   help="Output resolution in pixels. Defaults to the checkpoint's "
                        "training resolution (sample_size * patch_size * 8).")
    p.add_argument("--steps", type=int, default=28, help="Denoising steps (default 28)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    print(f"Loading transformer from {args.checkpoint} ...")
    model = PomSD3Transformer2DModel.from_pretrained(args.checkpoint).to(device=device, dtype=dtype)
    model.eval()

    # Infer resolution from config if not specified
    if args.image_size is None:
        cfg = model.config
        args.image_size = cfg.sample_size * cfg.patch_size * 8
        print(f"  Resolution: {args.image_size}px (from checkpoint config)")

    local_files_only = Path(args.model_id).exists()
    print(f"Loading VAE from {args.model_id} ...")
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae",
        torch_dtype=dtype, local_files_only=local_files_only,
    ).to(device)
    vae.eval()

    print(f"Loading text encoders from {args.model_id} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, transformer=model, vae=vae,
        torch_dtype=dtype, local_files_only=local_files_only,
    )
    pipe.set_progress_bar_config(disable=False)

    print("\nReady. Enter prompts interactively. Press Ctrl-D to quit.\n")

    last_cfg = 4.0
    last_filename: str | None = None

    while True:
        print("-" * 60)
        try:
            prompt = _prompt("Prompt", default=None)
        except (EOFError, ValueError):
            print("\nDone.")
            break

        try:
            cfg_str = _prompt("CFG scale", default=str(last_cfg))
        except EOFError:
            print("\nDone.")
            break
        try:
            cfg = float(cfg_str)
        except ValueError:
            print(f"  Invalid CFG '{cfg_str}' — using {last_cfg}")
            cfg = last_cfg

        default_filename = _increment_filename(last_filename) if last_filename else "output_001.png"
        try:
            filename = _prompt("Output file", default=default_filename)
        except EOFError:
            print("\nDone.")
            break

        print(f"  Generating {args.image_size}×{args.image_size} "
              f"({args.steps} steps, cfg={cfg}) ...")
        with torch.no_grad():
            result = pipe(
                prompt,
                num_inference_steps=args.steps,
                guidance_scale=cfg,
                height=args.image_size,
                width=args.image_size,
            )
        img = result.images[0]

        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        print(f"  Saved → {out_path.resolve()}")

        last_cfg = cfg
        last_filename = filename


if __name__ == "__main__":
    main()
