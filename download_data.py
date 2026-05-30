"""Download captions and/or model weights to local storage for offline training.

Run this on the frontend (internet-connected) node BEFORE moving to compute nodes.

Examples
--------
# Download captions only (fast, text-only):
python download_data.py --captions \
    --dataset_name laion/laion-aesthetics-v2-5plus \
    --caption_column TEXT \
    --max_samples 5000000 \
    --captions_dir ./captions

# Download model weights only:
python download_data.py --model \
    --model_id stabilityai/stable-diffusion-3.5-medium \
    --model_dir ./models/sd3.5-medium

# Download both:
python download_data.py --captions --model [... flags above ...]
"""
import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", action="store_true", help="Download captions dataset")
    p.add_argument("--model", action="store_true", help="Download model weights")

    # Caption options
    p.add_argument("--dataset_name", default="laion/laion-aesthetics-v2-5plus")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--caption_column", default="TEXT")
    p.add_argument("--max_samples", type=int, default=5_000_000)
    p.add_argument("--shard_size", type=int, default=100_000,
                   help="Number of captions per .jsonl shard file")
    p.add_argument("--captions_dir", default="./captions")

    # Model options
    p.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--model_dir", default="./models/sd3.5-medium")
    p.add_argument("--skip_vae", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip VAE weights (not needed for distillation training; pass --no-skip-vae to include)")

    return p.parse_args()


def download_captions(args):
    """Stream the dataset and save only caption text as .jsonl shards."""
    from datasets import load_dataset

    out_dir = Path(args.captions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing shards to allow resuming
    existing = sorted(out_dir.glob("shard_*.jsonl"))
    start_shard = len(existing)
    already_saved = start_shard * args.shard_size
    if already_saved > 0:
        print(f"Resuming: {already_saved} captions already saved in {start_shard} shards.")

    print(f"Streaming {args.dataset_name} [{args.dataset_split}], caption column='{args.caption_column}'")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split, streaming=True)

    shard_idx = start_shard
    buf = []
    total = already_saved
    skipped = 0
    f = None

    def open_shard():
        path = out_dir / f"shard_{shard_idx:05d}.jsonl"
        return open(path, "w", encoding="utf-8")

    try:
        for sample in tqdm(dataset, desc="Downloading captions", initial=already_saved):
            # Skip already-saved samples
            if skipped < already_saved:
                skipped += 1
                continue

            caption = sample.get(args.caption_column) or ""
            caption = caption.strip()
            if not caption:
                continue

            buf.append(caption)
            if len(buf) >= args.shard_size:
                if f is None:
                    f = open_shard()
                for c in buf:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
                f.close()
                f = None
                total += len(buf)
                print(f"  Saved shard_{shard_idx:05d}.jsonl ({total} total)")
                buf.clear()
                shard_idx += 1

            if total + len(buf) >= args.max_samples:
                break

    finally:
        if buf:
            if f is None:
                f = open_shard()
            for c in buf:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
            f.close()
            total += len(buf)
            shard_idx += 1

    # Write a metadata file so precompute_embeddings.py knows what's here
    meta = {
        "dataset_name": args.dataset_name,
        "caption_column": args.caption_column,
        "total_captions": total,
        "n_shards": shard_idx,
        "shard_size": args.shard_size,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. {total} captions in {shard_idx} shards → {out_dir}/")


def download_model(args):
    """Download model weights using huggingface_hub snapshot_download."""
    from huggingface_hub import snapshot_download

    out_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ignore = []
    if args.skip_vae:
        ignore += ["vae/*", "vae_decoder/*", "vae_encoder/*"]

    print(f"Downloading {args.model_id} → {out_dir}")
    print(f"  (skipping: {ignore if ignore else 'nothing'})")

    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(out_dir),
        ignore_patterns=ignore,
    )
    print(f"\nModel weights saved to {out_dir}/")
    print("  Pass --model_dir to precompute_embeddings.py and train.py to use offline.")


def main():
    args = parse_args()

    if not args.captions and not args.model:
        print("Nothing to do — pass --captions and/or --model.")
        return

    if args.captions:
        download_captions(args)

    if args.model:
        download_model(args)


if __name__ == "__main__":
    main()
