"""Precompute SD3.5 text embeddings from downloaded captions.

Run on a compute node (or frontend) AFTER download_data.py has run.
No internet access required — all weights and captions must be local.

Usage
-----
python precompute_embeddings.py \
    --model_dir ./models/sd3.5-medium \
    --captions_dir ./captions \
    --output_dir ./embeddings \
    --batch_size 32
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True,
                   help="Local path to SD3.5 model (from download_data.py --model)")
    p.add_argument("--captions_dir", required=True,
                   help="Local path to caption shards (from download_data.py --captions)")
    p.add_argument("--output_dir", required=True,
                   help="Where to write embedding .npz shards")
    p.add_argument("--shard_size", type=int, default=10_000,
                   help="Embeddings per output shard")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def iter_captions(captions_dir: Path):
    """Yield captions from all .jsonl shards in order."""
    shards = sorted(captions_dir.glob("shard_*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.jsonl files in {captions_dir}")
    for shard in shards:
        with open(shard, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    captions_dir = Path(args.captions_dir)

    # Count total captions from metadata if available
    meta_path = captions_dir / "meta.json"
    total_captions = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        total_captions = meta.get("total_captions")
        print(f"Caption dataset: {total_captions} captions from {meta.get('dataset_name', '?')}")

    print(f"Loading text encoders from {args.model_dir} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        transformer=None,
        vae=None,
        local_files_only=True,
    )
    pipe = pipe.to(args.device)

    # Check for existing output shards to allow resuming
    existing_shards = sorted(out_dir.glob("shard_*.npz"))
    n_done = len(existing_shards) * args.shard_size
    print(f"Output dir: {out_dir} — {len(existing_shards)} shards already done ({n_done} embeddings), resuming.")

    caption_iter = iter_captions(captions_dir)

    # Skip already-processed captions
    for _ in range(n_done):
        try:
            next(caption_iter)
        except StopIteration:
            print("All captions already processed.")
            return

    enc_hs_buf, pooled_buf = [], []
    shard_idx = len(existing_shards)
    total = n_done
    batch: list[str] = []

    def flush_shard():
        nonlocal shard_idx
        path = out_dir / f"shard_{shard_idx:05d}.npz"
        np.savez(
            path,
            encoder_hidden_states=np.stack(enc_hs_buf).astype(np.float16),
            pooled_projections=np.stack(pooled_buf).astype(np.float16),
        )
        enc_hs_buf.clear()
        pooled_buf.clear()
        shard_idx += 1

    def process_batch(captions: list[str]):
        prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
            prompt=captions,
            prompt_2=captions,
            prompt_3=captions,
        )
        enc_hs_buf.extend(prompt_embeds.cpu().float().numpy())
        pooled_buf.extend(pooled_embeds.cpu().float().numpy())

    pbar = tqdm(caption_iter, desc="Computing embeddings", total=total_captions, initial=total)
    for caption in pbar:
        batch.append(caption)
        if len(batch) < args.batch_size:
            continue

        process_batch(batch)
        total += len(batch)
        batch.clear()

        if len(enc_hs_buf) >= args.shard_size:
            flush_shard()
            pbar.set_postfix(shards=shard_idx)

        if args.max_samples and total >= args.max_samples:
            break

    # Flush remaining
    if batch:
        process_batch(batch)
        total += len(batch)
    if enc_hs_buf:
        flush_shard()

    print(f"\nDone. {total} embeddings in {shard_idx} shards → {out_dir}/")


if __name__ == "__main__":
    main()
