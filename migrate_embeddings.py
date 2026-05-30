"""Convert legacy npz embedding shards to mmap-friendly npy format.

Usage:
    python migrate_embeddings.py --embeddings_dir ./embeddings

The script is resumable: shards already converted are skipped.
Original npz files are removed after successful conversion.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_dir", required=True)
    p.add_argument("--keep_npz", action="store_true",
                   help="Keep original .npz files after conversion (default: delete them)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.embeddings_dir)
    index_path = out_dir / "index.json"

    if not index_path.exists():
        raise FileNotFoundError(f"No index.json in {out_dir}")

    index: dict[str, int] = json.loads(index_path.read_text())

    enc_dir = out_dir / "enc"
    pooled_dir = out_dir / "pooled"
    enc_dir.mkdir(exist_ok=True)
    pooled_dir.mkdir(exist_ok=True)

    new_index: dict[str, int] = {}

    for name, count in tqdm(sorted(index.items()), desc="Converting shards"):
        # Detect whether this entry is old (npz) or already converted (npy)
        npz_path = out_dir / f"{name}.npz"
        enc_npy = enc_dir / f"{name}.npy"
        pooled_npy = pooled_dir / f"{name}.npy"

        if enc_npy.exists() and pooled_npy.exists():
            # Already converted
            new_index[name] = count
            if npz_path.exists() and not args.keep_npz:
                npz_path.unlink()
            continue

        if not npz_path.exists():
            print(f"  WARNING: {npz_path.name} missing, skipping")
            continue

        data = np.load(npz_path)
        enc_arr = data["encoder_hidden_states"]   # (N, seq_len, 4096) float16
        pooled_arr = data["pooled_projections"]   # (N, 2048) float16

        enc_tmp = enc_dir / f"{name}.tmp.npy"
        pooled_tmp = pooled_dir / f"{name}.tmp.npy"
        np.save(enc_tmp, enc_arr)
        np.save(pooled_tmp, pooled_arr)
        enc_tmp.rename(enc_npy)
        pooled_tmp.rename(pooled_npy)

        new_index[name] = count

        if not args.keep_npz:
            npz_path.unlink()

    # Rewrite index with updated entries
    index_path.write_text(json.dumps(new_index, indent=2))
    print(f"\nDone. {len(new_index)} shards in npy format → {out_dir}/")


if __name__ == "__main__":
    main()
