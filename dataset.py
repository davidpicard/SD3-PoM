"""Datasets for SD3.5 training.

EmbeddingDataset — pre-computed text embeddings:
    embeddings/
        index.json              # {"shard_00000": N, ...}
        enc/shard_00000.npy    # (N, seq_len, 4096) float16
        pooled/shard_00000.npy # (N, 2048) float16

    Legacy layout (npz shards — loads the full shard into RAM, ~27 GB each):
        embeddings/
            index.json
            shard_00000.npz

    Use migrate_embeddings.py to convert legacy shards to the npy layout.

CaptionDataset — raw captions for online text encoding:
    captions/
        shard_00000.jsonl       # one JSON-encoded string per line
        shard_00001.jsonl
        ...

    Written by download_data.py. Use with caption_collate_fn and a
    StableDiffusion3Pipeline (transformer=None, vae=None) to encode online.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Random-access dataset over precomputed text embeddings.

    With npy layout, each __getitem__ reads only 2 rows (~2 MB) from
    mmap'd files — the OS page cache handles the rest. With legacy npz
    layout, the first access to a shard loads the full file (~27 GB);
    a warning is printed to encourage migration.
    """

    def __init__(
        self,
        embeddings_dir: str | Path,
        latent_height: int = 64,
        latent_width: int = 64,
        latent_channels: int = 16,
        max_timestep: int = 1000,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.latent_shape = (latent_channels, latent_height, latent_width)
        self.max_timestep = max_timestep

        index_path = self.embeddings_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"No index.json found in {embeddings_dir}")
        shard_counts: dict[str, int] = json.loads(index_path.read_text())

        # Detect layout
        enc_dir = self.embeddings_dir / "enc"
        self._use_npy = enc_dir.is_dir()

        if self._use_npy:
            self._enc_paths = [enc_dir / f"{name}.npy" for name in sorted(shard_counts)]
            self._pooled_paths = [
                self.embeddings_dir / "pooled" / f"{name}.npy"
                for name in sorted(shard_counts)
            ]
            # Open mmap handles once — workers inherit them via fork
            self._enc_mm = [
                np.load(str(p), mmap_mode="r") for p in self._enc_paths
            ]
            self._pooled_mm = [
                np.load(str(p), mmap_mode="r") for p in self._pooled_paths
            ]
        else:
            warnings.warn(
                "Legacy npz shards detected. Each shard loads fully into RAM (~27 GB). "
                "Run migrate_embeddings.py to convert to npy format.",
                stacklevel=2,
            )
            self._npz_paths = [
                self.embeddings_dir / f"{name}.npz" for name in sorted(shard_counts)
            ]
            self._npz_cache: dict[int, dict] = {}

        counts = [shard_counts[k] for k in sorted(shard_counts)]
        self._index: list[tuple[int, int]] = [
            (si, i) for si, n in enumerate(counts) for i in range(n)
        ]
        print(f"EmbeddingDataset: {len(self._index)} samples, "
              f"{'npy/mmap' if self._use_npy else 'npz (legacy)'} format")

    def __len__(self) -> int:
        return len(self._index)

    def _load_npz(self, shard_idx: int) -> dict:
        if shard_idx not in self._npz_cache:
            if len(self._npz_cache) > 1:
                oldest = next(iter(self._npz_cache))
                del self._npz_cache[oldest]
            data = np.load(self._npz_paths[shard_idx])
            self._npz_cache[shard_idx] = {
                "enc": data["encoder_hidden_states"],
                "pooled": data["pooled_projections"],
            }
        return self._npz_cache[shard_idx]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        shard_idx, sample_idx = self._index[idx]

        if self._use_npy:
            # .copy() makes the slice contiguous and writable (required by torch)
            enc = torch.from_numpy(self._enc_mm[shard_idx][sample_idx].copy()).float()
            pooled = torch.from_numpy(self._pooled_mm[shard_idx][sample_idx].copy()).float()
        else:
            shard = self._load_npz(shard_idx)
            enc = torch.from_numpy(shard["enc"][sample_idx]).float()
            pooled = torch.from_numpy(shard["pooled"][sample_idx]).float()

        return {
            "hidden_states": torch.randn(self.latent_shape),
            "encoder_hidden_states": enc,
            "pooled_projections": pooled,
            "timestep": torch.randint(0, self.max_timestep, (1,)).item(),
        }


class CaptionDataset(Dataset):
    """Random-access dataset over raw caption shards (shard_XXXXX.jsonl).

    Each shard is loaded into memory on first access and cached; at most two
    shards are kept in memory at once.  Captions are returned as plain strings
    and must be encoded online — use caption_collate_fn with the DataLoader.
    """

    def __init__(self, captions_dir: str | Path):
        shards = sorted(Path(captions_dir).glob("shard_*.jsonl"))
        if not shards:
            raise FileNotFoundError(f"No shard_*.jsonl files in {captions_dir}")
        self._shards = shards
        self._cache: dict[int, list[str]] = {}
        counts = [self._count_lines(s) for s in shards]
        self._index: list[tuple[int, int]] = [
            (si, li) for si, n in enumerate(counts) for li in range(n)
        ]
        print(f"CaptionDataset: {len(self._index)} captions in {len(shards)} shards")

    @staticmethod
    def _count_lines(path: Path) -> int:
        with open(path, encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())

    def _get_shard(self, si: int) -> list[str]:
        if si not in self._cache:
            if len(self._cache) > 1:
                del self._cache[next(iter(self._cache))]
            with open(self._shards[si], encoding="utf-8") as f:
                self._cache[si] = [json.loads(ln) for ln in f if ln.strip()]
        return self._cache[si]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        si, li = self._index[idx]
        return {"caption": self._get_shard(si)[li]}


def caption_collate_fn(batch: list[dict]) -> dict:
    """Collate CaptionDataset items into a batch with a list of caption strings."""
    return {"caption": [item["caption"] for item in batch]}
