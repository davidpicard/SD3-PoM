"""Dataset of precomputed SD3.5 text embeddings paired with random noise latents."""
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Loads precomputed text embeddings from a directory of .npz shards.

    Each shard is a dict with keys:
        encoder_hidden_states: (N, L, 4096) float16
        pooled_projections:    (N, 2048)    float16

    During __getitem__, returns a random noise latent paired with one embedding.
    The noise is freshly sampled each call so the model sees different inputs
    across epochs without storing latents on disk.
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

        self._shards: list[Path] = sorted(self.embeddings_dir.glob("*.npz"))
        if not self._shards:
            raise FileNotFoundError(f"No .npz shards found in {embeddings_dir}")

        # Build index: (shard_idx, sample_idx_within_shard)
        self._index: list[tuple[int, int]] = []
        self._shard_cache: dict[int, dict] = {}  # LRU would be nicer; dict is fine for now
        for shard_idx, shard_path in enumerate(self._shards):
            data = np.load(shard_path)
            n = data["encoder_hidden_states"].shape[0]
            self._index.extend((shard_idx, i) for i in range(n))

        print(f"EmbeddingDataset: {len(self._index)} samples across {len(self._shards)} shards")

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_idx: int) -> dict:
        if shard_idx not in self._shard_cache:
            if len(self._shard_cache) > 4:
                # Evict oldest
                oldest = next(iter(self._shard_cache))
                del self._shard_cache[oldest]
            data = np.load(self._shards[shard_idx])
            self._shard_cache[shard_idx] = {
                "encoder_hidden_states": torch.from_numpy(data["encoder_hidden_states"]),
                "pooled_projections": torch.from_numpy(data["pooled_projections"]),
            }
        return self._shard_cache[shard_idx]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        shard_idx, sample_idx = self._index[idx]
        shard = self._load_shard(shard_idx)

        encoder_hidden_states = shard["encoder_hidden_states"][sample_idx].float()
        pooled_projections = shard["pooled_projections"][sample_idx].float()

        # Fresh random noise latent each time
        noise = torch.randn(self.latent_shape)
        timestep = torch.randint(0, self.max_timestep, (1,)).item()

        return {
            "hidden_states": noise,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
        }
