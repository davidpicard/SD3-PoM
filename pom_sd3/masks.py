"""2D windowed attention mask for local-attention transformer blocks.

The image patch sequence is assumed to be in row-major order.  A square
2D window of half-size m means token i attends to every token j whose
grid-row and grid-column are both within m of i's own row/column.
Text tokens always attend globally (handled by the caller).
"""
import math

import torch
import torch.nn.functional as F

# Module-level cache: (n_img, m, device_str, dtype) → (N, N) tensor
_MASK_CACHE: dict = {}


def build_2d_window_mask(
    n_img: int,
    m: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an (n_img, n_img) additive attention mask for a 2D window.

    Entry [i, j] is 0.0 if token j falls within the (2m+1)×(2m+1) grid
    neighbourhood of token i, and -inf otherwise.
    Assumes a square patch grid: H = W = isqrt(n_img).
    """
    H = W = int(math.isqrt(n_img))
    if H * W != n_img:
        raise ValueError(
            f"n_img={n_img} is not a perfect square — cannot infer 2D grid shape."
        )
    device = torch.device(device)
    key = (n_img, m, str(device), dtype)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]

    idx = torch.arange(n_img, device=device)
    ri = idx // W          # row of each token
    ci = idx % W           # column of each token
    dr = (ri.unsqueeze(1) - ri.unsqueeze(0)).abs()   # (N, N)
    dc = (ci.unsqueeze(1) - ci.unsqueeze(0)).abs()   # (N, N)
    in_window = (dr <= m) & (dc <= m)
    mask = torch.zeros(n_img, n_img, device=device, dtype=dtype)
    mask[~in_window] = float('-inf')

    _MASK_CACHE[key] = mask
    return mask


def build_joint_mask(
    n_img: int,
    n_txt: int,
    m: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (joint_mask, img_mask) for a JointLocalAttnBlock.

    joint_mask: (1, 1, N+T, N+T) additive mask for the joint image+text attention.
        Image-to-image entries are windowed; all entries involving text are 0.
    img_mask:   (1, 1, N, N) additive mask for the image-only dual attention.
    """
    img_mask = build_2d_window_mask(n_img, m, device, dtype)   # (N, N)
    joint_mask = F.pad(img_mask, (0, n_txt, 0, n_txt), value=0.0)  # (N+T, N+T)
    return joint_mask.unsqueeze(0).unsqueeze(0), img_mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    def _count_attended(mask: torch.Tensor, i: int) -> int:
        """Number of finite (non-masked) entries in row i."""
        return int((mask[i] != float('-inf')).sum().item())

    failures = []

    for m in (1, 2, 4):
        for H in (4, 8, 16, 32):
            N = H * H
            mask = build_2d_window_mask(N, m, device="cpu", dtype=torch.float32)

            # Shape
            if mask.shape != (N, N):
                failures.append(f"m={m} H={H}: wrong shape {mask.shape}")
                continue

            # Symmetry
            if not torch.equal(mask, mask.T):
                failures.append(f"m={m} H={H}: mask is not symmetric")

            # Corner token (0,0): window is clamped to grid boundaries
            corner_expected = min(m + 1, H) * min(m + 1, H)  # square grid, W=H
            corner_got = _count_attended(mask, 0)
            if corner_got != corner_expected:
                failures.append(
                    f"m={m} H={H}: corner token attends to {corner_got}, "
                    f"expected {corner_expected}"
                )

            # Centre token at (H//2, H//2): full (2m+1)x(2m+1) window
            # (only when H is large enough that the window doesn't hit a boundary)
            if H > 2 * m:
                centre = (H // 2) * H + (H // 2)
                centre_expected = (2 * m + 1) ** 2
                centre_got = _count_attended(mask, centre)
                if centre_got != centre_expected:
                    failures.append(
                        f"m={m} H={H}: centre token attends to {centre_got}, "
                        f"expected {centre_expected}"
                    )

            # Self-attention (diagonal) is always unmasked
            if not (mask.diagonal() == 0.0).all():
                failures.append(f"m={m} H={H}: diagonal contains -inf entries")

    # Joint mask padding test
    N, T, m = 16, 6, 1
    jm, im = build_joint_mask(N, T, m, device="cpu")
    if jm.shape != (1, 1, N + T, N + T):
        failures.append(f"joint_mask shape {jm.shape} != (1,1,{N+T},{N+T})")
    if im.shape != (1, 1, N, N):
        failures.append(f"img_mask shape {im.shape} != (1,1,{N},{N})")
    # Text rows/cols in joint mask must be all-zero
    if not (jm[0, 0, N:, :] == 0).all() or not (jm[0, 0, :, N:] == 0).all():
        failures.append("joint_mask: text rows/cols contain non-zero entries")

    if failures:
        print("FAIL")
        for f in failures:
            print(" ", f)
        sys.exit(1)
    else:
        print("PASS — all mask tests passed")
