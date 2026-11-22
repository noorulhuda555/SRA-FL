"""Model-update compression helpers (Module 4)."""

from __future__ import annotations

import numpy as np


def compress_parameters(
    parameters: list[np.ndarray],
    compression_level: int,
) -> list[np.ndarray]:
    """
    Compress parameter arrays.

    Levels
    ------
    0 : no compression
    1 : float16 round-trip
    2 : keep top 30% magnitudes, then float16
    """
    level = int(compression_level)
    if level <= 0:
        return [np.asarray(p, dtype=np.float32) for p in parameters]
    if level == 1:
        return [np.asarray(p, dtype=np.float16).astype(np.float32) for p in parameters]

    # level >= 2 : sparsify then float16
    out: list[np.ndarray] = []
    for layer in parameters:
        arr = np.asarray(layer, dtype=np.float32)
        flat = arr.reshape(-1)
        if flat.size == 0:
            out.append(arr)
            continue
        k = max(1, int(0.3 * flat.size))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        mask = np.zeros_like(flat, dtype=bool)
        mask[idx] = True
        sparse = np.where(mask, flat, 0.0).reshape(arr.shape)
        out.append(sparse.astype(np.float16).astype(np.float32))
    return out


def compression_ratio(level: int) -> float:
    """Approximate retained payload fraction."""
    if level <= 0:
        return 1.0
    if level == 1:
        return 0.5
    return 0.3 * 0.5
