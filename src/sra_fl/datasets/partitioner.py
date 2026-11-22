"""IID and non-IID (Dirichlet) client data partitioning."""

from __future__ import annotations

from typing import Any

import numpy as np


def _iid_indices(labels: np.ndarray, num_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labels))
    return [np.array(part) for part in np.array_split(indices, num_clients)]


def _dirichlet_indices(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 10,
) -> list[np.ndarray]:
    """
    Partition sample indices with a Dirichlet distribution over class proportions.

    Retries until every client receives at least `min_size` samples.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    labels = labels.astype(np.int64)

    for attempt in range(100):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        for class_id in range(num_classes):
            class_idx = np.where(labels == class_id)[0]
            rng.shuffle(class_idx)
            proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
            # Avoid empty slices from extreme draws.
            proportions = np.maximum(proportions, 1e-8)
            proportions = proportions / proportions.sum()
            splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
            pieces = np.split(class_idx, splits)
            for client_id, piece in enumerate(pieces):
                client_indices[client_id].extend(piece.tolist())

        sizes = [len(v) for v in client_indices]
        if min(sizes) >= min_size:
            result = []
            for values in client_indices:
                arr = np.array(values, dtype=np.int64)
                rng.shuffle(arr)
                result.append(arr)
            return result

        rng = np.random.default_rng(seed + attempt + 1)

    raise RuntimeError(
        "Failed to create a valid Dirichlet partition. "
        "Try a larger alpha or fewer clients."
    )


def partition_dataset(
    labels: np.ndarray,
    num_clients: int,
    partition: str = "non_iid",
    dirichlet_alpha: float = 0.3,
    seed: int = 42,
) -> list[np.ndarray]:
    """
    Return a list of index arrays, one per client.

    Parameters
    ----------
    labels:
        Array of integer class labels for the training set.
    num_clients:
        Number of federated clients.
    partition:
        ``iid`` or ``non_iid``.
    dirichlet_alpha:
        Dirichlet concentration for non-IID partitioning.
    seed:
        Random seed.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")

    mode = partition.lower()
    if mode == "iid":
        return _iid_indices(labels, num_clients, seed)
    if mode in {"non_iid", "non-iid", "dirichlet"}:
        return _dirichlet_indices(labels, num_clients, dirichlet_alpha, seed)
    raise ValueError(f"Unsupported partition mode: {partition}")


def summarize_partitions(
    labels: np.ndarray, client_indices: list[np.ndarray]
) -> list[dict[str, Any]]:
    """Return per-client sample counts and class histograms."""
    summaries: list[dict[str, Any]] = []
    num_classes = int(labels.max()) + 1
    for client_id, indices in enumerate(client_indices):
        client_labels = labels[indices]
        hist = np.bincount(client_labels, minlength=num_classes).tolist()
        summaries.append(
            {
                "client_id": client_id,
                "num_samples": int(len(indices)),
                "class_counts": hist,
            }
        )
    return summaries
