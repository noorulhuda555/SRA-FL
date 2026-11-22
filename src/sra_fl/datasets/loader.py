"""Dataset factory used by the FL simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import Dataset, Subset

from sra_fl.datasets.cifar10 import load_cifar10
from sra_fl.datasets.fashion_mnist import load_fashion_mnist
from sra_fl.datasets.partitioner import partition_dataset, summarize_partitions


def load_federated_dataset(
    dataset_name: str,
    num_clients: int,
    partition: str = "non_iid",
    dirichlet_alpha: float = 0.3,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Load a dataset and create per-client training subsets.

    Returns a dict with keys:
    train_dataset, test_dataset, client_datasets, partition_summary, train_labels
    """
    name = dataset_name.lower()
    if name in {"fashion_mnist", "fashion-mnist", "fashionmnist"}:
        train_dataset, test_dataset, train_labels = load_fashion_mnist()
        canonical = "fashion_mnist"
    elif name in {"cifar10", "cifar-10"}:
        train_dataset, test_dataset, train_labels = load_cifar10()
        canonical = "cifar10"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    client_indices = partition_dataset(
        labels=train_labels,
        num_clients=num_clients,
        partition=partition,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
    )
    client_datasets: list[Dataset] = [
        Subset(train_dataset, indices.tolist()) for indices in client_indices
    ]
    summary = summarize_partitions(train_labels, client_indices)

    return {
        "dataset_name": canonical,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "client_datasets": client_datasets,
        "client_indices": client_indices,
        "partition_summary": summary,
        "train_labels": train_labels,
    }


def get_partition_sizes(client_datasets: list[Dataset]) -> list[int]:
    """Return number of samples per client partition."""
    return [len(ds) for ds in client_datasets]
