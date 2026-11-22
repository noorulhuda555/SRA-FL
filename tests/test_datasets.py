"""Tests for Module 1 dataset loading and partitioning."""

from __future__ import annotations

import numpy as np

from sra_fl.datasets.fashion_mnist import load_fashion_mnist
from sra_fl.datasets.loader import load_federated_dataset
from sra_fl.datasets.partitioner import partition_dataset


def test_load_fashion_mnist_shapes():
    train_ds, test_ds, labels = load_fashion_mnist()
    assert len(train_ds) == 60000
    assert len(test_ds) == 10000
    assert labels.shape == (60000,)
    image, label = train_ds[0]
    assert image.ndim == 3
    assert image.shape[0] == 1
    assert 0 <= int(label) <= 9


def test_iid_and_non_iid_partition_sizes():
    labels = np.array([i % 10 for i in range(1000)])
    iid = partition_dataset(labels, num_clients=10, partition="iid", seed=0)
    non_iid = partition_dataset(
        labels, num_clients=10, partition="non_iid", dirichlet_alpha=0.3, seed=0
    )
    assert len(iid) == 10
    assert len(non_iid) == 10
    assert sum(len(x) for x in iid) == 1000
    assert sum(len(x) for x in non_iid) == 1000


def test_federated_loader_fashion_mnist():
    fed = load_federated_dataset(
        dataset_name="fashion_mnist",
        num_clients=5,
        partition="non_iid",
        dirichlet_alpha=0.5,
        seed=42,
    )
    assert len(fed["client_datasets"]) == 5
    assert len(fed["test_dataset"]) == 10000
    assert sum(len(ds) for ds in fed["client_datasets"]) == 60000
