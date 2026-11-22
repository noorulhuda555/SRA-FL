"""Fashion-MNIST loading from local project data."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from sra_fl.datasets.preprocessing import get_transforms
from sra_fl.utils.paths import FASHION_MNIST_DIR, require_dir


def _has_fashion_mnist_files(path: Path) -> bool:
    return (path / "train-images-idx3-ubyte").exists() or (
        path / "fashion-mnist_train.csv"
    ).exists()


def _resolve_data_dir() -> Path:
    data_dir = require_dir(FASHION_MNIST_DIR, "Fashion-MNIST raw directory")
    if not _has_fashion_mnist_files(data_dir):
        raise FileNotFoundError(
            "Fashion-MNIST files missing. Expected IDX "
            "(train-images-idx3-ubyte) or CSV (fashion-mnist_train.csv) under: "
            f"{data_dir}"
        )
    return data_dir


def _read_idx_images(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic, num, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image IDX magic number in {path}: {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic, num = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label IDX magic number in {path}: {magic}")
        return np.frombuffer(handle.read(), dtype=np.uint8).copy()


def _load_from_idx(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images = _read_idx_images(data_dir / "train-images-idx3-ubyte")
    train_labels = _read_idx_labels(data_dir / "train-labels-idx1-ubyte")
    test_images = _read_idx_images(data_dir / "t10k-images-idx3-ubyte")
    test_labels = _read_idx_labels(data_dir / "t10k-labels-idx1-ubyte")
    return train_images, train_labels, test_images, test_labels


def _load_from_csv(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = pd.read_csv(data_dir / "fashion-mnist_train.csv")
    test_df = pd.read_csv(data_dir / "fashion-mnist_test.csv")

    train_labels = train_df["label"].to_numpy(dtype=np.uint8)
    test_labels = test_df["label"].to_numpy(dtype=np.uint8)
    train_images = train_df.drop(columns=["label"]).to_numpy(dtype=np.uint8).reshape(-1, 28, 28)
    test_images = test_df.drop(columns=["label"]).to_numpy(dtype=np.uint8).reshape(-1, 28, 28)
    return train_images, train_labels, test_images, test_labels


class FashionMNISTArrayDataset(Dataset):
    """Torch dataset wrapping uint8 image arrays."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool = True):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = get_transforms("fashion_mnist", train=train)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        from PIL import Image

        image = Image.fromarray(self.images[index], mode="L")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


def load_fashion_mnist() -> tuple[FashionMNISTArrayDataset, FashionMNISTArrayDataset, np.ndarray]:
    """
    Load Fashion-MNIST train/test datasets from local files.

    Returns
    -------
    train_dataset, test_dataset, train_labels
    """
    data_dir = _resolve_data_dir()
    idx_train = data_dir / "train-images-idx3-ubyte"
    if idx_train.exists():
        train_images, train_labels, test_images, test_labels = _load_from_idx(data_dir)
    else:
        train_images, train_labels, test_images, test_labels = _load_from_csv(data_dir)

    train_ds = FashionMNISTArrayDataset(train_images, train_labels, train=True)
    test_ds = FashionMNISTArrayDataset(test_images, test_labels, train=False)
    return train_ds, test_ds, train_labels.astype(np.int64)
