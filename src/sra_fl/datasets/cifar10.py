"""CIFAR-10 loading from local project data."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from sra_fl.datasets.preprocessing import get_transforms
from sra_fl.utils.logger import get_logger
from sra_fl.utils.paths import CIFAR10_CACHE, CIFAR10_DIR, require_dir, require_file

logger = get_logger(__name__)


def _resolve_data_dir() -> Path:
    data_dir = require_dir(CIFAR10_DIR, "CIFAR-10 raw directory")
    require_file(data_dir / "train.csv", "CIFAR-10 train.csv")
    require_file(data_dir / "test.csv", "CIFAR-10 test.csv")
    return data_dir


def _decode_png_cell(cell: object) -> np.ndarray:
    if isinstance(cell, dict):
        payload = cell["bytes"]
    elif isinstance(cell, (bytes, bytearray)):
        payload = bytes(cell)
    elif isinstance(cell, str):
        parsed = ast.literal_eval(cell)
        payload = parsed["bytes"] if isinstance(parsed, dict) else parsed
    else:
        raise TypeError(f"Unsupported CIFAR image cell type: {type(cell)}")

    from io import BytesIO

    image = Image.open(BytesIO(payload)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _build_cache(data_dir: Path, cache_path: Path) -> None:
    logger.info("Building CIFAR-10 cache at %s (one-time)", cache_path)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    train_images = np.stack([_decode_png_cell(v) for v in train_df["img"]], axis=0)
    test_images = np.stack([_decode_png_cell(v) for v in test_df["img"]], axis=0)
    train_labels = train_df["label"].to_numpy(dtype=np.int64)
    test_labels = test_df["label"].to_numpy(dtype=np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )
    logger.info("CIFAR-10 cache ready")


def _load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not CIFAR10_CACHE.exists():
        _build_cache(_resolve_data_dir(), CIFAR10_CACHE)
    data = np.load(CIFAR10_CACHE)
    return (
        data["train_images"],
        data["train_labels"],
        data["test_images"],
        data["test_labels"],
    )


class CIFAR10ArrayDataset(Dataset):
    """Torch dataset wrapping uint8 RGB image arrays."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool = True):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = get_transforms("cifar10", train=train)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index], mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


def load_cifar10() -> tuple[CIFAR10ArrayDataset, CIFAR10ArrayDataset, np.ndarray]:
    """
    Load CIFAR-10 train/test datasets from local files (cached after first load).

    Returns
    -------
    train_dataset, test_dataset, train_labels
    """
    train_images, train_labels, test_images, test_labels = _load_arrays()
    train_ds = CIFAR10ArrayDataset(train_images, train_labels, train=True)
    test_ds = CIFAR10ArrayDataset(test_images, test_labels, train=False)
    return train_ds, test_ds, train_labels.astype(np.int64)
