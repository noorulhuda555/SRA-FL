"""Canonical project data paths matching the local ``data/raw`` layout."""

from __future__ import annotations

from pathlib import Path

from sra_fl.utils.config_loader import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

# Image datasets (as present on disk)
FASHION_MNIST_DIR = RAW_DIR / "Fashionmnist"
CIFAR10_DIR = RAW_DIR / "cifar-10"

# System / simulation datasets (as present on disk)
DEVICE_PROFILES_DIR = RAW_DIR / "device_profiles"
AVAILABILITY_TRACES_DIR = RAW_DIR / "availability_traces"
NETWORK_TRACES_DIR = RAW_DIR / "network_traces"

DEVICE_CAPACITY_FILE = DEVICE_PROFILES_DIR / "client_device_capacity"
AVAILABILITY_TRACE_FILE = AVAILABILITY_TRACES_DIR / "client_behave_trace"
# MobiPerf Measurement1–5 are JSON files (not directories).
NETWORK_MEASUREMENT_FILES = [
    NETWORK_TRACES_DIR / f"Measurement{i}" for i in range(1, 6)
]

# Processed caches
CIFAR10_CACHE = PROCESSED_DIR / "cifar10_arrays.npz"
CLIENT_PARTITIONS_DIR = PROCESSED_DIR / "client_partitions"
DEVICE_PROFILES_CACHE = PROCESSED_DIR / "device_profiles" / "sampled_device_profiles.json"
NETWORK_PROFILES_CACHE = PROCESSED_DIR / "network_profiles" / "tcp_throughput_kbps.json"
AVAILABILITY_CACHE = PROCESSED_DIR / "device_profiles" / "sampled_availability_traces.json"


def require_dir(path: Path, label: str) -> Path:
    """Return ``path`` if it exists, otherwise raise a clear error."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at: {path}")
    return path


def require_file(path: Path, label: str) -> Path:
    """Return ``path`` if it is an existing file, otherwise raise a clear error."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found at: {path}")
    return path
