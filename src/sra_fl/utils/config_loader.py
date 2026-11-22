"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def load_config(
    dataset: str | None = None,
    overrides: dict[str, Any] | None = None,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Load base config and merge dataset-specific YAML.

    Parameters
    ----------
    dataset:
        Optional dataset name used to select `fashion_mnist.yaml` or `cifar10.yaml`.
        If omitted, uses `dataset` from base.yaml.
    overrides:
        Optional dict merged on top of loaded configs.
    """
    root = config_dir or CONFIGS_DIR
    config = load_yaml(root / "base.yaml")

    dataset_name = dataset or config.get("dataset", "fashion_mnist")
    dataset_file = root / f"{dataset_name}.yaml"
    if dataset_file.exists():
        config = _deep_update(config, load_yaml(dataset_file))
    else:
        raise FileNotFoundError(f"Dataset config not found: {dataset_file}")

    for extra_name in (
        "attack_config.yaml",
        "negotiation_config.yaml",
        "rag_config.yaml",
        "communication_config.yaml",
        "privacy_config.yaml",
    ):
        extra_path = root / extra_name
        if extra_path.exists():
            config = _deep_update(config, load_yaml(extra_path))

    config["dataset"] = dataset_name
    if overrides:
        config = _deep_update(config, overrides)
    return config
