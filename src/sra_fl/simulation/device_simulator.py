"""FedScale device-capacity loading and compute-delay simulation."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sra_fl.utils.logger import get_logger
from sra_fl.utils.paths import DEVICE_CAPACITY_FILE, DEVICE_PROFILES_CACHE, require_file

logger = get_logger(__name__)


@dataclass(frozen=True)
class DeviceProfile:
    """Compute/communication capacity for one simulated device."""

    profile_id: int
    computation: float  # FedScale compute rate (higher = faster)
    communication: float  # FedScale communication rate (KB/s-like)
    speed_tier: str  # fast | medium | slow

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify_speed(computation: float, q33: float, q66: float) -> str:
    if computation >= q66:
        return "fast"
    if computation >= q33:
        return "medium"
    return "slow"


def load_device_profiles(
    *,
    max_profiles: int = 5000,
    seed: int = 42,
    cache_path: Path | None = DEVICE_PROFILES_CACHE,
) -> list[DeviceProfile]:
    """
    Load FedScale ``client_device_capacity`` profiles.

    Samples up to ``max_profiles`` entries and caches them as JSON for faster reloads.
    """
    cache_path = cache_path or DEVICE_PROFILES_CACHE
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        profiles = [DeviceProfile(**item) for item in raw]
        if profiles:
            logger.info("Loaded %s cached device profiles from %s", len(profiles), cache_path)
            return profiles

    path = require_file(DEVICE_CAPACITY_FILE, "FedScale device capacity file")
    logger.info("Loading FedScale device profiles from %s", path)
    with path.open("rb") as handle:
        data = pickle.load(handle)

    rng = np.random.default_rng(seed)
    keys = list(data.keys())
    if len(keys) > max_profiles:
        keys = rng.choice(keys, size=max_profiles, replace=False).tolist()

    comps = np.array([float(data[k]["computation"]) for k in keys], dtype=np.float64)
    q33, q66 = np.percentile(comps, [33.3, 66.6])

    profiles: list[DeviceProfile] = []
    for key in keys:
        item = data[key]
        computation = float(item["computation"])
        communication = float(item["communication"])
        profiles.append(
            DeviceProfile(
                profile_id=int(key),
                computation=computation,
                communication=communication,
                speed_tier=_classify_speed(computation, float(q33), float(q66)),
            )
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump([p.to_dict() for p in profiles], handle)
    logger.info("Cached %s device profiles to %s", len(profiles), cache_path)
    return profiles


def assign_device_profiles(
    num_clients: int,
    profiles: list[DeviceProfile] | None = None,
    *,
    seed: int = 42,
) -> list[DeviceProfile]:
    """Assign one device profile to each client (with replacement if needed)."""
    profiles = profiles or load_device_profiles(seed=seed)
    if not profiles:
        raise RuntimeError("No device profiles available")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(profiles), size=num_clients, replace=len(profiles) < num_clients)
    return [profiles[int(i)] for i in indices]


def estimate_training_seconds(
    *,
    num_samples: int,
    local_epochs: int,
    computation: float,
    min_seconds: float = 0.05,
) -> float:
    """Estimate local training time from FedScale computation capacity."""
    rate = max(float(computation), 1.0)
    seconds = (max(num_samples, 1) * max(local_epochs, 1)) / rate
    return float(max(seconds, min_seconds))
