"""MobiPerf network-trace loading and upload/download delay simulation."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sra_fl.utils.logger import get_logger
from sra_fl.utils.paths import NETWORK_MEASUREMENT_FILES, NETWORK_PROFILES_CACHE

logger = get_logger(__name__)


@dataclass(frozen=True)
class NetworkProfile:
    """Bandwidth profile for one simulated client link."""

    bandwidth_kbps: float
    speed_tier: str  # fast | medium | slow
    source_measurement: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify_bandwidth(kbps: float, q33: float, q66: float) -> str:
    if kbps >= q66:
        return "fast"
    if kbps >= q33:
        return "medium"
    return "slow"


def _throughput_from_tcp_row(row: dict[str, Any]) -> float | None:
    """Derive KB/s from a successful MobiPerf tcpthroughput record."""
    if row.get("type") != "tcpthroughput" or not row.get("success"):
        return None
    values = row.get("values") or {}
    try:
        duration = float(values.get("duration") or 0.0)
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None

    ctx_raw = values.get("context_results") or "[]"
    try:
        context = ast.literal_eval(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
    except (SyntaxError, ValueError):
        return None
    if not isinstance(context, list) or not context:
        return None

    bytes_up = sum(int(item.get("inc_total_bytes_send", 0)) for item in context)
    bytes_down = sum(int(item.get("inc_total_bytes_recv", 0)) for item in context)
    dir_up = bool((row.get("parameters") or {}).get("dir_up"))
    nbytes = bytes_up if dir_up else bytes_down
    if nbytes <= 0:
        nbytes = max(bytes_up, bytes_down)
    if nbytes <= 0:
        return None
    return float((nbytes / duration) / 1024.0)


def load_network_profiles(
    *,
    cache_path: Path | None = NETWORK_PROFILES_CACHE,
) -> list[NetworkProfile]:
    """Load TCP throughput samples from MobiPerf Measurement1–5 JSON files."""
    cache_path = cache_path or NETWORK_PROFILES_CACHE
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        profiles = [NetworkProfile(**item) for item in raw]
        if profiles:
            logger.info("Loaded %s cached network profiles from %s", len(profiles), cache_path)
            return profiles

    speeds: list[tuple[float, int]] = []
    for measurement_id, path in enumerate(NETWORK_MEASUREMENT_FILES, start=1):
        if not path.is_file():
            logger.warning("Missing network measurement file: %s", path)
            continue
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        for row in data:
            kbps = _throughput_from_tcp_row(row)
            if kbps is not None and kbps > 0:
                speeds.append((kbps, measurement_id))

    if not speeds:
        raise RuntimeError(
            "No TCP throughput samples found in MobiPerf Measurement1–5 files"
        )

    values = np.array([s[0] for s in speeds], dtype=np.float64)
    q33, q66 = np.percentile(values, [33.3, 66.6])
    profiles = [
        NetworkProfile(
            bandwidth_kbps=float(kbps),
            speed_tier=_classify_bandwidth(float(kbps), float(q33), float(q66)),
            source_measurement=int(src),
        )
        for kbps, src in speeds
    ]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump([p.to_dict() for p in profiles], handle)
    logger.info("Cached %s network profiles to %s", len(profiles), cache_path)
    return profiles


def assign_network_profiles(
    num_clients: int,
    profiles: list[NetworkProfile] | None = None,
    *,
    seed: int = 42,
) -> list[NetworkProfile]:
    """Assign one network profile to each client."""
    profiles = profiles or load_network_profiles()
    if not profiles:
        raise RuntimeError("No network profiles available")
    rng = np.random.default_rng(seed + 17)
    indices = rng.choice(len(profiles), size=num_clients, replace=True)
    return [profiles[int(i)] for i in indices]


def estimate_transfer_seconds(
    *,
    num_bytes: int,
    bandwidth_kbps: float,
    min_seconds: float = 0.01,
) -> float:
    """Estimate upload/download time for ``num_bytes`` at the given KB/s rate."""
    rate = max(float(bandwidth_kbps), 1.0)
    seconds = (max(num_bytes, 1) / 1024.0) / rate
    return float(max(seconds, min_seconds))


def estimate_model_nbytes(parameters: list[np.ndarray]) -> int:
    """Approximate serialized model size from NumPy parameter arrays."""
    return int(sum(np.asarray(layer).nbytes for layer in parameters))
