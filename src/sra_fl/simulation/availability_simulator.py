"""FedScale availability-trace loading and online/offline simulation."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sra_fl.utils.logger import get_logger
from sra_fl.utils.paths import AVAILABILITY_CACHE, AVAILABILITY_TRACE_FILE, require_file

logger = get_logger(__name__)


@dataclass
class AvailabilityTrace:
    """One client's availability behavior from FedScale traces."""

    trace_id: int
    model: str
    duration: int
    finish_time: int
    active: list[int] = field(default_factory=list)
    inactive: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_available_at(self, timestamp: float) -> bool:
        """
        Return whether the client is online at ``timestamp``.

        FedScale stores paired active/inactive event times. A client is available
        in ``[active[i], inactive[i])`` windows, wrapped by ``finish_time``.
        """
        finish = max(int(self.finish_time), 1)
        t = int(timestamp) % finish
        n = min(len(self.active), len(self.inactive))
        for i in range(n):
            start = int(self.active[i])
            end = int(self.inactive[i])
            lo, hi = (start, end) if start <= end else (end, start)
            if lo <= t < hi:
                return True
        return False

    def availability_ratio(self) -> float:
        """Approximate fraction of time the client is online."""
        finish = max(int(self.finish_time), 1)
        n = min(len(self.active), len(self.inactive))
        online = 0
        for i in range(n):
            start = int(self.active[i])
            end = int(self.inactive[i])
            online += abs(end - start)
        return float(min(1.0, online / finish))


def load_availability_traces(
    *,
    max_traces: int = 5000,
    seed: int = 42,
    cache_path: Path | None = AVAILABILITY_CACHE,
) -> list[AvailabilityTrace]:
    """Load FedScale ``client_behave_trace`` entries (cached after first load)."""
    cache_path = cache_path or AVAILABILITY_CACHE
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        traces = [AvailabilityTrace(**item) for item in raw]
        if traces:
            logger.info("Loaded %s cached availability traces from %s", len(traces), cache_path)
            return traces

    path = require_file(AVAILABILITY_TRACE_FILE, "FedScale availability trace file")
    logger.info("Loading FedScale availability traces from %s", path)
    with path.open("rb") as handle:
        data = pickle.load(handle)

    rng = np.random.default_rng(seed + 3)
    keys = list(data.keys())
    if len(keys) > max_traces:
        keys = rng.choice(keys, size=max_traces, replace=False).tolist()

    traces: list[AvailabilityTrace] = []
    for key in keys:
        item = data[key]
        traces.append(
            AvailabilityTrace(
                trace_id=int(key),
                model=str(item.get("model", "unknown")),
                duration=int(item.get("duration", 0)),
                finish_time=int(item.get("finish_time", 1)),
                active=[int(x) for x in item.get("active", [])],
                inactive=[int(x) for x in item.get("inactive", [])],
            )
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump([t.to_dict() for t in traces], handle)
    logger.info("Cached %s availability traces to %s", len(traces), cache_path)
    return traces


def assign_availability_traces(
    num_clients: int,
    traces: list[AvailabilityTrace] | None = None,
    *,
    seed: int = 42,
) -> list[AvailabilityTrace]:
    """Assign one availability trace to each client."""
    traces = traces or load_availability_traces(seed=seed)
    if not traces:
        raise RuntimeError("No availability traces available")
    rng = np.random.default_rng(seed + 29)
    indices = rng.choice(len(traces), size=num_clients, replace=len(traces) < num_clients)
    return [traces[int(i)] for i in indices]


def round_timestamp(server_round: int, seconds_per_round: float) -> float:
    """Map a federated round index to a virtual wall-clock timestamp."""
    return float(max(server_round - 1, 0) * seconds_per_round)
