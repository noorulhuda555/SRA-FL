"""Inject dishonest resource reports into client states (Module 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sra_fl.client.client_state import ClientState, ResourceReport
from sra_fl.utils.logger import get_logger

logger = get_logger(__name__)

LieType = Literal[
    "exaggerate_compute",
    "underreport_compute",
    "exaggerate_bandwidth",
    "underreport_bandwidth",
    "fake_availability",
]


@dataclass
class DishonestAssignment:
    client_id: int
    lie_type: LieType


class LyingInjector:
    """Mark a fraction of clients as dishonest and falsify their reports."""

    def __init__(
        self,
        *,
        dishonest_fraction: float = 0.3,
        lie_compute_factor: float = 3.0,
        lie_bandwidth_factor: float = 4.0,
        seed: int = 42,
        forced_lie_type: LieType | None = None,
    ):
        self.dishonest_fraction = float(np.clip(dishonest_fraction, 0.0, 1.0))
        self.lie_compute_factor = float(lie_compute_factor)
        self.lie_bandwidth_factor = float(lie_bandwidth_factor)
        self.rng = np.random.default_rng(seed)
        self.forced_lie_type = forced_lie_type
        self.assignments: dict[int, LieType] = {}

    def assign(self, client_ids: list[int]) -> list[DishonestAssignment]:
        n = len(client_ids)
        k = int(round(self.dishonest_fraction * n))
        if k <= 0:
            self.assignments = {}
            return []
        chosen = self.rng.choice(client_ids, size=min(k, n), replace=False)
        lie_types: list[LieType] = [
            "exaggerate_compute",
            "underreport_compute",
            "exaggerate_bandwidth",
            "underreport_bandwidth",
            "fake_availability",
        ]
        out: list[DishonestAssignment] = []
        for cid in chosen:
            lie = self.forced_lie_type or lie_types[int(self.rng.integers(0, len(lie_types)))]
            self.assignments[int(cid)] = lie
            out.append(DishonestAssignment(client_id=int(cid), lie_type=lie))
        logger.info(
            "LyingInjector marked %s/%s clients dishonest: %s",
            len(out),
            n,
            {a.client_id: a.lie_type for a in out},
        )
        return out

    def is_dishonest(self, client_id: int) -> bool:
        return int(client_id) in self.assignments

    def apply_to_state(self, state: ClientState, *, actually_available: bool) -> ResourceReport:
        """Write a (possibly false) report onto ``state.reported``."""
        honest = state.build_honest_report(available=actually_available)
        lie = self.assignments.get(state.client_id)
        if lie is None:
            state.reported = honest
            state.metadata["is_dishonest"] = False
            state.metadata["lie_type"] = None
            return honest

        reported = ResourceReport(
            computation=honest.computation,
            communication=honest.communication,
            bandwidth_kbps=honest.bandwidth_kbps,
            available=honest.available,
            speed_tier=honest.speed_tier,
            network_tier=honest.network_tier,
        )
        if lie == "exaggerate_compute":
            reported.computation = honest.computation * self.lie_compute_factor
            reported.speed_tier = "fast"
        elif lie == "underreport_compute":
            reported.computation = max(1.0, honest.computation / self.lie_compute_factor)
            reported.speed_tier = "slow"
        elif lie == "exaggerate_bandwidth":
            reported.bandwidth_kbps = honest.bandwidth_kbps * self.lie_bandwidth_factor
            reported.communication = honest.communication * self.lie_bandwidth_factor
            reported.network_tier = "fast"
        elif lie == "underreport_bandwidth":
            reported.bandwidth_kbps = max(1.0, honest.bandwidth_kbps / self.lie_bandwidth_factor)
            reported.communication = max(1.0, honest.communication / self.lie_bandwidth_factor)
            reported.network_tier = "slow"
        elif lie == "fake_availability":
            reported.available = not actually_available

        state.reported = reported
        state.metadata["is_dishonest"] = True
        state.metadata["lie_type"] = lie
        return reported
