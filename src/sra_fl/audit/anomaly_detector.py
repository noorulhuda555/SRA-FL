"""Historical anomaly detection over client reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sra_fl.audit.ema_tracker import EMATracker
from sra_fl.audit.feasibility_validator import FeasibilityResult


@dataclass
class AnomalyRecord:
    client_id: int
    round_id: int
    suspicion: float
    ema_suspicion: float
    flagged: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "round_id": self.round_id,
            "suspicion": self.suspicion,
            "ema_suspicion": self.ema_suspicion,
            "flagged": self.flagged,
            "reasons": self.reasons,
        }


class AnomalyDetector:
    """Accumulate repeated misreporting signals with EMA smoothing."""

    def __init__(self, *, threshold: float = 0.35, ema_alpha: float = 0.4):
        self.threshold = float(threshold)
        self.ema = EMATracker(alpha=ema_alpha)
        self.history: list[AnomalyRecord] = []

    def observe(self, round_id: int, result: FeasibilityResult) -> AnomalyRecord:
        ema_value = self.ema.update(result.client_id, result.score)
        flagged = ema_value >= self.threshold or result.suspicious
        record = AnomalyRecord(
            client_id=result.client_id,
            round_id=round_id,
            suspicion=result.score,
            ema_suspicion=ema_value,
            flagged=flagged,
            reasons=list(result.reasons),
        )
        self.history.append(record)
        return record
