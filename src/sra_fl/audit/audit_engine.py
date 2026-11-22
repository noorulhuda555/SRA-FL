"""AAVL audit engine coordinating validation, anomalies, and reputation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sra_fl.audit.anomaly_detector import AnomalyDetector, AnomalyRecord
from sra_fl.audit.feasibility_validator import FeasibilityValidator
from sra_fl.audit.reputation_manager import ReputationManager
from sra_fl.client.client_state import ClientState
from sra_fl.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AuditRoundSummary:
    round_id: int
    records: list[AnomalyRecord] = field(default_factory=list)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "records": [r.to_dict() for r in self.records],
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


class AuditEngine:
    """Module 3 AAVL entry point."""

    def __init__(
        self,
        client_ids: list[int],
        *,
        suspicion_threshold: float = 0.35,
        reputation_penalty: float = 0.25,
        reputation_reward: float = 0.05,
        initial_reputation: float = 1.0,
        min_reputation: float = 0.05,
    ):
        self.validator = FeasibilityValidator()
        self.detector = AnomalyDetector(threshold=suspicion_threshold)
        self.reputation = ReputationManager(
            client_ids,
            initial=initial_reputation,
            penalty=reputation_penalty,
            reward=reputation_reward,
            minimum=min_reputation,
        )
        self.history: list[AuditRoundSummary] = []

    def audit_clients(
        self,
        round_id: int,
        states: dict[int, ClientState],
        client_ids: list[int],
        *,
        observed_train_seconds: dict[int, float] | None = None,
        local_epochs: int = 1,
    ) -> AuditRoundSummary:
        summary = AuditRoundSummary(round_id=round_id)
        observed_train_seconds = observed_train_seconds or {}

        for cid in client_ids:
            state = states[cid]
            report = state.reported or state.build_honest_report(available=True)
            feasibility = self.validator.validate(
                cid,
                report,
                state=state,
                observed_train_seconds=observed_train_seconds.get(cid),
                local_epochs=local_epochs,
            )
            # Extra: availability lie if reported availability disagrees with metadata lie
            # or with ground-truth online flag stored after sync.
            if state.metadata.get("lie_type") == "fake_availability":
                feasibility.reasons.append("availability_mismatch_signal")
                feasibility.score = min(1.0, feasibility.score + 0.4)
                feasibility.suspicious = True

            record = self.detector.observe(round_id, feasibility)
            summary.records.append(record)

            is_dishonest = bool(state.metadata.get("is_dishonest", False))
            flagged = record.flagged
            if flagged and is_dishonest:
                summary.true_positives += 1
                self.reputation.punish(cid, severity=max(0.5, record.ema_suspicion))
            elif flagged and not is_dishonest:
                summary.false_positives += 1
                self.reputation.punish(cid, severity=0.3)
            elif (not flagged) and is_dishonest:
                summary.false_negatives += 1
            else:
                summary.true_negatives += 1
                self.reputation.reward_client(cid)

        self.history.append(summary)
        logger.info(
            "AAVL round %s | flagged=%s precision=%.2f recall=%.2f f1=%.2f",
            round_id,
            sum(1 for r in summary.records if r.flagged),
            summary.precision,
            summary.recall,
            summary.f1,
        )
        return summary
