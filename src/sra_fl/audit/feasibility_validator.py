"""Feasibility checks on reported client resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sra_fl.client.client_state import ClientState, ResourceReport


@dataclass
class FeasibilityResult:
    client_id: int
    suspicious: bool
    score: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "suspicious": self.suspicious,
            "score": self.score,
            "reasons": self.reasons,
        }


class FeasibilityValidator:
    """
    Flag impossible / inconsistent self-reports.

    Uses reported values plus optional observed train time. When ground-truth
    ``ClientState`` is provided (simulation), also scores report-vs-actual gaps.
    """

    def __init__(
        self,
        *,
        max_compute: float = 500.0,
        max_bandwidth_kbps: float = 50000.0,
        relative_gap_threshold: float = 0.5,
    ):
        self.max_compute = max_compute
        self.max_bandwidth_kbps = max_bandwidth_kbps
        self.relative_gap_threshold = relative_gap_threshold

    def validate(
        self,
        client_id: int,
        report: ResourceReport,
        *,
        state: ClientState | None = None,
        observed_train_seconds: float | None = None,
        local_epochs: int = 1,
    ) -> FeasibilityResult:
        reasons: list[str] = []
        score = 0.0

        if report.computation <= 0 or report.bandwidth_kbps <= 0:
            reasons.append("non_positive_resources")
            score += 0.5
        if report.computation > self.max_compute:
            reasons.append("compute_above_plausible_max")
            score += 0.3
        if report.bandwidth_kbps > self.max_bandwidth_kbps:
            reasons.append("bandwidth_above_plausible_max")
            score += 0.3

        if state is not None:
            actual_c = max(state.actual_computation, 1e-6)
            actual_b = max(state.actual_bandwidth_kbps, 1e-6)
            gap_c = abs(report.computation - state.actual_computation) / actual_c
            gap_b = abs(report.bandwidth_kbps - state.actual_bandwidth_kbps) / actual_b
            if gap_c > self.relative_gap_threshold:
                reasons.append(f"compute_gap={gap_c:.2f}")
                score += min(0.5, gap_c / 4.0)
            if gap_b > self.relative_gap_threshold:
                reasons.append(f"bandwidth_gap={gap_b:.2f}")
                score += min(0.5, gap_b / 4.0)
            if report.available != state.is_available(0) and "fake_availability" in str(
                state.metadata.get("lie_type")
            ):
                # Direct availability mismatch is checked by caller with round timestamp.
                pass

        if observed_train_seconds is not None and report.computation > 0 and state is not None:
            # Expected train time from claimed compute vs observed (actual) time.
            expected = (state.num_samples * max(local_epochs, 1)) / max(report.computation, 1.0)
            if expected > 0:
                ratio = observed_train_seconds / expected
                # Claimed much faster than observed behavior => exaggerate compute.
                if ratio > 1.8:
                    reasons.append(f"train_time_inconsistent_ratio={ratio:.2f}")
                    score += min(0.4, (ratio - 1.0) / 5.0)

        score = float(min(1.0, score))
        return FeasibilityResult(
            client_id=client_id,
            suspicious=score >= 0.25,
            score=score,
            reasons=reasons,
        )
