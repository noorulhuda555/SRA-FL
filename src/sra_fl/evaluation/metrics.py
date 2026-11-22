"""Module 1 monitoring: global accuracy/loss and participation tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def weighted_average(metrics: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    """Aggregate client metrics weighted by number of examples."""
    if not metrics:
        return {}

    totals: dict[str, float] = {}
    total_examples = 0
    for num_examples, metric in metrics:
        total_examples += num_examples
        for key, value in metric.items():
            if key == "client_id":
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * num_examples

    if total_examples == 0:
        return {key: 0.0 for key in totals}

    return {key: value / total_examples for key, value in totals.items()}


@dataclass
class MetricsTracker:
    """Collect per-round federated learning metrics and save them to disk."""

    results_dir: Path
    history: list[dict[str, Any]] = field(default_factory=list)
    participation: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir)
        (self.results_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "logs").mkdir(parents=True, exist_ok=True)

    def log_round(
        self,
        server_round: int,
        *,
        loss: float | None,
        accuracy: float | None,
        fit_participants: int | None = None,
        evaluate_participants: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "round": server_round,
            "loss": loss,
            "accuracy": accuracy,
            "fit_participants": fit_participants,
            "evaluate_participants": evaluate_participants,
        }
        if extra:
            record.update(extra)
        self.history.append(record)

    def log_participation(self, server_round: int, client_ids: list[int], phase: str) -> None:
        self.participation.append(
            {
                "round": server_round,
                "phase": phase,
                "client_ids": client_ids,
                "num_clients": len(client_ids),
            }
        )

    def save(self, run_name: str = "module1_run") -> Path:
        out_path = self.results_dir / "metrics" / f"{run_name}.json"
        payload = {
            "history": self.history,
            "participation": self.participation,
        }
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return out_path
