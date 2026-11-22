"""Client reputation scores for selection (Module 3)."""

from __future__ import annotations

from typing import Any


class ReputationManager:
    """Increase/decrease reputation from audit outcomes."""

    def __init__(
        self,
        client_ids: list[int],
        *,
        initial: float = 1.0,
        penalty: float = 0.25,
        reward: float = 0.05,
        minimum: float = 0.05,
        maximum: float = 1.0,
    ):
        self.penalty = penalty
        self.reward = reward
        self.minimum = minimum
        self.maximum = maximum
        self.scores: dict[int, float] = {int(cid): float(initial) for cid in client_ids}

    def get(self, client_id: int) -> float:
        return float(self.scores.get(int(client_id), self.minimum))

    def punish(self, client_id: int, severity: float = 1.0) -> float:
        cid = int(client_id)
        self.scores[cid] = max(
            self.minimum, self.get(cid) - self.penalty * float(severity)
        )
        return self.get(cid)

    def reward_client(self, client_id: int) -> float:
        cid = int(client_id)
        self.scores[cid] = min(self.maximum, self.get(cid) + self.reward)
        return self.get(cid)

    def as_dict(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in sorted(self.scores.items())}

    def ranked_ids(self, candidate_ids: list[int]) -> list[int]:
        return sorted(candidate_ids, key=lambda cid: self.get(cid), reverse=True)

    def summary(self) -> dict[str, Any]:
        values = list(self.scores.values())
        return {
            "mean": float(sum(values) / max(len(values), 1)),
            "min": float(min(values) if values else 0.0),
            "max": float(max(values) if values else 0.0),
            "scores": self.as_dict(),
        }
