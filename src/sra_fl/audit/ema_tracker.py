"""Exponential moving average tracker for suspicion scores."""

from __future__ import annotations


class EMATracker:
    """Track a smoothed per-client suspicion score."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = float(alpha)
        self.values: dict[int, float] = {}

    def update(self, client_id: int, observation: float) -> float:
        prev = self.values.get(client_id, float(observation))
        value = self.alpha * float(observation) + (1.0 - self.alpha) * prev
        self.values[client_id] = value
        return value

    def get(self, client_id: int, default: float = 0.0) -> float:
        return float(self.values.get(client_id, default))
