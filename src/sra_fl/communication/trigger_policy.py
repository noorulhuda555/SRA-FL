"""Learning-based upload trigger policy (epsilon-greedy Q-learning)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Action = Literal["upload", "wait", "compress", "skip"]
ACTIONS: tuple[Action, ...] = ("upload", "wait", "compress", "skip")


def discretize_state(
    *,
    bandwidth_kbps: float,
    train_loss: float,
    recent_delay: float,
) -> str:
    bw = "low" if bandwidth_kbps < 100 else "mid" if bandwidth_kbps < 1000 else "high"
    loss = "high" if train_loss > 1.0 else "mid" if train_loss > 0.4 else "low"
    delay = "slow" if recent_delay > 1.0 else "fast"
    return f"{bw}|{loss}|{delay}"


@dataclass
class TriggerPolicy:
    """Tabular Q-learning (or fixed/threshold) policy over communication actions."""

    epsilon: float = 0.15
    alpha: float = 0.3
    gamma: float = 0.5
    seed: int = 42
    mode: Literal["learning", "fixed", "threshold"] = "learning"
    bandwidth_threshold_kbps: float = 100.0
    q_table: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        mode = str(self.mode).lower()
        if mode not in {"learning", "fixed", "threshold"}:
            raise ValueError(f"Unsupported comm policy mode: {self.mode}")
        self.mode = mode  # type: ignore[assignment]

    def _ensure(self, state: str) -> None:
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ACTIONS}

    def select_action(self, state: str) -> Action:
        if self.mode == "fixed":
            return "upload"
        if self.mode == "threshold":
            bw = state.split("|", 1)[0] if state else "mid"
            if bw == "low":
                return "compress"
            if bw == "high":
                return "upload"
            return "wait"
        self._ensure(state)
        if self.rng.random() < self.epsilon:
            return ACTIONS[int(self.rng.integers(0, len(ACTIONS)))]
        q = self.q_table[state]
        best = max(ACTIONS, key=lambda a: q[a])
        return best

    def update(self, state: str, action: Action, reward: float, next_state: str) -> None:
        if self.mode != "learning":
            return
        self._ensure(state)
        self._ensure(next_state)
        best_next = max(self.q_table[next_state].values())
        old = self.q_table[state][action]
        self.q_table[state][action] = old + self.alpha * (
            reward + self.gamma * best_next - old
        )
