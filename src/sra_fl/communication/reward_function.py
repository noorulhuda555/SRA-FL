"""Reward signal for communication decisions."""

from __future__ import annotations


def communication_reward(
    *,
    action: str,
    accuracy_delta: float,
    upload_seconds: float,
    bandwidth_kbps: float,
) -> float:
    """
    Higher is better.

    Encourages useful uploads, penalizes expensive/slow communication,
    and mildly rewards skip/wait when bandwidth is poor and gain is tiny.
    """
    cost = upload_seconds / max(bandwidth_kbps / 1000.0, 0.01)
    if action == "upload":
        return float(accuracy_delta * 10.0 - 0.2 * cost)
    if action == "compress":
        return float(accuracy_delta * 8.0 - 0.1 * cost + 0.05)
    if action == "wait":
        return float(-0.02 + (0.05 if bandwidth_kbps < 100 else -0.05))
    # skip
    return float(-0.05 + (0.08 if accuracy_delta < 0.001 else -0.1))
