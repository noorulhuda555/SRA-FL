"""Tests for Module 6 learning-based communication."""

from __future__ import annotations

import numpy as np

from sra_fl.communication.trigger_policy import TriggerPolicy, discretize_state
from sra_fl.communication.upload_manager import UploadManager


def test_trigger_policy_learns():
    policy = TriggerPolicy(epsilon=0.0, seed=0)
    state = discretize_state(bandwidth_kbps=50.0, train_loss=0.8, recent_delay=2.0)
    action = policy.select_action(state)
    assert action in {"upload", "wait", "compress", "skip"}
    policy.update(state, action, reward=1.0, next_state=state)
    assert policy.q_table[state][action] > 0


def test_upload_manager_actions():
    manager = UploadManager(policy=TriggerPolicy(epsilon=0.0, seed=1))
    params = [np.ones((4, 4), dtype=np.float32)]
    action = manager.decide(0, train_loss=0.5, bandwidth_kbps=80.0)
    out, metrics = manager.apply(
        0,
        action,
        params,
        compression_level=1,
        upload_seconds=0.5,
        accuracy_delta=0.01,
    )
    if action in {"wait", "skip"}:
        assert out is None
        assert metrics["uploaded"] is False
    else:
        assert out is not None
        assert metrics["uploaded"] is True
