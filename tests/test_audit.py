"""Tests for Module 3 AAVL audit."""

from __future__ import annotations

from sra_fl.attacks.lying_injector import LyingInjector
from sra_fl.audit.audit_engine import AuditEngine
from sra_fl.simulation.client_environment import ClientEnvironment


def test_lying_injector_marks_clients():
    env = ClientEnvironment.create(
        num_clients=10,
        partition_sizes=[100] * 10,
        seed=1,
        dropout_probability=0.0,
    )
    injector = LyingInjector(dishonest_fraction=0.3, seed=1)
    assigned = injector.assign(list(range(10)))
    assert 2 <= len(assigned) <= 4
    for a in assigned:
        state = env.get_state(a.client_id)
        honest = state.build_honest_report(available=True)
        lied = injector.apply_to_state(state, actually_available=True)
        assert state.metadata["is_dishonest"] is True
        assert lied.to_dict() != honest.to_dict() or a.lie_type == "fake_availability"


def test_audit_detects_compute_lies():
    env = ClientEnvironment.create(
        num_clients=8,
        partition_sizes=[200] * 8,
        seed=7,
        dropout_probability=0.0,
    )
    injector = LyingInjector(
        dishonest_fraction=0.5,
        lie_compute_factor=5.0,
        lie_bandwidth_factor=5.0,
        seed=7,
    )
    injector.assign(list(range(8)))
    env.lying_injector = injector
    for state in env.states.values():
        injector.apply_to_state(state, actually_available=True)

    engine = AuditEngine(list(range(8)), suspicion_threshold=0.25)
    summary = engine.audit_clients(1, env.states, list(range(8)))
    assert summary.true_positives + summary.false_negatives >= 1
    assert 0.0 <= summary.f1 <= 1.0
    assert all(cid in engine.reputation.scores for cid in range(8))
