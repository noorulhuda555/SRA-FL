"""Tests for Module 2 heterogeneous client simulation."""

from __future__ import annotations

import numpy as np

from sra_fl.simulation.availability_simulator import (
    AvailabilityTrace,
    load_availability_traces,
)
from sra_fl.simulation.client_environment import ClientEnvironment
from sra_fl.simulation.device_simulator import (
    estimate_training_seconds,
    load_device_profiles,
)
from sra_fl.simulation.network_simulator import (
    estimate_transfer_seconds,
    load_network_profiles,
)


def test_load_device_profiles():
    profiles = load_device_profiles(max_profiles=200, seed=0)
    assert len(profiles) > 0
    assert all(p.computation > 0 for p in profiles)
    assert {p.speed_tier for p in profiles} <= {"fast", "medium", "slow"}


def test_load_network_profiles():
    profiles = load_network_profiles()
    assert len(profiles) > 0
    assert all(p.bandwidth_kbps > 0 for p in profiles)


def test_load_availability_traces():
    traces = load_availability_traces(max_traces=200, seed=0)
    assert len(traces) > 0
    sample = traces[0]
    assert sample.finish_time > 0


def test_availability_window_logic():
    trace = AvailabilityTrace(
        trace_id=1,
        model="x",
        duration=100,
        finish_time=100,
        active=[10, 60],
        inactive=[30, 80],
    )
    assert trace.is_available_at(15) is True
    assert trace.is_available_at(40) is False
    assert trace.is_available_at(70) is True


def test_delay_estimators():
    train_s = estimate_training_seconds(num_samples=1000, local_epochs=1, computation=100)
    assert train_s == 10.0
    upload_s = estimate_transfer_seconds(num_bytes=1024 * 100, bandwidth_kbps=100)
    assert abs(upload_s - 1.0) < 1e-6


def test_client_environment_selection():
    env = ClientEnvironment.create(
        num_clients=8,
        partition_sizes=[100] * 8,
        seed=42,
        seconds_per_round=1000.0,
        dropout_probability=0.0,
    )
    part = env.select_participants(1, fraction=0.5, min_clients=2)
    assert len(part.selected_ids) >= 2
    assert set(part.selected_ids).issubset(set(part.available_ids))
    costs = env.simulate_round_costs(
        part.selected_ids[0],
        local_epochs=1,
        parameters=[np.zeros((4, 4), dtype=np.float32)],
    )
    assert costs["train_seconds"] > 0
    assert costs["upload_seconds"] > 0
