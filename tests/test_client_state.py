"""Tests for Module 1–2 client state and local training."""

from __future__ import annotations

from sra_fl.client.client_state import ClientState
from sra_fl.client.local_trainer import evaluate_model, train_model
from sra_fl.datasets.loader import load_federated_dataset
from sra_fl.models.cnn import create_model
from sra_fl.models.model_utils import get_parameters, set_parameters
from sra_fl.simulation.availability_simulator import AvailabilityTrace
from sra_fl.simulation.device_simulator import DeviceProfile
from sra_fl.simulation.network_simulator import NetworkProfile


def test_client_state_actual_vs_reported():
    state = ClientState(
        client_id=3,
        num_samples=120,
        device=DeviceProfile(
            profile_id=1, computation=50.0, communication=1000.0, speed_tier="medium"
        ),
        network=NetworkProfile(
            bandwidth_kbps=200.0, speed_tier="medium", source_measurement=1
        ),
        availability=AvailabilityTrace(
            trace_id=1,
            model="test",
            duration=100,
            finish_time=100,
            active=[0],
            inactive=[50],
        ),
    )
    payload = state.to_dict()
    assert payload["client_id"] == 3
    assert payload["actual"]["computation"] == 50.0
    assert payload["reported"]["bandwidth_kbps"] == 200.0
    assert state.is_available(10.0) is True
    assert state.is_available(60.0) is False


def test_local_train_and_eval_smoke():
    fed = load_federated_dataset(
        dataset_name="fashion_mnist",
        num_clients=10,
        partition="iid",
        seed=0,
    )
    model = create_model("fashion_mnist")
    params_before = get_parameters(model)
    metrics = train_model(
        model,
        fed["client_datasets"][0],
        epochs=1,
        batch_size=32,
        learning_rate=0.05,
    )
    assert metrics["num_samples"] > 0
    loss, acc, n = evaluate_model(model, fed["client_datasets"][0], batch_size=64)
    assert n > 0
    assert 0.0 <= acc <= 1.0
    assert loss >= 0.0

    set_parameters(model, params_before)
    restored = get_parameters(model)
    assert len(restored) == len(params_before)
