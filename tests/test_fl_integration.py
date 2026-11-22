"""Lightweight federated learning integration tests."""

from __future__ import annotations

from sra_fl.server.orchestration import run_simulation
from sra_fl.utils.config_loader import load_config


def test_module1_fedavg_short_run():
    config = load_config(
        dataset="fashion_mnist",
        overrides={
            "num_clients": 4,
            "num_rounds": 1,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
            "min_available_clients": 4,
            "local_epochs": 1,
            "batch_size": 64,
            "strategy": "fedavg",
            "partition": "iid",
            "num_cpus_per_client": 0.25,
            "enable_heterogeneity": False,
            "enable_rag": False,
            "enable_communication": False,
            "enable_privacy": False,
        },
    )
    result = run_simulation(config)
    assert result["final_accuracy"] is not None
    assert 0.0 <= float(result["final_accuracy"]) <= 1.0
    assert result["metrics_path"].endswith(".json")
    assert result["client_environment"] is None


def test_module2_hetero_short_run():
    config = load_config(
        dataset="fashion_mnist",
        overrides={
            "num_clients": 6,
            "num_rounds": 1,
            "fraction_fit": 1.0,
            "fraction_evaluate": 0.5,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
            "min_available_clients": 2,
            "local_epochs": 1,
            "batch_size": 64,
            "strategy": "fedavg",
            "partition": "iid",
            "enable_heterogeneity": True,
            "enable_audit": False,
            "enable_negotiation": False,
            "enable_rag": False,
            "enable_communication": False,
            "enable_privacy": False,
            "dropout_probability": 0.0,
            "seconds_per_round": 1000.0,
        },
    )
    result = run_simulation(config)
    assert result["client_environment"] is not None
    assert result["final_accuracy"] is not None
    assert "module2_" in result["metrics_path"]


def test_module3_and_4_short_run():
    config = load_config(
        dataset="fashion_mnist",
        overrides={
            "num_clients": 6,
            "num_rounds": 1,
            "fraction_fit": 1.0,
            "fraction_evaluate": 0.5,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
            "local_epochs": 1,
            "batch_size": 64,
            "strategy": "fedavg",
            "partition": "iid",
            "enable_heterogeneity": True,
            "enable_audit": True,
            "enable_negotiation": True,
            "enable_rag": False,
            "enable_communication": False,
            "enable_privacy": False,
            "dishonest_fraction": 0.5,
            "dropout_probability": 0.0,
            "seconds_per_round": 1000.0,
        },
    )
    result = run_simulation(config)
    assert result["audit_engine"] is not None
    assert len(result["negotiation_logs"]) >= 1
    assert result["final_accuracy"] is not None
    assert "module4_" in result["metrics_path"]


def test_modules_5_6_7_short_run():
    config = load_config(
        dataset="fashion_mnist",
        overrides={
            "num_clients": 6,
            "num_rounds": 1,
            "fraction_fit": 1.0,
            "fraction_evaluate": 0.5,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
            "local_epochs": 1,
            "batch_size": 64,
            "strategy": "fedavg",
            "partition": "iid",
            "enable_heterogeneity": True,
            "enable_audit": True,
            "enable_negotiation": True,
            "enable_rag": True,
            "enable_communication": True,
            "enable_privacy": True,
            "enable_llm": False,
            "dishonest_fraction": 0.3,
            "dropout_probability": 0.0,
            "seconds_per_round": 1000.0,
            "comm_epsilon": 0.0,
        },
    )
    result = run_simulation(config)
    assert result["orchestrator"] is not None
    assert result["upload_manager"] is not None
    assert result["privacy_manager"] is not None
    assert len(result["rag_logs"]) >= 1
    assert result["final_accuracy"] is not None
    assert "module7_" in result["metrics_path"]
