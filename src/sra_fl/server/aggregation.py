"""Federated aggregation strategies: FedAvg and FedProx."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flwr.common import NDArrays, Parameters, ndarrays_to_parameters
from flwr.server.strategy import FedAvg, FedProx, Strategy

from sra_fl.evaluation.metrics import weighted_average


def build_strategy(
    strategy_name: str,
    *,
    fraction_fit: float,
    fraction_evaluate: float,
    min_fit_clients: int,
    min_evaluate_clients: int,
    min_available_clients: int,
    initial_parameters: Parameters | None = None,
    evaluate_fn: Callable[[int, NDArrays, dict[str, Any]], Any] | None = None,
    on_fit_config_fn: Callable[[int], dict[str, Any]] | None = None,
    on_evaluate_config_fn: Callable[[int], dict[str, Any]] | None = None,
    proximal_mu: float = 0.01,
) -> Strategy:
    """Create a Flower strategy for FedAvg or FedProx."""
    common_kwargs = {
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_available_clients": min_available_clients,
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate_fn,
        "on_fit_config_fn": on_fit_config_fn,
        "on_evaluate_config_fn": on_evaluate_config_fn,
        "fit_metrics_aggregation_fn": weighted_average,
        "evaluate_metrics_aggregation_fn": weighted_average,
    }

    name = strategy_name.lower()
    if name == "fedavg":
        return FedAvg(**common_kwargs)
    if name == "fedprox":
        return FedProx(**common_kwargs, proximal_mu=proximal_mu)
    raise ValueError(f"Unsupported strategy: {strategy_name}. Use 'fedavg' or 'fedprox'.")


def parameters_from_ndarrays(ndarrays: list) -> Parameters:
    """Convert NumPy parameter arrays to Flower Parameters."""
    return ndarrays_to_parameters(ndarrays)
