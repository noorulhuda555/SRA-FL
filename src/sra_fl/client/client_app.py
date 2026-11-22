"""Flower client implementation for SRA-FL Module 1."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from flwr.client import NumPyClient
from flwr.common import Context
from torch.utils.data import Dataset

from sra_fl.client.local_trainer import evaluate_model, train_model
from sra_fl.models.cnn import create_model
from sra_fl.models.model_utils import get_parameters, set_parameters, train_eval_device


class FlowerClient(NumPyClient):
    """Simulated Flower client that trains on a local data partition."""

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        dataset_name: str,
        train_config: dict[str, Any],
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.train_config = train_config
        self.device = train_eval_device(str(train_config.get("device", "cpu")))
        self.model = create_model(dataset_name)

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        set_parameters(self.model, parameters)

        epochs = int(config.get("local_epochs", self.train_config.get("local_epochs", 1)))
        batch_size = int(config.get("batch_size", self.train_config.get("batch_size", 32)))
        learning_rate = float(
            config.get("learning_rate", self.train_config.get("learning_rate", 0.01))
        )
        momentum = float(config.get("momentum", self.train_config.get("momentum", 0.9)))
        proximal_mu = float(config.get("proximal_mu", 0.0))

        metrics = train_model(
            self.model,
            self.dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            device=self.device,
            proximal_mu=proximal_mu,
            global_parameters=parameters if proximal_mu > 0.0 else None,
        )
        return (
            get_parameters(self.model),
            len(self.dataset),
            {
                "train_loss": float(metrics["loss"]),
                "train_accuracy": float(metrics["accuracy"]),
                "client_id": float(self.client_id),
            },
        )

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, float]]:
        set_parameters(self.model, parameters)
        batch_size = int(config.get("batch_size", self.train_config.get("batch_size", 32)))
        loss, accuracy, num_samples = evaluate_model(
            self.model,
            self.dataset,
            batch_size=batch_size,
            device=self.device,
        )
        return float(loss), int(num_samples), {"accuracy": float(accuracy)}


def create_client_fn(
    client_datasets: list[Dataset],
    dataset_name: str,
    train_config: dict[str, Any],
) -> Callable[[Context], Any]:
    """Build a Flower ``client_fn(context)`` for simulation."""

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        if partition_id < 0 or partition_id >= len(client_datasets):
            raise IndexError(
                f"partition-id {partition_id} out of range for "
                f"{len(client_datasets)} clients"
            )
        client = FlowerClient(
            client_id=partition_id,
            dataset=client_datasets[partition_id],
            dataset_name=dataset_name,
            train_config=train_config,
        )
        return client.to_client()

    return client_fn
