"""Local training and evaluation routines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def _global_param_tensors(
    model: nn.Module,
    global_parameters: Sequence[np.ndarray],
    device: torch.device,
) -> list[torch.Tensor]:
    """Map Flower state-dict parameter arrays onto ``model.parameters()`` tensors."""
    state_keys = list(model.state_dict().keys())
    key_to_array = {
        key: torch.tensor(np.array(value), dtype=torch.float32, device=device)
        for key, value in zip(state_keys, global_parameters, strict=True)
    }
    return [key_to_array[name] for name, _ in model.named_parameters()]


def _proximal_penalty(
    model: nn.Module,
    global_params: Sequence[torch.Tensor],
    proximal_mu: float,
) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for local_param, global_param in zip(model.parameters(), global_params, strict=True):
        penalty = penalty + torch.sum((local_param - global_param) ** 2)
    return (proximal_mu / 2.0) * penalty


def train_model(
    model: nn.Module,
    dataset: Dataset,
    *,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    device: torch.device | None = None,
    proximal_mu: float = 0.0,
    global_parameters: Sequence[np.ndarray] | None = None,
) -> dict[str, float]:
    """
    Train a model on a local client dataset.

    When ``proximal_mu > 0`` and ``global_parameters`` are provided, applies the
    FedProx proximal term.
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    use_proximal = proximal_mu > 0.0 and global_parameters is not None
    global_tensors = (
        _global_param_tensors(model, global_parameters, device) if use_proximal else None
    )

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if use_proximal and global_tensors is not None:
                loss = loss + _proximal_penalty(model, global_tensors, proximal_mu)

            loss.backward()
            optimizer.step()

            batch_size_actual = labels.size(0)
            total_loss += float(loss.item()) * batch_size_actual
            total_correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total_samples += batch_size_actual

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "num_samples": 0.0}

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "num_samples": float(total_samples),
    }


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> tuple[float, float, int]:
    """Evaluate model loss and accuracy. Returns (loss, accuracy, num_samples)."""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_size_actual = labels.size(0)
            total_loss += float(loss.item()) * batch_size_actual
            total_correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total_samples += batch_size_actual

    if total_samples == 0:
        return 0.0, 0.0, 0

    return total_loss / total_samples, total_correct / total_samples, total_samples
