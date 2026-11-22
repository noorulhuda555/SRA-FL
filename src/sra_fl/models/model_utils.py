"""Model parameter helpers shared by client and server."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """Extract model parameters as NumPy arrays (Flower format)."""
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: Sequence[np.ndarray]) -> None:
    """Load Flower NumPy parameters into a model."""
    state_dict = model.state_dict()
    new_state = {
        key: torch.tensor(np.array(value))
        for key, value in zip(state_dict.keys(), parameters, strict=True)
    }
    model.load_state_dict(new_state, strict=True)


def train_eval_device(requested: str = "cpu") -> torch.device:
    """Resolve the torch device used for local training/evaluation."""
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")
