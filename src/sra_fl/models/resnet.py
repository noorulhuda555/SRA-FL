"""Optional ResNet helper (available for later experiments)."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def create_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Create a ResNet-18 classifier head for CIFAR-scale inputs."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
