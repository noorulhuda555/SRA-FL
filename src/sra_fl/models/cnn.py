"""CNN models for Fashion-MNIST and CIFAR-10."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Simple CNN suitable for both grayscale and RGB image classification."""

    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # Fashion-MNIST: 28 -> 14 -> 7; CIFAR-10: 32 -> 16 -> 8
        self.fc1 = nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.input_channels = input_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def create_model(dataset_name: str, num_classes: int = 10) -> CNN:
    """Create the default CNN for the selected dataset."""
    name = dataset_name.lower()
    if name in {"fashion_mnist", "fashion-mnist", "fashionmnist"}:
        return CNN(input_channels=1, num_classes=num_classes)
    if name in {"cifar10", "cifar-10"}:
        return CNN(input_channels=3, num_classes=num_classes)
    raise ValueError(f"Unsupported dataset for model creation: {dataset_name}")
