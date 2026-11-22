"""Image preprocessing transforms."""

from __future__ import annotations

from torchvision import transforms


def get_transforms(dataset_name: str, train: bool = True) -> transforms.Compose:
    """Return torchvision transforms for the selected dataset."""
    name = dataset_name.lower()
    if name in {"fashion_mnist", "fashion-mnist", "fashionmnist"}:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )

    if name in {"cifar10", "cifar-10"}:
        if train:
            return transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616),
                    ),
                ]
            )
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        )

    raise ValueError(f"Unsupported dataset for transforms: {dataset_name}")
