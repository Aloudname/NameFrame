"""User dataset stub, extends BaseDataset and registers itself via decorator."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from nameframe.dataset.base import BaseDataset
from nameframe.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("my_dataset")
class MyDataset(BaseDataset):
    """Example dataset.

    This stub demonstrates the minimum required interface. For real use,
    implement file discovery in ``__init__`` and loading in ``__getitem__``.
    """

    def __init__(self, config: Dict[str, Any], split: str = "train") -> None:
        """Initialize the dataset.

        Args:
            config: The ``config.data`` subtree.
            split: One of ``"train"``, ``"val"``, ``"test"``.
        """
        super().__init__(config, split)
        self.num_samples: int = int(config.get("num_samples", 100))
        self.num_classes: int = int(config.get("num_classes", 10))
        self.image_size: int = int(config.get("image_size", 64))
        self.num_channels: int = int(config.get("in_channels", 3))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve one sample.

        Args:
            idx: Sample index.

        Returns:
            ``(image, target)`` where image is ``(C, H, W)`` and target
            is ``(H, W)`` with integer class labels.
        """
        # synthetic data for demonstration — replace with real file loading
        image: torch.Tensor = torch.randn(self.num_channels, self.image_size, self.image_size)
        target: torch.Tensor = torch.randint(0, self.num_classes, (self.image_size, self.image_size))
        return image, target

    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_samples
