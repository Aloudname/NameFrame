"""Abstract dataset interface, all datasets must inherit from BaseDataset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]], ABC):
    """Abstract base class for all datasets.

    Defines the minimum contract that the data loading pipeline depends on.

    Attributes:
        config: The data sub-tree of the project configuration.
    """

    def __init__(self, config: Dict[str, Any], split: str = "train") -> None:
        """Initialize the dataset.

        Args:
            config: The ``config.data`` subtree (dict or Munch).
            split: One of ``"train"``, ``"val"``, ``"test"``.
        """
        self.config: Dict[str, Any] = config
        self.split: str = split

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve one sample.

        Args:
            idx: Sample index.

        Returns:
            ``(image, target)`` where *image* has shape ``(C, H, W)``
            and *target* has shape ``(H, W)`` with integer class labels.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in this split."""
        ...

    @property
    def num_classes(self) -> int:
        """Number of output classes (read from config)."""
        return int(self.config.get("num_classes", 1))

    @property
    def class_names(self) -> List[str]:
        """Human-readable class names (read from config)."""
        return list(self.config.get("class_names", []))
