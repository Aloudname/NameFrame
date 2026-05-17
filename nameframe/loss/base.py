"""Abstract loss interface, all losses must inherit from BaseLoss."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """Abstract base class for all losses.

    Defines the contract that :class:`Trainer` depends on.

    Attributes:
        config: The loss sub-tree of the project configuration.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with the loss config subtree.

        Args:
            config: The ``config.loss`` subtree (dict or Munch).
        """
        super().__init__()
        self.config: Dict[str, Any] = config

    @abstractmethod
    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Compute scalar loss from logits and targets.

        Args:
            logits: Model output of shape ``(B, C, H, W)``.
            targets: Ground truth labels of shape ``(B, H, W)``.
            **kwargs: Additional context (e.g. aux outputs).

        Returns:
            Scalar loss tensor (shape ``()``).
        """
        ...

    def get_components(self) -> Dict[str, float]:
        """Return per-component loss values for logging.

        Returns:
            Dict mapping component name to scalar float value.
        """
        return {}
