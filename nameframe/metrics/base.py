"""Abstract metric interface, all metrics must inherit from BaseMetric."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class BaseMetric(ABC):
    """Abstract base class for all metrics.

    Supports two modes:
      1. **Stateless** — :meth:`compute` directly from raw predictions.
      2. **Stateful** — accumulate via :meth:`update`, then :meth:`compute`.
    """

    def __init__(self) -> None:
        """Initialize the metric (called once per evaluation run)."""

    @abstractmethod
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch of predictions.

        Args:
            preds: Predicted labels ``(B, H, W)`` or logits ``(B, C, H, W)``.
            targets: Ground truth labels ``(B, H, W)``.
        """
        ...

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Compute final metric values from accumulated state.

        Returns:
            Dict mapping metric name strings to scalar values.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state for a fresh evaluation pass."""
        ...
