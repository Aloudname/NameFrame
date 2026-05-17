"""Abstract model interface, all models must inherit from BaseModel."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models.

    Defines the minimum contract that the :class:`Pipeline` and
    :class:`Trainer` depend on. Concrete models must implement
    :meth:`forward` and may optionally override the auxiliary methods.

    Attributes:
        config: The model sub-tree of the project configuration
            (injected by :func:`build_model`).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the model with its config subtree.

        Args:
            config: The ``config.model`` dictionary (or Munch).
        """
        super().__init__()
        self.config: Dict[str, Any] = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing class logits.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Logits tensor of shape ``(B, num_classes, H, W)``.
        """
        ...

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return intermediate feature maps (before the final classification head).

        The default implementation calls :meth:`forward`. Override for
        models that expose a separate feature extractor.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Feature tensor.
        """
        return self.forward(x)

    def get_aux_outputs(self) -> Dict[str, torch.Tensor]:
        """Return auxiliary outputs for multi-task loss computation.

        Called after :meth:`forward`. The default returns an empty dict.

        Returns:
            A dict mapping auxiliary head names to their output tensors.
        """
        return {}

    def forward_with_aux(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass returning both logits and auxiliary outputs.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            A ``(logits, aux_dict)`` tuple.
        """
        logits: torch.Tensor = self.forward(x)
        aux: Dict[str, torch.Tensor] = self.get_aux_outputs()
        return logits, aux
