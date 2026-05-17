"""User model stub, extends BaseModel and registers itself via decorator."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nameframe.model.base import BaseModel
from nameframe.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("my_model")
class MyModel(BaseModel):
    """Example"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Init the model from its config subtree.

        Args:
            config: The ``config.model`` dictionary.
        """
        super().__init__(config)
        in_channels: int = int(config.get("in_channels", 3))
        num_classes: int = int(config.get("num_classes", 10))
        hidden: int = int(config.get("hidden_dim", 128))

        self.conv1: nn.Conv2d = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.conv2: nn.Conv2d = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3: nn.Conv2d = nn.Conv2d(hidden, num_classes, 1)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(hidden)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing class logits.

        Args:
            x: Input tensor ``(B, C, H, W)``.

        Returns:
            Logits tensor ``(B, num_classes, H, W)``.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x
