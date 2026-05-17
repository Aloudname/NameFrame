"""Built-in loss functions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nameframe.loss.base import BaseLoss
from nameframe.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss with optional class weights and label smoothing."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        weight: Optional[torch.Tensor] = None
        if config.get("class_weights"):
            weight = torch.tensor(config["class_weights"], dtype=torch.float32)
        self.smoothing: float = float(config.get("label_smoothing", 0.0))
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=self.smoothing
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: ``(B, C, H, W)``.
            targets: ``(B, H, W)`` with class indices.

        Returns:
            Scalar loss.
        """
        return self.criterion(logits, targets.long())


@LOSS_REGISTRY.register("dice")
class DiceLoss(BaseLoss):
    """Soft Dice loss for segmentation tasks.

    Optionally restricts computation to foreground classes only.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.foreground_only: bool = bool(config.get("foreground_only", True))
        self.smooth: float = float(config.get("dice_smooth", 1.0))

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Compute soft Dice loss.

        Args:
            logits: ``(B, C, H, W)``.
            targets: ``(B, H, W)`` with class indices.

        Returns:
            Scalar loss (1 - Dice).
        """
        num_classes: int = logits.shape[1]
        probs: torch.Tensor = F.softmax(logits, dim=1)

        # one-hot encode targets
        targets_one_hot: torch.Tensor = F.one_hot(
            targets.long(), num_classes=num_classes
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        start_idx: int = 1 if self.foreground_only else 0

        dims: tuple[int, ...] = (0, 2, 3)
        intersection: torch.Tensor = (probs[:, start_idx:] * targets_one_hot[:, start_idx:]).sum(dim=dims)
        union: torch.Tensor = (
            probs[:, start_idx:] + targets_one_hot[:, start_idx:]
        ).sum(dim=dims)

        dice: torch.Tensor = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


@LOSS_REGISTRY.register("focal")
class FocalLoss(BaseLoss):
    """Focal loss for addressing class imbalance."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.alpha: float = float(config.get("focal_alpha", 0.25))
        self.gamma: float = float(config.get("focal_gamma", 2.0))

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: ``(B, C, H, W)``.
            targets: ``(B, H, W)`` with class indices.

        Returns:
            Scalar loss.
        """
        ce: torch.Tensor = F.cross_entropy(logits, targets.long(), reduction="none")
        pt: torch.Tensor = torch.exp(-ce)
        focal: torch.Tensor = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


@LOSS_REGISTRY.register("composite")
class CompositeLoss(BaseLoss):
    """Weighted sum of multiple loss components, configured via config."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.components: List[Dict[str, Any]] = list(config.get("components", []))

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Compute weighted composite loss.

        Args:
            logits: ``(B, C, H, W)``.
            targets: ``(B, H, W)`` with class indices.

        Returns:
            Weighted sum of component losses.
        """
        total: torch.Tensor = torch.tensor(0.0, device=logits.device)
        for comp in self.components:
            name: str = comp["name"]
            weight: float = float(comp.get("weight", 1.0))
            loss_fn: nn.Module = LOSS_REGISTRY.get_or_build(name, config=comp)
            total = total + weight * loss_fn(logits, targets, **kwargs)
        return total
