"""Built-in metric implementations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from nameframe.metrics.base import BaseMetric
from nameframe.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("accuracy")
class Accuracy(BaseMetric):
    """Pixel-wise accuracy, with optional per-class breakdown."""

    def __init__(self, num_classes: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__()
        self.num_classes: Optional[int] = num_classes
        self._correct: float = 0.0
        self._total: float = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate correct and total pixel counts.

        Args:
            preds: ``(B, H, W)`` integer labels.
            targets: ``(B, H, W)`` integer labels.
        """
        self._correct += float((preds == targets).sum().item())
        self._total += float(targets.numel())

    def compute(self) -> Dict[str, float]:
        """Compute overall accuracy.

        Returns:
            ``{"accuracy": value}``.
        """
        acc: float = self._correct / self._total if self._total > 0 else 0.0
        return {"accuracy": acc}

    def reset(self) -> None:
        self._correct = 0.0
        self._total = 0.0


@METRIC_REGISTRY.register("dice_score")
class DiceScore(BaseMetric):
    """Per-class and mean Dice / IoU scores for segmentation."""

    def __init__(self, num_classes: int = 2, smooth: float = 1.0, **kwargs: Any) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        self.smooth: float = smooth
        self._intersection: torch.Tensor = torch.zeros(num_classes)
        self._union: torch.Tensor = torch.zeros(num_classes)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate intersection and union per class.

        Args:
            preds: ``(B, H, W)`` integer labels.
            targets: ``(B, H, W)`` integer labels.
        """
        for c in range(self.num_classes):
            pred_c: torch.Tensor = (preds == c)
            tgt_c: torch.Tensor = (targets == c)
            self._intersection[c] += float((pred_c & tgt_c).sum().item())
            self._union[c] += float((pred_c | tgt_c).sum().item())

    def compute(self) -> Dict[str, float]:
        """Compute per-class Dice and mean Dice.

        Returns:
            Dict with ``"dice_mean"`` and ``"dice_class_{i}"`` keys.
        """
        dice: torch.Tensor = (
            (2.0 * self._intersection + self.smooth)
            / (self._union + self.smooth)
        )
        result: Dict[str, float] = {"dice_mean": float(dice.mean().item())}
        for c in range(self.num_classes):
            result[f"dice_class_{c}"] = float(dice[c].item())
        return result

    def reset(self) -> None:
        self._intersection = torch.zeros(self.num_classes)
        self._union = torch.zeros(self.num_classes)


@METRIC_REGISTRY.register("confusion_matrix")
class ConfusionMatrix(BaseMetric):
    """Confusion matrix accumulation for multi-class segmentation."""

    def __init__(self, num_classes: int = 2, **kwargs: Any) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        self._cm: torch.Tensor = torch.zeros(num_classes, num_classes)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate per-pixel confusion counts.

        Args:
            preds: ``(B, H, W)`` integer labels.
            targets: ``(B, H, W)`` integer labels.
        """
        preds_flat: torch.Tensor = preds.flatten().long()
        targets_flat: torch.Tensor = targets.flatten().long()
        for t in range(self.num_classes):
            t_mask: torch.Tensor = (targets_flat == t)
            for p in range(self.num_classes):
                self._cm[t, p] += float((preds_flat[t_mask] == p).sum().item())

    def compute(self) -> Dict[str, float]:
        """Compute precision, recall, and per-class IoU from the confusion matrix.

        Returns:
            Dict with ``"prec_mean"``, ``"rec_mean"``, ``"iou_mean"``
            and per-class variants.
        """
        cm: torch.Tensor = self._cm
        tp: torch.Tensor = cm.diag()
        fp: torch.Tensor = cm.sum(dim=0) - tp
        fn: torch.Tensor = cm.sum(dim=1) - tp

        prec: torch.Tensor = tp / (tp + fp + 1e-8)
        rec: torch.Tensor = tp / (tp + fn + 1e-8)
        iou: torch.Tensor = tp / (tp + fp + fn + 1e-8)

        result: Dict[str, float] = {
            "prec_mean": float(prec.mean().item()),
            "rec_mean": float(rec.mean().item()),
            "iou_mean": float(iou.mean().item()),
        }
        for c in range(self.num_classes):
            result[f"prec_class_{c}"] = float(prec[c].item())
            result[f"rec_class_{c}"] = float(rec[c].item())
            result[f"iou_class_{c}"] = float(iou[c].item())
        return result

    def cm_tensor(self) -> torch.Tensor:
        """Return the raw confusion matrix.

        Returns:
            ``(num_classes, num_classes)`` tensor.
        """
        return self._cm.clone()

    def reset(self) -> None:
        self._cm = torch.zeros(self.num_classes, self.num_classes)


class MetricCollection:
    """Aggregate multiple :class:`BaseMetric` instances into a single call."""

    def __init__(self, metrics: List[BaseMetric]) -> None:
        """Initialize with a list of metric instances.

        Args:
            metrics: Pre-instantiated metric objects.
        """
        self._metrics: List[BaseMetric] = list(metrics)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Forward batch predictions to every managed metric."""
        for m in self._metrics:
            m.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        """Compute and merge results from all managed metrics.

        Returns:
            Merged dict of all metric name -> value pairs.
        """
        result: Dict[str, float] = {}
        for m in self._metrics:
            result.update(m.compute())
        return result

    def reset(self) -> None:
        """Reset all managed metrics."""
        for m in self._metrics:
            m.reset()
