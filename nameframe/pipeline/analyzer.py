"""Metrics analyzer, computation of classification/segmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class MetricsBundle:
    """Container for computed evaluation metrics.

    Attributes:
        summary: Flat dict of metric name -> scalar value.
        per_class: Dict of per-class metric name -> list of per-class values.
        confusion_matrix: Raw confusion matrix ``(num_classes, num_classes)``.
        roc_curves: Optional dict of class_index -> (fpr, tpr) tuples.
    """

    summary: Dict[str, float] = field(default_factory=dict)
    per_class: Dict[str, List[float]] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    roc_curves: Optional[Dict[int, tuple]] = None


class Analyzer:
    """Stateless metrics computer for classification and segmentation.

    Takes raw predictions and targets, returns a :class:`MetricsBundle`.
    """

    def compute(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        num_classes: int = 2,
    ) -> MetricsBundle:
        """Compute a comprehensive set of metrics.

        Args:
            preds: Predicted class indices ``(N,)`` or ``(B, H, W)``.
            targets: Ground truth labels, same shape as *preds*.
            probs: Optional probability/logit tensor ``(N, C)`` for ROC.
            num_classes: Number of classes.

        Returns:
            A :class:`MetricsBundle` with summary and per-class results.
        """
        preds_flat: np.ndarray = preds.cpu().numpy().flatten().astype(np.int64)
        targets_flat: np.ndarray = targets.cpu().numpy().flatten().astype(np.int64)

        cm: np.ndarray = self._confusion_matrix(preds_flat, targets_flat, num_classes)

        # summary metrics
        tp: np.ndarray = np.diag(cm)
        fp: np.ndarray = cm.sum(axis=0) - tp
        fn: np.ndarray = cm.sum(axis=1) - tp

        eps: float = 1e-8
        prec: np.ndarray = tp / (tp + fp + eps)
        rec: np.ndarray = tp / (tp + fn + eps)
        iou: np.ndarray = tp / (tp + fp + fn + eps)
        dice: np.ndarray = 2 * tp / (2 * tp + fp + fn + eps)
        acc: float = float(tp.sum() / (cm.sum() + eps))

        summary: Dict[str, float] = {
            "accuracy": acc,
            "prec_mean": float(prec.mean()),
            "rec_mean": float(rec.mean()),
            "iou_mean": float(iou.mean()),
            "dice_mean": float(dice.mean()),
        }

        per_class: Dict[str, List[float]] = {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "iou": iou.tolist(),
            "dice": dice.tolist(),
        }

        return MetricsBundle(
            summary=summary,
            per_class=per_class,
            confusion_matrix=cm,
        )

    def summarize(self, bundle: MetricsBundle) -> str:
        """Format a :class:`MetricsBundle` as a readable multi-line summary.

        Args:
            bundle: The computed metrics.

        Returns:
            Human-readable summary string.
        """
        lines: List[str] = ["--- Metrics Summary ---"]
        for name, value in bundle.summary.items():
            lines.append(f"  {name}: {value:.4f}")
        if bundle.per_class:
            num_classes: int = len(list(bundle.per_class.values())[0])
            for c in range(num_classes):
                parts: List[str] = []
                for metric_name, values in bundle.per_class.items():
                    parts.append(f"{metric_name}={values[c]:.4f}")
                lines.append(f"  class_{c}: " + ", ".join(parts))
        return "\n".join(lines)

    def per_class_report(self, bundle: MetricsBundle) -> str:
        """Generate a per-class precision/recall/F1 report.

        Args:
            bundle: The computed metrics.

        Returns:
            Per-class report string.
        """
        prec: List[float] = bundle.per_class.get("precision", [])
        rec: List[float] = bundle.per_class.get("recall", [])
        lines: List[str] = ["Class | Precision | Recall | F1"]
        lines.append("-" * 35)
        for c, (p, r) in enumerate(zip(prec, rec)):
            f1: float = 2 * p * r / (p + r + 1e-8)
            lines.append(f"  {c:3d} |     {p:.4f} |  {r:.4f} | {f1:.4f}")
        return "\n".join(lines)

    @staticmethod
    def _confusion_matrix(
        preds: np.ndarray,
        targets: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        cm: np.ndarray = np.zeros((num_classes, num_classes), dtype=np.float64)
        for t in range(num_classes):
            t_mask: np.ndarray = (targets == t)
            if t_mask.sum() == 0:
                continue
            for p in range(num_classes):
                cm[t, p] = float((preds[t_mask] == p).sum())
        return cm
