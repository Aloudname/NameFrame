"""Visualization utilities, training curves, confusion matrices, feature plots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from munch import Munch


class Visualizer:
    """Figures saved to ``output_dir/figures/``.

    Attributes:
        output_dir: Root dir for saved figures.
        dpi: Resolution.
        fig_format: File format (``"png"``, ``"pdf"``, ``"svg"``).
    """

    def __init__(self, output_dir: str, config: Munch) -> None:
        """
        Args:
            output_dir: Root dir.
            config: ``config.visualization``.
        """
        self.output_dir: str = os.path.join(output_dir, "figures")
        os.makedirs(self.output_dir, exist_ok=True)
        viz_cfg: Munch = getattr(config, "visualization", Munch())
        self.dpi: int = int(viz_cfg.get("dpi", 150))
        self.fig_format: str = str(viz_cfg.get("fig_format", "png"))

    def plot_training_curves(
        self, history: List[Dict[str, float]], fname: str = "training_curves"
    ) -> Optional[str]:
        """Plot loss and metric curves over epochs (train vs val).

        Args:
            history: List of per-epoch metric dicts with ``"train_loss"``,
                ``"val_loss"`` keys.
            fname: Output filename without extension name.

        Returns:
            Path to the saved figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        epochs: List[int] = list(range(1, len(history) + 1))
        train_loss: List[float] = [h.get("train_loss", 0.0) for h in history]
        val_loss: List[float] = [h.get("val_loss", 0.0) for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=self.dpi)
        ax1.plot(epochs, train_loss, label="train")
        ax1.plot(epochs, val_loss, label="val")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # plot any metrics besides loss
        metric_keys: set[str] = set()
        for h in history:
            metric_keys.update(h.keys())
        metric_keys.difference_update({"train_loss", "val_loss", "epoch"})
        for key in sorted(metric_keys):
            vals: List[float] = [h.get(key, 0.0) for h in history]
            ax2.plot(epochs, vals, label=key, marker="o", markersize=3)
        if metric_keys:
            ax2.set_xlabel("Epoch")
            ax2.set_title("Metrics")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        path: str = self._save_fig(fig, fname)
        plt.close(fig)
        return path

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        fname: str = "confusion_matrix",
    ) -> Optional[str]:
        """Plot a normalized confusion matrix as a heatmap.

        Args:
            cm: ``(num_classes, num_classes)`` confusion matrix.
            class_names: Optional list of class name strings.
            fname: Output filename without extension name.

        Returns:
            Path to the saved figure.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            return None

        cm_norm: np.ndarray = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
        num_classes: int = cm.shape[0]
        labels: List[str] = (
            class_names if class_names else [str(i) for i in range(num_classes)]
        )

        fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.8), max(5, num_classes * 0.7)), dpi=self.dpi)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                     xticklabels=labels, yticklabels=labels, ax=ax,
                     vmin=0, vmax=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Normalized Confusion Matrix")

        path: str = self._save_fig(fig, fname)
        plt.close(fig)
        return path

    def plot_feature_distribution(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        fname: str = "feature_tsne",
    ) -> Optional[str]:
        """Plot t-SNE or PCA visualization of extracted features.

        Uses t-SNE when scikit-learn is available, otherwise PCA.

        Args:
            features: ``(N, D)`` feature array.
            labels: ``(N,)`` integer label array.
            fname: Output filename without extension name.

        Returns:
            Path to the saved figure, or ``None`` if plotting fails.
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
        except ImportError:
            return None

        # reduce to 2D
        n: int = min(features.shape[0], 1000)
        idx: np.ndarray = np.random.choice(features.shape[0], n, replace=False)
        feats_sub: np.ndarray = features[idx]
        labels_sub: np.ndarray = labels[idx]

        projected: np.ndarray
        if n > 50:
            try:
                projected = TSNE(n_components=2, random_state=42).fit_transform(feats_sub)
            except Exception:
                projected = PCA(n_components=2).fit_transform(feats_sub)
        else:
            projected = PCA(n_components=2).fit_transform(feats_sub)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        unique_labels: np.ndarray = np.unique(labels_sub)
        for lab in unique_labels:
            mask: np.ndarray = labels_sub == lab
            ax.scatter(projected[mask, 0], projected[mask, 1],
                        label=str(lab), alpha=0.6, s=15)
        ax.legend(markerscale=2)
        ax.set_title("Feature Distribution (t-SNE / PCA)")

        path: str = self._save_fig(fig, fname)
        plt.close(fig)
        return path

    def _save_fig(self, fig: Any, name: str) -> str:
        """Save and ensure consistent naming."""
        path: str = os.path.join(self.output_dir, f"{name}.{self.fig_format}")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        return path
