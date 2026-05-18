"""Checkpoint save / load and rotation management."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint to disk.

    Args:
        path: Save file path in ``.pt``.
        model: The model whose ``state_dict`` will be saved.
        optimizer: Optional optimizer to persist.
        scheduler: Optional LR scheduler to persist.
        epoch: Current epoch number (0-start).
        metrics: Optional dict of scalar metrics to embed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint and restore model / optimizer / scheduler state.

    Args:
        path: Checkpoint file path.
        model: Model to load weights into (in-place).
        optimizer: Optional optimizer to restore.
        scheduler: Optional LR scheduler to restore.
        device: Target device for the checkpoint tensors. If ``None``,
            uses the current model parameter device or auto-detects.

    Returns:
        The full checkpoint dictionary (epoch, metrics, etc.).
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    checkpoint: Dict[str, Any] = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


class CheckpointManager:
    """Manages checkpoint rotation, keeping only the top-K best checkpoints.

    Attributes:
        output_dir: Checkpoint save dir.
        keep_top_k: Max checkpoints.
        mode: ``"max"`` for higher-better metrics, ``"min"`` for lower-better.
    """

    def __init__(
        self,
        output_dir: str,
        keep_top_k: int = 3,
        mode: str = "max",
    ) -> None:
        """
        Args:
            output_dir: Checkpoint save dir.
            keep_top_k: Max checkpoints.
            mode: ``"max"`` or ``"min"``.
        """
        self.output_dir: str = output_dir
        self.keep_top_k: int = keep_top_k
        self.mode: str = mode
        self._records: List[Dict[str, Any]] = []
        self._record_path: str = os.path.join(output_dir, "checkpoints.json")

        os.makedirs(output_dir, exist_ok=True)
        self._load_records()

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[object],
        epoch: int,
        metrics: Dict[str, Any],
        metric_key: str,
    ) -> Optional[str]:
        """Save a checkpoint and rotate old ones if necessary.

        Args:
            model: Model to persist.
            optimizer: Optional optimizer.
            scheduler: Optional scheduler.
            epoch: Current epoch.
            metrics: Dict of metric name -> value.
            metric_key: Which metric key to use for ranking.

        Returns:
            Path to the saved checkpoint file,
            or ``None`` if this checkpoint failed to make top-K cut.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        filename: str = f"epoch_{epoch:04d}.pt"
        filepath: str = os.path.join(self.output_dir, filename)

        save_checkpoint(filepath, model, optimizer, scheduler, epoch, metrics)

        score: float = float(metrics.get(metric_key, 0.0))
        self._records.append(
            {
                "epoch": epoch,
                "score": score,
                "filepath": filepath,
            }
        )
        self._records.sort(
            key=lambda r: r["score"],
            reverse=(self.mode == "max"),
        )

        removed: Optional[str] = None
        while len(self._records) > self.keep_top_k:
            stale: Dict[str, Any] = self._records.pop()
            if os.path.exists(stale["filepath"]):
                os.remove(stale["filepath"])
            removed = stale["filepath"]

        self._save_records()
        return filepath if removed != filepath else None

    def best_path(self) -> Optional[str]:
        """Return the path to the best checkpoint, or ``None``."""
        if not self._records:
            return None
        return str(self._records[0]["filepath"])

    def _load_records(self) -> None:
        if os.path.exists(self._record_path):
            with open(self._record_path, "r") as f:
                self._records = json.load(f)

    def _save_records(self) -> None:
        with open(self._record_path, "w") as f:
            json.dump(self._records, f, indent=2)
