"""Training loop, Trainer encapsulates the training/validation/evaluation loop."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from munch import Munch
from torch.utils.data import DataLoader

from nameframe.optim import build_optimizer, build_scheduler
from nameframe.utils import tprint
from nameframe.utils.checkpoint import CheckpointManager, save_checkpoint
from nameframe.utils.device import auto_device


@dataclass
class TrainerResult:
    """Container for training results.

    Attributes:
        history: List of per-epoch metric dicts.
        best_epoch: Epoch index with the best monitored metric.
        best_metric: Value of the best monitored metric.
        checkpoint_path: Path to the best checkpoint file.
    """

    history: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_metric: float = 0.0
    checkpoint_path: str = ""


class ModelEMA:
    """Exponential Moving Average of model weights.

    Attributes:
        model: The original model.
        decay: EMA decay rate.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model: nn.Module = model
        self.decay: float = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """Apply one EMA update step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.shadow[name] * self.decay + param.data * (1.0 - self.decay)

    def apply_shadow(self) -> None:
        """Replace model weights with EMA shadow weights (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self) -> None:
        """Restore original model weights from shadow application."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self._backup[name]
        self._backup.clear()


class Trainer:
    """Training loop engine.

    Knows nothing about model architecture or data format — only depends
    on ``nn.Module``, ``DataLoader``, and loss/metrics through their
    abstract interfaces.

    Attributes:
        model: The model being trained.
        config: Full project configuration Munch.
        device: The torch device used for training.
    """

    def __init__(self, model: nn.Module, config: Munch) -> None:
        """Initialize the trainer.

        Args:
            model: A :class:`BaseModel` subclass instance.
            config: Full project configuration Munch.
        """
        self.model: nn.Module = model
        self.config: Munch = config
        self.device: torch.device = auto_device(config.runtime.get("device"))
        self.model.to(self.device)

        self.optimizer: torch.optim.Optimizer = build_optimizer(model, config)
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self._use_amp: bool = bool(config.runtime.get("use_amp", True))
        if self._use_amp and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

        self.ema: Optional[ModelEMA] = None
        if float(config.train.get("ema_decay", 0.0)) > 0:
            self.ema = ModelEMA(model, float(config.train.ema_decay))

        self._output_dir: str = str(config.pipeline.get("output_dir", "./outputs"))
        os.makedirs(self._output_dir, exist_ok=True)

        self._chk_mgr: CheckpointManager = CheckpointManager(
            os.path.join(self._output_dir, "checkpoints"),
            keep_top_k=int(config.pipeline.get("checkpoint_keep_top_k", 3)),
        )

        self._loss_fn: Optional[nn.Module] = None
        self._metrics: Optional[Any] = None  # MetricCollection

        self._history: List[Dict[str, float]] = []
        self._best_epoch: int = 0
        self._best_metric_val: float = float("-inf")
        self._patience_counter: int = 0
        self._early_stop_patience: int = int(config.train.get("early_stop_patience", 10))
        self._early_stop_metric: str = str(config.train.get("early_stop_metric", "val_loss"))

    def set_loss(self, loss_fn: nn.Module) -> None:
        """Inject the loss function.

        Args:
            loss_fn: A :class:`BaseLoss` instance.
        """
        self._loss_fn = loss_fn

    def set_metrics(self, metrics: Any) -> None:
        """Inject the metrics collection.

        Args:
            metrics: A :class:`MetricCollection` instance.
        """
        self._metrics = metrics

    def fit(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        epochs: int,
    ) -> TrainerResult:
        """Run the full training loop with early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs to train.

        Returns:
            A :class:`TrainerResult` with training history and best checkpoint info.

        Raises:
            ValueError: If loss_fn or metrics have not been set.
        """
        if self._loss_fn is None:
            raise ValueError("loss_fn not set. Call trainer.set_loss() first.")
        if self._metrics is None:
            raise ValueError("metrics not set. Call trainer.set_metrics() first.")

        # build scheduler after data loaders are known (for step-based schedulers)
        steps_per_epoch: int = len(train_loader)
        scheduler: object = build_scheduler(self.optimizer, self.config, steps_per_epoch)

        for epoch in range(1, epochs + 1):
            tprint(f"Epoch {epoch}/{epochs}")

            train_metrics: Dict[str, float] = self.train_one_epoch(train_loader)
            val_metrics: Dict[str, float] = self.validate(val_loader)

            epoch_record: Dict[str, float] = {"epoch": float(epoch), **train_metrics, **val_metrics}
            self._history.append(epoch_record)

            # step epoch-level scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                monitor_val: float = val_metrics.get(self._early_stop_metric, 0.0)
                scheduler.step(monitor_val)
            else:
                scheduler.step()

            # checkpointing
            monitor_score: float = val_metrics.get(self._early_stop_metric, 0.0)
            is_plateau_based: bool = "loss" in self._early_stop_metric.lower()
            is_better: bool = (
                monitor_score < self._best_metric_val if is_plateau_based
                else monitor_score > self._best_metric_val
            )

            if self._best_metric_val == float("-inf") or is_better:
                self._best_metric_val = monitor_score
                self._best_epoch = epoch
                self._patience_counter = 0
                # save best checkpoint
                ckpt_path: Optional[str] = self._chk_mgr.save(
                    self.model, self.optimizer, scheduler, epoch,
                    epoch_record, self._early_stop_metric,
                )
                if ckpt_path:
                    tprint(f"  [best] saved {ckpt_path}")
            else:
                self._patience_counter += 1

            current_lr: float = float(self.optimizer.param_groups[0]["lr"])
            tprint(f"  lr={current_lr:.2e} | train_loss={train_metrics.get('train_loss', 0.0):.4f} | val_loss={val_metrics.get('val_loss', 0.0):.4f}")

            if self._patience_counter >= self._early_stop_patience:
                tprint(f"Early stopping at epoch {epoch}")
                break

        return TrainerResult(
            history=self._history,
            best_epoch=self._best_epoch,
            best_metric=self._best_metric_val,
            checkpoint_path=self._chk_mgr.best_path() or "",
        )

    def train_one_epoch(
        self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """Run a single training epoch.

        Args:
            loader: Training data loader.

        Returns:
            Dict with ``"train_loss"`` and other batch-aggregated metrics.
        """
        self.model.train()
        total_loss: float = 0.0
        batches: int = 0
        grad_accum: int = int(self.config.train.get("grad_accumulation_steps", 1))

        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            if self._use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss: torch.Tensor = self._loss_fn(self.model(images), targets)
                self.scaler.scale(loss).backward()
            else:
                loss = self._loss_fn(self.model(images), targets)
                loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                # gradient clipping
                grad_clip: float = float(self.config.train.get("grad_clip", 1.0))
                if self._use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                if self._use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

                if self.ema is not None:
                    self.ema.update()

            total_loss += float(loss.item())
            batches += 1

        avg_loss: float = total_loss / max(batches, 1)
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(
        self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """Run validation on the entire loader.

        Args:
            loader: Validation data loader.

        Returns:
            Dict with ``"val_loss"`` and metric keys.
        """
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()

        total_loss: float = 0.0
        batches: int = 0
        if self._metrics is not None:
            self._metrics.reset()

        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            logits: torch.Tensor = self.model(images)
            if self._loss_fn is not None:
                total_loss += float(self._loss_fn(logits, targets).item())

            preds: torch.Tensor = logits.argmax(dim=1)
            if self._metrics is not None:
                self._metrics.update(preds, targets)

            batches += 1

        if self.ema is not None:
            self.ema.restore()

        result: Dict[str, float] = {"val_loss": total_loss / max(batches, 1)}
        if self._metrics is not None:
            result.update(self._metrics.compute())
        return result

    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        keep_images: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run inference and return predictions.

        Args:
            loader: Data loader for inference.
            keep_images: If ``True``, include input images in output.

        Returns:
            Dict with ``"preds"``, ``"targets"``, and optionally ``"images"``.
        """
        self.model.eval()
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        all_images: List[torch.Tensor] = []

        for images, targets in loader:
            images = images.to(self.device)
            logits: torch.Tensor = self.model(images)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_targets.append(targets.cpu())
            if keep_images:
                all_images.append(images.cpu())

        result: Dict[str, torch.Tensor] = {
            "preds": torch.cat(all_preds),
            "targets": torch.cat(all_targets),
        }
        if keep_images:
            result["images"] = torch.cat(all_images)
        return result
