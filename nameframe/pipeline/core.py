"""Core wires all components."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch.nn as nn
from munch import Munch
from torch.utils.data import DataLoader

from nameframe.dataset import build_dataloaders
from nameframe.loss import LOSS_REGISTRY
from nameframe.metrics import METRIC_REGISTRY, MetricCollection
from nameframe.model import build_model
from nameframe.pipeline.analyzer import Analyzer, MetricsBundle
from nameframe.pipeline.monitor import Monitor, MonitorReport
from nameframe.pipeline.trainer import Trainer, TrainerResult
from nameframe.pipeline.visualize import Visualizer
from nameframe.utils import seed_everything, tprint


@dataclass
class PipelineResult:
    """Container returned by :meth:`Pipeline.run`.

    Attributes:
        model_key: The model name from config.
        output_dir: Path to the output directory.
        best_epoch: Epoch index with the best metric.
        best_val_metric: Value of the best validation metric.
        metrics_json: Path to the saved metrics JSON file.
    """

    model_key: str = ""
    output_dir: str = ""
    best_epoch: int = 0
    best_val_metric: float = 0.0
    metrics_json: str = ""


class Pipeline:
    """Central orchestrator for the NameFrame training workflow.

    Wires together config, data loaders, model, loss, metrics, trainer,
    visualizer, and monitor. Depends only on abstract interfaces.
    """

    def __init__(self, config: Munch) -> None:
        """Initialize the pipeline from configuration.

        Args:
            config: Full project configuration Munch.
        """
        self.config: Munch = config
        self.output_dir: str = str(config.pipeline.get("output_dir", "./outputs"))
        os.makedirs(self.output_dir, exist_ok=True)

        seed: int = int(config.runtime.get("seed", 42))
        seed_everything(seed)

        self.analyzer: Analyzer = Analyzer()
        self.visualizer: Visualizer = Visualizer(self.output_dir, config)
        self.monitor: Monitor = Monitor(log_interval=float(config.runtime.get("monitor_interval", 5.0)))

        self.model: Optional[nn.Module] = None
        self.trainer: Optional[Trainer] = None
        self._data_loaders: Dict[str, DataLoader] = {}

    def run(self) -> PipelineResult:
        """Execute the full training workflow.

        build data -> build model -> train -> evaluate -> visualize.

        Returns:
            A :class:`PipelineResult` with training outcomes.
        """
        tprint("Building data loaders ...")
        self._data_loaders = self.build_data()

        tprint("Building model ...")
        self.model = self.build_model()

        tprint("Building loss and metrics ...")
        loss_fn: nn.Module = self.build_loss()
        metrics: MetricCollection = self.build_metrics()

        tprint("Starting training ...")
        self.monitor.start()
        train_result: TrainerResult = self.train(
            self.model, self._data_loaders, loss_fn, metrics
        )
        monitor_report: MonitorReport = self.monitor.stop()

        tprint("Running evaluation ...")
        eval_metrics: Dict[str, float] = self.evaluate(
            self.model, self._data_loaders.get("test", self._data_loaders["val"]), metrics
        )

        tprint("Generating visualizations ...")
        self._run_visualizations(train_result)

        tprint("Saving metrics ...")
        metrics_path: str = self._save_metrics(train_result, eval_metrics, monitor_report)

        best_metric: float = (
            train_result.best_metric
            if train_result.best_metric != float("-inf")
            else eval_metrics.get("accuracy", 0.0)
        )

        return PipelineResult(
            model_key=self.config.model.get("name", "unknown"),
            output_dir=self.output_dir,
            best_epoch=train_result.best_epoch,
            best_val_metric=best_metric,
            metrics_json=metrics_path,
        )

    def build_data(self) -> Dict[str, DataLoader]:
        """Build train/val/test DataLoaders from config."""
        return build_dataloaders(self.config)

    def build_model(self) -> nn.Module:
        """Instantiate the model from config via registry."""
        return build_model(self.config)

    def build_loss(self) -> nn.Module:
        """Build the loss function from config via registry."""
        loss_name: str = self.config.loss.get("name", "cross_entropy")
        loss_cfg: Dict[str, Any] = (
            self.config.loss.toDict() if isinstance(self.config.loss, Munch)
            else dict(self.config.loss)
        )
        loss_fn: nn.Module = LOSS_REGISTRY.get_or_build(loss_name, config=loss_cfg)
        return loss_fn

    def build_metrics(self) -> MetricCollection:
        """Build metric collection from config."""
        metric_names: list = list(self.config.metrics.get("names", ["accuracy"]))
        num_classes: int = int(self.config.model.get("num_classes", 2))
        instances: list = []
        for name in metric_names:
            instances.append(METRIC_REGISTRY.get_or_build(name, num_classes=num_classes))
        return MetricCollection(instances)

    def train(
        self,
        model: nn.Module,
        loaders: Dict[str, DataLoader],
        loss_fn: nn.Module,
        metrics: MetricCollection,
    ) -> TrainerResult:
        """Delegate training to the :class:`Trainer`.

        Args:
            model: The model to train.
            loaders: Dict with ``"train"``, ``"val"`` keys.
            loss_fn: Loss function.
            metrics: Metric collection.

        Returns:
            A :class:`TrainerResult`.
        """
        self.trainer = Trainer(model, self.config)
        self.trainer.set_loss(loss_fn)
        self.trainer.set_metrics(metrics)
        return self.trainer.fit(
            loaders["train"],
            loaders.get("val", loaders["train"]),
            int(self.config.train.epochs),
        )

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        metrics: MetricCollection,
    ) -> Dict[str, float]:
        """Run evaluation using the :class:`Trainer`.

        Args:
            model: The trained model.
            loader: Test or validation loader.
            metrics: Metric collection.

        Returns:
            Dict of metric name -> value.
        """
        if self.trainer is None:
            self.trainer = Trainer(model, self.config)
            self.trainer.set_metrics(metrics)
            dummy_loss: nn.Module = nn.CrossEntropyLoss()
            self.trainer.set_loss(dummy_loss)

        if self.trainer.model is not model:
            self.trainer.model = model

        return self.trainer.validate(loader)

    def export_onnx(self, model: nn.Module, input_shape: tuple = (1, 3, 224, 224)) -> Optional[str]:
        """Export the model to ONNX format.

        Args:
            model: The model to export.
            input_shape: Dummy input shape.

        Returns:
            Path to the exported ``.onnx`` file, or ``None`` if export fails.
        """
        try:
            import torch

            path: str = os.path.join(self.output_dir, "model.onnx")
            dummy: torch.Tensor = torch.randn(*input_shape, device="cpu")
            torch.onnx.export(model.cpu(), dummy, path, opset_version=14,
                              input_names=["input"], output_names=["output"],
                              dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
            return path
        except Exception as e:
            tprint(f"ONNX export failed: {e}")
            return None

    def _run_visualizations(self, train_result: TrainerResult) -> None:
        """Run post-training visualizations."""
        if not self.config.visualization.get("enabled", True):
            return
        if train_result.history:
            self.visualizer.plot_training_curves(train_result.history)

    def _save_metrics(
        self,
        train_result: TrainerResult,
        eval_metrics: Dict[str, float],
        monitor_report: MonitorReport,
    ) -> str:
        """Save all metrics to a JSON file.

        Returns:
            Path to the JSON file.
        """
        data: Dict[str, Any] = {
            "best_epoch": train_result.best_epoch,
            "best_metric": train_result.best_metric,
            "eval_metrics": eval_metrics,
            "history": train_result.history,
            "monitor": {
                "peak_gpu_memory_mib": monitor_report.peak_gpu_memory,
                "avg_gpu_util_pct": monitor_report.avg_gpu_util,
                "peak_cpu_memory_mib": monitor_report.peak_cpu_memory,
            },
        }
        path: str = os.path.join(self.output_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=float)
        return path
