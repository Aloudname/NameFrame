"""
`Metric` follows `MetricProtocol`,
serves as an optional mixin for metric components.

[NOTE] `update`, `compute`, and `reset` *must* be implemented in subclasses.
"""

from __future__ import annotations

from torch import Tensor

from nameframe.components.batch import Batch
from nameframe.utils.typing import MetricProtocol

__all__ = ["Metric", "MetricProtocol"]


class Metric:
    """
    optional mixin as a skeleton for metric implementations.

    [NOTE] `update`, `compute`, and `reset` *must* be implemented in subclasses.
    usage::

        class Accuracy(Metric):
            def __init__(self):
                self.correct = 0
                self.total = 0

            def update(self, pred, batch):
                self.correct += (pred.argmax(-1) == batch.target).sum().item()
                self.total += batch.target.numel()

            def compute(self):
                return {"acc": self.correct / max(self.total, 1)}

            def reset(self):
                self.correct = 0
                self.total = 0
    """

    # [NOTE] below methods must be implemented in subclasses.

    def update(self, pred: Tensor, batch: Batch) -> None:
        """
        accumulates one predict.

        [NOTE] *must* be implemented in subclasses.

        params:
        - `pred`: `Tensor` type, model output.
        - `batch`: `Batch` type of original batch with ground truth.
        """
        raise NotImplementedError

    def compute(self) -> dict[str, float]:
        """
        computes and returns the current metrics.

        [NOTE] *must* be implemented in subclasses.

        returns:
        - `dict[str, float]` type, metric name to value mapping.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        resets metric state to empty.

        [NOTE] *must* be implemented in subclasses.
        """
        raise NotImplementedError
