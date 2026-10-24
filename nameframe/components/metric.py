"""Optional base for stateful metrics; plain protocol implementations also work."""

from __future__ import annotations

from torch import Tensor

from nameframe.components.batch import Batch
from nameframe.utils.typing import MetricProtocol

__all__ = ["Metric", "MetricProtocol"]


class Metric:
    """Implement update/compute/reset; each instance owns its accumulated state."""

    config_schema: dict = {}

    def update(self, pred: Tensor, batch: Batch) -> None:
        raise NotImplementedError

    def compute(self) -> dict[str, float]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
