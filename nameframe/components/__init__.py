"""
component layer.

minimum interface is defined for model, dataset, loss and metric.
all components communicate through `Batch` data object.
"""

from nameframe.components.base import from_config
from nameframe.components.batch import Batch
from nameframe.components.dataset import Dataset
from nameframe.components.loss import LossProtocol
from nameframe.components.metric import Metric, MetricProtocol
from nameframe.components.model import ModelProtocol

__all__ = [
    "Batch",
    "Dataset",
    "LossProtocol",
    "Metric",
    "MetricProtocol",
    "ModelProtocol",
    "from_config",
]
