"""Component registry system, bottom layer."""

from nameframe.registry.registry import (
    DATASET_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    Registry,
)

__all__ = [
    "Registry",
    "MODEL_REGISTRY",
    "DATASET_REGISTRY",
    "LOSS_REGISTRY",
    "METRIC_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "SCHEDULER_REGISTRY",
]
