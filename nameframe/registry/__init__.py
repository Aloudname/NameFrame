"""
global registry instances for the framework.

each component category has a singleton registry instance.
all other `nameframe` modules import from here to register or look up components.
"""

from nameframe.registry.core import Registry, RegistryConflictError, RegistryKeyError
from nameframe.registry.discovery import (
    DiscoveryError,
    DiscoveryFailure,
    DiscoveryReport,
    discover,
)

model: Registry = Registry("model")
"""registry for model components."""

dataset: Registry = Registry("dataset")
"""registry for dataset components."""

loss: Registry = Registry("loss")
"""registry for loss components."""

metric: Registry = Registry("metric")
"""registry for metric components."""

__all__ = [
    "DiscoveryError",
    "DiscoveryFailure",
    "DiscoveryReport",
    "Registry",
    "RegistryConflictError",
    "RegistryKeyError",
    "dataset",
    "discover",
    "loss",
    "metric",
    "model",
]
