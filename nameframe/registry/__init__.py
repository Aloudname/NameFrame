"""
global registry instances for the framework.

each component category has a singleton registry instance.
all other `nameframe` modules import from here to register or look up components.
"""

from nameframe.registry.core import Registry, RegistryKeyError
from nameframe.registry.discovery import discover

model: Registry = Registry("model")
"""registry for model components."""

dataset: Registry = Registry("dataset")
"""registry for dataset components."""

loss: Registry = Registry("loss")
"""registry for loss components."""

metric: Registry = Registry("metric")
"""registry for metric components."""

ops: Registry = Registry("ops")
"""registry for custom operator components."""

plugin: Registry = Registry("plugin")
"""registry for plugin components."""

__all__ = [
    "Registry",
    "RegistryKeyError",
    "dataset",
    "discover",
    "loss",
    "metric",
    "model",
    "ops",
    "plugin",
]