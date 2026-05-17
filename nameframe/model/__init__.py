"""Model factory, instantiates models from the registry or dynamic import."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn
from munch import Munch

from nameframe.registry import MODEL_REGISTRY


def build_model(config: Munch) -> nn.Module:
    """Instantiate a model from configuration.

    Looks up ``config.model.name`` in :data:`MODEL_REGISTRY`. If not found,
    attempts a dynamic import using the name as a dotted path.

    Args:
        config: The full project configuration (Munch).

    Returns:
        An instance of a :class:`BaseModel` subclass.

    Raises:
        ValueError: If ``model.name`` is not set in the config.
        KeyError: If the name is not found in the registry or as an import path.

    Example:
        >>> cfg = load_config("config.yaml")
        >>> model = build_model(cfg)
    """
    name: str = config.model.get("name") or config.model.get("family")
    if name is None:
        raise ValueError(
            "config.model.name (or config.model.family) must be set to "
            "a registered model name or dotted import path."
        )

    model_subtree: Dict[str, Any] = config.model.toDict() if isinstance(config.model, Munch) else dict(config.model)
    model: nn.Module = MODEL_REGISTRY.get_or_build(name, config=model_subtree)
    return model
