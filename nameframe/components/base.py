"""
construct component instances with config.

provides a factory function `from_config()`,
construct component instances from a *frozen* Config.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, TypeVar

from nameframe.config import (
    Config,
    ConfigConflictError,
    validate_schema,
    validate_schema_definition,
)

_Component = TypeVar("_Component")


def from_config(
    cls: Callable[..., _Component], config: Config, namespace: str
) -> _Component:
    """
    construct component instances from a *frozen* Config.

    supports classes and callable factories, schema defaults are copied before
    validation. absent schemas permit external constructor parameters.

    params:
    - `cls`: class or callable factory.
    - `config`: `Config` type of config instance.
    - `namespace`: `str` type, dot separated namespace path.

    returns:
    - instance of `cls`.
    """
    if not config.is_frozen:
        raise RuntimeError("config must be frozen before constructing components.")
    if not callable(cls):
        raise TypeError(f"component factory at '{namespace}' must be callable.")
    params: dict = config.get_namespace(namespace)

    # `name` is the registry key used to look up the class,
    # it belongs to the config namespace rather than __init__
    params.pop("name", None)

    # discover schema via structural subtyping.

    raw_schema = getattr(cls, "config_schema", None)
    if raw_schema is not None:
        try:
            validate_schema_definition(raw_schema)
            schema = {k: v for k, v in raw_schema.items() if k != "name"}
            _check_concrete(params, namespace)
            validate_schema(params, schema)
        except ConfigConflictError as exc:
            raise ConfigConflictError(
                [f"{namespace}: {error}" for error in exc.errors]
            ) from exc
        for key, field in schema.items():
            if key not in params and "default" in field:
                params[key] = copy.deepcopy(field["default"])
        _check_concrete(params, namespace)
    else:
        _check_concrete(params, namespace)
    return cls(**params)


def _check_concrete(value: Any, path: str) -> None:
    """reject unresolved search spaces even for components without a schema."""
    if isinstance(value, str) and value.startswith("${search:"):
        raise ConfigConflictError(
            [
                f"search expressions are unsupported at '{path}', provide a concrete value."
            ]
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _check_concrete(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_concrete(item, f"{path}[{index}]")
