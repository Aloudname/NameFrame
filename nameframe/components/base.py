"""
construct component instances with config.

provides a factory function `from_config()`,
construct component instances from a *frozen* Config.
"""

from __future__ import annotations

from typing import Any

from nameframe.config import Config, validate_schema


def from_config(cls: type, config: Config, namespace: str) -> Any:
    """
    construct component instances from a *frozen* Config.

    works with any class with `config_schema`.

    params:
    - `cls`: `type` of instance.
    - `config`: `Config` type of config instance.
    - `namespace`: `str` type, dot separated namespace path.

    returns:
    - instance of `cls`.
    """
    if not config.is_frozen:
        raise RuntimeError("config must be frozen before constructing components.")
    params: dict = config.get_namespace(namespace)

    # `name` is the registry key used to look up the class,
    # it belongs to the config namespace rather than __init__
    params.pop("name", None)

    # discover schema via structural subtyping.

    raw_schema: dict = getattr(cls, "config_schema", {})
    schema = {k: v for k, v in raw_schema.items() if k != "name"}
    if schema:
        validate_schema(params, schema)
    return cls(**params)
