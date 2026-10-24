"""YAML loading, detached configuration snapshots and validation."""

from nameframe.config.freezer import FrozenConfigError
from nameframe.config.loader import Config
from nameframe.config.schema import (
    ConfigConflictError,
    validate_schema,
    validate_schema_definition,
)

__all__ = [
    "Config",
    "ConfigConflictError",
    "FrozenConfigError",
    "validate_schema",
    "validate_schema_definition",
]
