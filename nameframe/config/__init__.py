"""
config system.

provides config objects with yaml loading, schema validation,
namespace extraction, and variable resolution.
the three-step pattern from_yaml() to resolve() to freeze(),
is the dataflow entry point for all pipelines.
"""

from nameframe.config.freezer import FrozenConfigError
from nameframe.config.loader import Config
from nameframe.config.schema import ConfigConflictError, validate_schema

__all__ = [
    "Config",
    "ConfigConflictError",
    "FrozenConfigError",
    "validate_schema",
]
