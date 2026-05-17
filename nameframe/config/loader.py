"""YAML configuration loading, merging, and dotted-key access utilities."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from munch import Munch, munchify


_ENV_VAR_RE: re.Pattern[str] = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: str) -> str:
    """Replace ``${VAR}`` placeholders in *value* with environment variables."""

    def _replacer(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")

    return _ENV_VAR_RE.sub(_replacer, value)


def _resolve_env_in_obj(obj: Any) -> Any:
    """Recursively resolve ``${VAR}`` placeholders in strings within *obj*."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_in_obj(item) for item in obj]
    return obj


def _to_munch(obj: Any) -> Munch:
    """Recursively convert nested dicts and lists to :class:`Munch` objects.

    Args:
        obj: A dict, list, or scalar value.

    Returns:
        A :class:`Munch` with attribute-accessible keys.
    """
    return munchify(obj)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *src* into *dst* in-place.

    Nested dicts are merged; non-dict values are overwritten by *src*.

    Args:
        dst: Destination dict (modified in-place).
        src: Source dict whose values take precedence.

    Returns:
        The merged dictionary (same object as *dst*).
    """
    for key, value in src.items():
        if key in dst and isinstance(dst[key], dict) and isinstance(value, dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _set_nested(container: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set *value* at a dotted-key path, creating intermediate dicts as needed.

    Args:
        container: Root dictionary to mutate.
        dotted_key: Dot-separated path (e.g. ``"train.lr"``).
        value: The value to assign.
    """
    parts: List[str] = dotted_key.split(".")
    current: Dict[str, Any] = container
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def load_config(path: Union[str, Path]) -> Munch:
    """Load a YAML configuration file and convert it to a :class:`Munch`.

    ``${ENV_VAR}`` placeholders in string values are replaced by the
    corresponding environment variables.

    Args:
        path: Path to a ``.yaml`` or ``.yml`` file.

    Returns:
        A :class:`Munch` with attribute-accessible configuration keys.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    resolved: Dict[str, Any] = _resolve_env_in_obj(raw)
    return _to_munch(resolved or {})


def merge_args(
    config: Munch,
    overrides: Optional[Union[Dict[str, Any], List[tuple]]] = None,
) -> Munch:
    """Merge key-value overrides into an existing config Munch.

    Supports dotted-key notation for nested access.

    Args:
        config: The base configuration (modified in-place).
        overrides: Either a flat dict of key-value pairs, or a list of
            ``(dotted_key, value)`` tuples (e.g. ``[("train.lr", 0.001)]``).

    Returns:
        The merged :class:`Munch` (same object as *config*).

    Example:
        >>> cfg = load_config("config.yaml")
        >>> merge_args(cfg, {"train.lr": 0.001})
        >>> merge_args(cfg, [("train.epochs", 200)])
    """
    if overrides is None:
        return config

    if isinstance(overrides, dict):
        overrides = list(overrides.items())

    config_dict: Dict[str, Any] = config.toDict()
    for key, value in overrides:
        _set_nested(config_dict, key, value)

    return _to_munch(config_dict)
