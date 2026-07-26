"""
config loader for the nameframe framework.

parses yaml config files with include support, merges cli overrides,
constructs an attribute-accessible tree of _ConfigNode objects, and
provides namespace extraction for component instantiation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class _ConfigNode:
    """
    single node in the config tree.

    wraps dict values as attribute-accessible properties.
    recursively convert nested `dicts` to `_ConfigNode`.

    if `_frozen` = `True`, `__setattr__` -> `FrozenConfigError`.

    params:
    - `**kwargs`: `dict` to store as attributes.
    """

    def __init__(self, **kwargs: Any) -> None:
        # write directly to __dict__
        self.__dict__["_frozen"] = False
        for key, value in kwargs.items():
            # recursively convert
            if isinstance(value, dict):
                value = _ConfigNode(**value)
            elif isinstance(value, list):
                value = [_ConfigNode(**v) if isinstance(v, dict) else v for v in value]
            # until store root value
            self.__dict__[key] = value

    def __getattr__(self, name: str) -> Any:
        if name not in self.__dict__:
            raise AttributeError(f"config has no key '{name}'.")
        return self.__dict__[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_frozen", False):
            from nameframe.config.freezer import FrozenConfigError, _get_caller_location
            raise FrozenConfigError(name, _get_caller_location())
        self.__dict__[name] = value

    def __repr__(self) -> str:
        items = {k: v for k, v in self.__dict__.items() if k != "_frozen"}
        return f"_ConfigNode({items})"


class Config:
    """
    config tree with attribute access.

    usage:

        config = Config.from_yaml("exp.yaml")
        config.resolve()
        config.freeze()

    after freezing, any attempt to modify a value raises `FrozenConfigError`.
    frozen config can be dicted via `to_dict()` and stored in checkpoints.

    params:
    - `data`: `dict` type, raw config dict.
    - `source_path`: optional `Path` type, `.yaml` file path.
    """

    def __init__(self, data: dict, source_path: Path | None = None) -> None:
        self._data: _ConfigNode = _ConfigNode(**data)
        self._source_path: Path | None = source_path
        self._frozen: bool = False

    # construction
    @classmethod
    def from_yaml(cls, path: Path, overrides: dict | None = None) -> Config:
        """
        loads `.yaml` config file, returns `Config` instance.

        processes `!include` directives within `.yaml` file,
        merge cli overrides after loading & before freezing.

        params:
        - `path`: `Path` type to `.yaml` config file.
        - `overrides`: optional `dict` type, cli-specified overrides.

        returns:
        - `Config` instance with loaded and merged params.

        raises:
        - `FileNotFoundError`: if file not found.
        - `yaml.YAMLError`: if file structure not valid.
        """
        if not path.exists():
            raise FileNotFoundError(f"config file not found: '{path}'")
        raw: dict = _load_yaml_with_includes(path)
        if overrides:
            raw = _deep_merge(raw, overrides)
        return cls(raw, source_path=path)

    # lifecycle
    def resolve(self) -> None:
        """
        resolves variable syntax in config values, e.g. `${search: ...}` placeholders.

        NOTE: **call this after `from_yaml()` and before `freeze()`**.
        """
        from nameframe.config.resolver import _resolve_search_vars
        _resolve_search_vars(self._data)

    def freeze(self) -> None:
        """
        freezes config tree. after this call,
        any `__setattr__` call raises `FrozenConfigError`.

        NOTE: **call this after `resolve()`**.
        """
        self._frozen = True
        self._freeze_node(self._data)

    @property
    def is_frozen(self) -> bool:
        """if `Config` instance frozen."""
        return self._frozen

    # access
    def get_namespace(self, ns: str) -> dict:
        """
        extracts sub namespace as `dict`.

        param namespace **must be a dot path**, e.g. `training.optimizer`.
        returns plain `dict` as `**kwargs` passing to a component constructor.

        params:
        - `ns`: `str` type, dot namespace path.

        returns:
        - `dict` type, the extracted key-value pairs.
        """
        node: _ConfigNode = self._data
        for part in ns.split("."):
            node = getattr(node, part)
        return _config_node_to_dict(node)

    def to_dict(self) -> dict:
        """
        exports config tree as plain `dict`.

        returns:
        - `dict` type, copy of config for checkpoint snapshots.
        """
        return _config_node_to_dict(self._data)

    # attribute access
    def __getattr__(self, name: str) -> Any:
        # _ for internal attributes
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._data, name)

    def __repr__(self) -> str:
        source = self._source_path or "memory"
        status = "frozen" if self._frozen else "mutable"
        return f"Config(source='{source}', status='{status}')"

    # internal helpers
    @staticmethod
    def _freeze_node(node: _ConfigNode) -> None:
        """recursively freezes a `_ConfigNode` and all its children."""
        node.__dict__["_frozen"] = True
        for _key, value in node.__dict__.items():
            if isinstance(value, _ConfigNode):
                Config._freeze_node(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, _ConfigNode):
                        Config._freeze_node(item)


# yaml loading
class _IncludeLoader(yaml.SafeLoader):
    """custom yaml loader that supports `!include` directives."""

    name: str | None = None  # path to current yaml file
    _visited_includes: set[str] = set()  # track visited include paths
    pass

def _include_constructor(loader: _IncludeLoader, node: yaml.Node) -> Any:
    """
    handles `!include path/to/file.yaml` in yaml configs.

    params:
    - `loader`: `yaml.Loader` type.
    - `node`: `yaml.Node` type of yaml node.

    returns:
    - `Any` type of parsed included content.
    """

    # check node type
    if not isinstance(node, (yaml.ScalarNode, yaml.MappingNode)):
        raise TypeError(
            f"`!include` expects a scalar value, got {type(node).__name__}"
        )

    include_path_str: str = loader.construct_scalar(node)
    _source = loader.name
    current_file = Path(_source) if _source else Path.cwd()
    include_path = (current_file.parent / include_path_str).resolve()

    if not include_path.exists():
        raise FileNotFoundError(
            f"included config file not found: '{include_path}' "
            f"referenced from '{current_file}'"
        )

    # track visited paths
    visited: set[str] = loader._visited_includes
    path_key = str(include_path)
    if path_key in visited:
        raise ValueError(f"circular !include detected: '{include_path}'")
    visited.add(path_key)
    loader._visited_includes = visited

    return _load_yaml_with_includes(include_path, visited)


_IncludeLoader.add_constructor("!include", _include_constructor)


def _load_yaml_with_includes(path: Path, _visited: set[str] | None = None) -> dict:
    """
    loads `.yaml` file and processes `!include` directives.

    params:
    - `path`: `Path` type to `.yaml` file.
    - `_visited`: optional `set[str]` for tracking visited paths.

    returns:
    - `dict` type of parsed config.
    """
    loader = _IncludeLoader(path.read_bytes())
    loader.name = str(path.resolve())
    loader._visited_includes = _visited or set()
    loader._visited_includes.add(str(path.resolve()))
    return loader.get_single_data() or {}


# dict utilities
def _deep_merge(base: dict, override: dict) -> dict:
    """
    recursively merges two `dict`.

    params:
    - `base`: `dict` type of original dict.
    - `override`: `dict` type of the overrides to apply.

    returns:
    - `dict` type of merged result.
    """
    result: dict = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _config_node_to_dict(node: _ConfigNode) -> dict:
    """
    converts `_ConfigNode` tree back to `dict`.

    params:
    - `node`: `_ConfigNode` type.

    returns:
    - `dict` type.
    """
    result: dict = {}
    for key, value in node.__dict__.items():
        # skip internal `_frozen`
        if key == "_frozen":
            continue
        # recursively convert
        if isinstance(value, _ConfigNode):
            result[key] = _config_node_to_dict(value)
        elif isinstance(value, list):
            result[key] = [
                _config_node_to_dict(v) if isinstance(v, _ConfigNode) else v
                for v in value
            ]
        else:
            # store root value
            result[key] = value
    return result
