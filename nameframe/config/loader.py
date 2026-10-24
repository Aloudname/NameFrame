"""yaml loading, recursive configuration containers, and immutable snapshots."""

from __future__ import annotations

import copy
import operator
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

from nameframe.config.freezer import FrozenConfigError, _get_caller_location


def _frozen_error(path: str) -> FrozenConfigError:
    return FrozenConfigError(path, _get_caller_location())


class _ConfigList(list):
    """list that wraps nested values and rejects mutations after freezing."""

    def __init__(
        self, values: Iterable[Any], path: str, ancestors: frozenset[int] = frozenset()
    ) -> None:
        ancestors = _enter_container(values, path, ancestors)
        self._path = path
        self._frozen = False
        super().__init__(
            _wrap(value, f"{path}[{i}]", ancestors) for i, value in enumerate(values)
        )

    def _reindex(self) -> None:
        for index, value in enumerate(self):
            _set_path(value, f"{self._path}[{index}]")

    def _check_mutable(self) -> None:
        if self._frozen:
            raise _frozen_error(self._path)

    def __setitem__(self, index: Any, value: Any) -> None:
        self._check_mutable()
        if isinstance(index, slice):
            value = [_wrap(item, self._path) for item in value]
        else:
            if self[index] is value:
                return
            value = _wrap(value, f"{self._path}[{index}]")
        super().__setitem__(index, value)
        self._reindex()

    def __delitem__(self, index: Any) -> None:
        self._check_mutable()
        super().__delitem__(index)
        self._reindex()

    def append(self, value: Any) -> None:
        self._check_mutable()
        super().append(_wrap(value, f"{self._path}[{len(self)}]"))

    def extend(self, values: Iterable[Any]) -> None:
        self._check_mutable()
        # Validate before mutation, including when values is self.
        wrapped = [
            _wrap(value, f"{self._path}[{len(self) + index}]")
            for index, value in enumerate(values)
        ]
        super().extend(wrapped)

    def insert(self, index: int, value: Any) -> None:
        self._check_mutable()
        super().insert(index, _wrap(value, f"{self._path}[{index}]"))
        self._reindex()

    def pop(self, index: int = -1) -> Any:
        self._check_mutable()
        value = super().pop(index)
        self._reindex()
        return value

    def remove(self, value: Any) -> None:
        self._check_mutable()
        super().remove(value)
        self._reindex()

    def clear(self) -> None:
        self._check_mutable()
        super().clear()

    def reverse(self) -> None:
        self._check_mutable()
        super().reverse()
        self._reindex()

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        self._check_mutable()
        try:
            super().sort(key=key, reverse=reverse)
        finally:
            # A failed comparison can still leave a partially reordered list.
            self._reindex()

    def __iadd__(self, values: Iterable[Any]) -> _ConfigList:
        self.extend(values)
        return self

    def __imul__(self, count: int) -> _ConfigList:
        self._check_mutable()
        count = operator.index(count)
        if count <= 0:
            super().clear()
        elif count > 1:
            # Repeated containers need independent values and unambiguous paths.
            originals = list(self)
            self.extend(value for _ in range(count - 1) for value in originals)
        return self


class _ConfigNode:
    """attribute and key access to a recursively wrapped configuration mapping."""

    def __init__(
        self,
        data: Mapping[str, Any],
        path: str = "",
        ancestors: frozenset[int] = frozenset(),
    ) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("config must be a mapping with string keys.")
        ancestors = _enter_container(data, path, ancestors)
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_frozen", False)
        object.__setattr__(self, "_values", {})
        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError(f"config keys must be strings at '{path or '<root>'}'.")
            self._values[key] = _wrap(value, self._key_path(key), ancestors)

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        if name not in values:
            raise AttributeError(f"config has no key '{name}'.")
        return values[name]

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __setattr__(self, name: str, value: Any) -> None:
        if self._frozen:
            raise _frozen_error(self._key_path(name))
        if name.startswith("_") or hasattr(type(self), name):
            raise AttributeError(f"reserved config attribute '{name}', use key access.")
        self[name] = value

    def __setitem__(self, key: str, value: Any) -> None:
        if self._frozen:
            raise _frozen_error(self._key_path(key))
        if not isinstance(key, str):
            raise TypeError("config keys must be strings.")
        if key in self._values and self._values[key] is value:
            # Python assigns the result of += and *= back to the parent.
            return
        self._values[key] = _wrap(value, self._key_path(key))

    def __delattr__(self, name: str) -> None:
        if self._frozen:
            raise _frozen_error(self._key_path(name))
        if name.startswith("_") or hasattr(type(self), name):
            raise AttributeError(f"reserved config attribute '{name}'.")
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __delitem__(self, key: str) -> None:
        if self._frozen:
            raise _frozen_error(self._key_path(key))
        del self._values[key]

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def _key_path(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

    def __repr__(self) -> str:
        return f"_ConfigNode({self._values!r})"


def _enter_container(
    value: Any, path: str, ancestors: frozenset[int]
) -> frozenset[int]:
    if id(value) in ancestors:
        raise ValueError(f"cyclic config container at '{path or '<root>'}'.")
    return ancestors | {id(value)}


def _set_path(value: Any, path: str) -> None:
    """update diagnostic paths while preserving references to existing nodes."""
    if isinstance(value, _ConfigNode):
        object.__setattr__(value, "_path", path)
        for key, item in value._values.items():
            _set_path(item, value._key_path(key))
    elif isinstance(value, _ConfigList):
        value._path = path
        value._reindex()


def _wrap(value: Any, path: str, ancestors: frozenset[int] = frozenset()) -> Any:
    """detach external values, including mappings nested inside sequences."""
    if isinstance(value, _ConfigNode):
        value = _plain(value)
    if isinstance(value, Mapping):
        return _ConfigNode(value, path, ancestors)
    if isinstance(value, (list, tuple)):
        return _ConfigList(value, path, ancestors)
    # yaml sets and arbitrary mutable objects cannot satisfy the freeze contract.
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported config value at '{path}': {type(value).__name__}.")


def _plain(value: Any) -> Any:
    """export detached builtin containers."""
    if isinstance(value, _ConfigNode):
        return {key: _plain(item) for key, item in value._values.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return copy.deepcopy(value)


def _freeze(value: Any) -> None:
    if isinstance(value, _ConfigNode):
        for item in value._values.values():
            _freeze(item)
        object.__setattr__(value, "_frozen", True)
    elif isinstance(value, _ConfigList):
        for item in value:
            _freeze(item)
        value._frozen = True


class Config(_ConfigNode):
    """config tree with load → freeze lifecycle and detached exports.

    accepts mappings containing string keys, scalar values, and nested lists.
    public attribute, key, and list mutations raise FrozenConfigError once frozen.
    methods return self so both sequential and chained lifecycle calls work.
    """

    def __init__(self, data: dict, source_path: Path | None = None) -> None:
        super().__init__(data)
        object.__setattr__(self, "_source_path", source_path)

    @classmethod
    def from_yaml(cls, path: Path | str, overrides: dict | None = None) -> Config:
        """load a mapping, resolve relative includes, then deep-merge overrides.

        repeated includes are allowed, cycles along the active include chain fail.
        empty documents produce an empty config, non-mapping roots are rejected.
        """
        path = Path(path).resolve()
        raw = _load_yaml_with_includes(path)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError(f"config root must be a mapping: '{path}'.")
        try:
            config = cls(raw, source_path=path)
            if overrides is not None:
                if not isinstance(overrides, dict):
                    raise TypeError("config overrides must be a mapping.")
                # Validate both trees before recursive merging can encounter cycles.
                concrete_overrides = _plain(_wrap(overrides, "overrides"))
                config = cls(
                    _deep_merge(config.to_dict(), concrete_overrides), source_path=path
                )
            return config
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{exc} (config '{path}')") from exc

    def freeze(self) -> Config:
        """recursively freeze the existing tree, including held list references."""
        _freeze(self)
        return self

    @property
    def is_frozen(self) -> bool:
        """whether the config is read-only."""
        return self._frozen

    def get_namespace(self, ns: str) -> dict:
        """return a detached mapping at a dot path, reject scalar namespaces."""
        node: Any = self
        for part in ns.split("."):
            if not isinstance(node, _ConfigNode) or part not in node:
                raise AttributeError(f"config has no namespace '{ns}'.")
            node = node[part]
        if not isinstance(node, _ConfigNode):
            raise TypeError(f"config namespace '{ns}' must be a mapping.")
        return _plain(node)

    def to_dict(self) -> dict:
        """export detached plain containers for snapshots or constructor kwargs."""
        return _plain(self)

    def __repr__(self) -> str:
        source = self._source_path or "memory"
        status = "frozen" if self.is_frozen else "mutable"
        return f"Config(source='{source}', status='{status}')"


class _IncludeLoader(yaml.SafeLoader):
    """safe yaml loader with an instance-local active include chain."""

    source_path: Path
    include_chain: tuple[Path, ...]

    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self._checked_mappings: set[int] = set()
        self._flattening_mappings: set[int] = set()

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        if id(node) in self._flattening_mappings:
            raise yaml.constructor.ConstructorError(
                "while resolving YAML merge keys",
                node.start_mark,
                "cyclic YAML merge alias",
                node.start_mark,
            )
        # Check original pairs before PyYAML inserts inherited merge-key pairs.
        # Aliased mappings may already have been flattened by another mapping.
        if id(node) not in self._checked_mappings:
            self._checked_mappings.add(id(node))
            seen: dict[Any, yaml.Node] = {}
            merge_key = object()
            for key_node, _ in node.value:
                key = (
                    merge_key
                    if key_node.tag == "tag:yaml.org,2002:merge"
                    else self.construct_object(key_node)
                )
                try:
                    if key in seen:
                        raise yaml.constructor.ConstructorError(
                            "while constructing a mapping",
                            seen[key].start_mark,
                            f"duplicate explicit key {'<<' if key is merge_key else key!r}",
                            key_node.start_mark,
                        )
                    seen[key] = key_node
                except TypeError as exc:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        "found unhashable key",
                        key_node.start_mark,
                    ) from exc
        self._flattening_mappings.add(id(node))
        try:
            super().flatten_mapping(node)
        finally:
            self._flattening_mappings.remove(id(node))


def _include_constructor(loader: _IncludeLoader, node: yaml.Node) -> Any:
    """load a scalar include path relative to the file containing it."""
    if not isinstance(node, yaml.ScalarNode):
        raise TypeError(f"`!include` expects a scalar path at {node.start_mark}.")
    include_path = loader.source_path.parent / loader.construct_scalar(node)
    return _load_yaml_with_includes(include_path, loader.include_chain)


_IncludeLoader.add_constructor("!include", _include_constructor)


def _load_yaml_with_includes(path: Path, chain: tuple[Path, ...] = ()) -> Any:
    """load any included yaml value, reject active recursion, dispose the loader."""
    path = path.resolve()
    if path in chain:
        raise ValueError(
            "circular !include detected: " + " -> ".join(map(str, (*chain, path)))
        )
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        detail = " -> ".join(map(str, (*chain, path)))
        raise type(exc)(
            exc.errno, f"{exc.strerror}; config include chain: {detail}", exc.filename
        ) from exc
    with stream:
        loader = _IncludeLoader(stream)
        loader.source_path = path
        loader.include_chain = (*chain, path)
        try:
            return loader.get_single_data()
        except yaml.MarkedYAMLError as exc:
            if not getattr(exc, "_nameframe_include_context", False):
                detail = " -> ".join(map(str, loader.include_chain))
                exc.context = (
                    f"{exc.context or 'invalid YAML'}; include chain: {detail}"
                )
                exc._nameframe_include_context = True
            raise
        finally:
            loader.dispose()


def _deep_merge(base: dict, override: dict) -> dict:
    """recursively merge mappings without retaining references to either input."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
