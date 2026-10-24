"""
core module registry for the framework.

the registry is the backbone of component system.
every swappable component type, e.g. model, dataset, loss and metric,
has its own registry instance, enabling name-based lookup,
namespace isolation and automatic module discovery.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, TypeVar

_Item = TypeVar("_Item", bound=Callable[..., Any])


def _item_source(item: Callable[..., Any]) -> str:
    """describe callable definitions without depending on an object repr."""
    module = getattr(item, "__module__", type(item).__module__)
    name = getattr(item, "__qualname__", type(item).__qualname__)
    return f"{module}.{name}"


class RegistryConflictError(ValueError):
    """raised when a registration would replace a different definition."""

    def __init__(
        self,
        key: str,
        registry_name: str,
        existing: Callable[..., Any],
        incoming: Callable[..., Any],
    ) -> None:
        self.key = key
        self.registry_name = registry_name
        self.existing = existing
        self.incoming = incoming
        super().__init__(
            f"'{key}' is already registered in {registry_name}: "
            f"{_item_source(existing)}; cannot replace it with "
            f"{_item_source(incoming)} without replace=True."
        )


class RegistryKeyError(KeyError):
    """
    raised when a requested key is not found in a registry.

    params:
    - `key`: `str` type, requested key.
    - `registry_name`: `str` type, name of the registry.
    - `available`: `list[str]` type, list of registered keys.
    """

    def __init__(self, key: str, registry_name: str, available: list[str]) -> None:
        self.key = key
        self.registry_name = registry_name
        self.available = available
        msg = f"'{key}' not found in {registry_name} registry. available: {available}"
        super().__init__(msg)


def _camel_to_snake(name: str) -> str:
    """
    converts a CamelCase string to snake_case.

    params:
    - `name`: `str` type, the CamelCase input.

    returns:
    - `str` type, the snake_case output.
    """

    # insert _ between capitals followed by a capital
    s1: str = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    # insert _ between a lowercase/digit and a capital,
    s2: str = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    # convert into lower case
    return s2.lower()


class Registry:
    """
    registry for discoverable, swappable components.

    each component category has a global registry instance,
    e.g. model, dataset, loss and metric,
    components are registered via decorator, function call or auto-discovery.

    params:
    - `name`: `str` type, name prepared for error messages.
    - `base_type`: optional `type`. all registered items must be subclasses of this type.
    - `strict`: reject replacement of different definitions unless replace=True;
      defaults to False for compatibility and is inherited by new namespaces.
    """

    def __init__(
        self, name: str, base_type: type | None = None, *, strict: bool = False
    ) -> None:
        if not isinstance(strict, bool):
            raise TypeError("registry strict must be a boolean.")
        self.name: str = name
        self.base_type: type | None = base_type
        self.strict = strict
        self._items: dict[str, Callable[..., Any]] = {}
        self._namespaces: dict[str, Registry] = {}

    # decorator registry
    def register(
        self, name: str | None = None, *, replace: bool | None = None
    ) -> Callable[[_Item], _Item]:
        """
        decorator that registers a class or callable under the given key.

        if `name` is `None`, auto convert class name from CamelCase to snake_case.
        usage:

            @model.register("my_vit")
            class MyViT(nn.Module): ...

            @model.register()       # registers as 'my_vit'
            class MyViT(nn.Module): ...

        params:
        - `name`: optional `str` type as registry key.
        - `replace`: explicit conflict policy; defaults to not `strict`.
          registering the same object again is always idempotent.

        returns:
        - decorator `callable` that registers decorated class.
        """

        def decorator(cls: _Item) -> _Item:
            if name is None and not hasattr(cls, "__name__"):
                raise ValueError(
                    "callable instances require an explicit registry name."
                )
            key: str = name if name is not None else _camel_to_snake(cls.__name__)
            self.register_external(key, cls, replace=replace)
            return cls

        return decorator

    def register_external(
        self,
        name: str,
        item: Callable[..., Any],
        *,
        replace: bool | None = None,
    ) -> None:
        """
        registers an externally-defined class or factory under the given key.

        functional calling is the second way of registry,
        used for importing components from third-party libraries.
        usage:

            model.register_external("resnet50", torchvision.models.resnet50)

        params:
        - `name`: `str` type as registry key.
        - `item`: class or callable factory to register.
        - `replace`: True explicitly allows replacement, False rejects conflicts.
          None preserves the registry's default policy (overwrite unless strict).
        """
        self._validate_type(name, item)
        if replace is not None and not isinstance(replace, bool):
            raise TypeError("registry replace must be a boolean or None.")
        allow_replace = not self.strict if replace is None else replace
        existing = self._items.get(name)
        if existing is not None and existing is not item and not allow_replace:
            raise RegistryConflictError(name, self.name, existing, item)
        self._items[name] = item

    def _validate_type(self, key: str, cls: Callable[..., Any]) -> None:
        """validates that cls is a subclass of base_type, if set."""
        if not isinstance(key, str) or not key or ":" in key:
            raise ValueError("registry keys must be nonempty strings without ':'.")
        if not callable(cls):
            raise TypeError(f"'{key}' must be a class or callable.")
        if self.base_type is not None and (
            not isinstance(cls, type) or not issubclass(cls, self.base_type)
        ):
            raise TypeError(
                f"'{key}' must be a subclass of {self.base_type.__name__}, "
                f"got {getattr(cls, '__name__', type(cls).__name__)}."
            )

    # lookup
    def get(self, name: str) -> Callable[..., Any]:
        """
        retrieves a registered class or callable by name.

        listing all available keys,
        supports namespace syntax `ns:name` for isolation.

        params:
        - `name`: `str` type as registry key, optionally prefixed with 'namespace:'.

        returns:
        - registered class or callable.

        raises:
        - `RegistryKeyError`: if key is not registered.
        """
        if ":" in name:
            ns, key = name.split(":", 1)
            if ns not in self._namespaces:
                raise RegistryKeyError(name, self.name, list(self._items.keys()))
            return self._namespaces[ns].get(key)

        if name not in self._items:
            raise RegistryKeyError(name, self.name, list(self._items.keys()))
        return self._items[name]

    def list_all(self, namespace: str | None = None) -> dict[str, Callable[..., Any]]:
        """
        returns all registered items as a key-to-type mapping.

        params:
        - `namespace`: optional `str` type, if provided, only returns
          items registered under that namespace.

        returns:
        - `dict[str, type]` type, mapping registration keys to types.
        """
        if namespace is not None:
            if namespace not in self._namespaces:
                return {}
            return dict(self._namespaces[namespace]._items)
        return dict(self._items)

    def conflicting_keys(self, namespace: str) -> list[str]:
        """
        check if registered components conflict with given namespace.

        params:
        - `namespace`: `str` type.

        returns:
        - `List[str]` of repeated components.
        """
        if namespace not in self._namespaces:
            return []
        ns_items = self._namespaces[namespace]._items
        return [k for k in self._items if k in ns_items]

    # namespace
    def create_namespace(self, name: str) -> Registry:
        """
        creates or retrieves a sub-namespace within *this* registry.

        namespaces isolate different projects' registry.
        model registered as 'proj_a:resnet' does not collide with 'proj_b:resnet'.

        params:
        - `name`: `str` type as namespace id.

        returns:
        - `Registry` instance scoped to the namespace.
        """
        if not isinstance(name, str) or not name or ":" in name:
            raise ValueError("namespace names must be nonempty strings without ':'.")
        if name not in self._namespaces:
            self._namespaces[name] = Registry(
                f"{self.name}:{name}", self.base_type, strict=self.strict
            )
        return self._namespaces[name]

    # dunder
    def __contains__(self, name: str) -> bool:
        """checks whether a key is registered. supports `ns:name`."""
        if ":" in name:
            ns, key = name.split(":", 1)
            return ns in self._namespaces and key in self._namespaces[ns]
        return name in self._items

    def __len__(self) -> int:
        """number of registered items."""
        return len(self._items)

    def __repr__(self) -> str:
        return (
            f"Registry(name='{self.name}', "
            f"items={len(self._items)}, "
            f"namespaces={list(self._namespaces.keys())})"
        )
