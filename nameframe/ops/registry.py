"""Operator registry — maps op names to Python fallbacks and native backends."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from nameframe.registry.registry import Registry

OP_REGISTRY: Registry = Registry("op")
"""Global registry: op name -> Python fallback function."""

NATIVE_REGISTRY: Dict[str, Dict[str, Callable[..., Any]]] = {}
"""Maps op name -> {backend_name: native_callable}.
Backend names: "cuda", "triton", "cython", "cpp".
"""


def register_op(name: str, fn: Callable[..., Any]) -> None:
    """Store a Python function as the fallback implementation for *name*.

    Args:
        name: Op identifier.
        fn: The pure-Python function (the fallback).
    """
    OP_REGISTRY._registry[name] = fn  # bypass register() to allow re-registration


def register_native_fn(
    name: str, backend: str, fn: Callable[..., Any]
) -> None:
    """Store a native implementation for a named op.

    Args:
        name: Op identifier.
        backend: Backend name (``"cuda"``, ``"triton"``, etc.).
        fn: The native callable.
    """
    NATIVE_REGISTRY.setdefault(name, {})[backend] = fn


def get_native(name: str, backend: str) -> Optional[Callable[..., Any]]:
    """Retrieve a specific native implementation, or ``None``.

    Args:
        name: Op identifier.
        backend: Backend name.

    Returns:
        The native callable if registered, else ``None``.
    """
    return NATIVE_REGISTRY.get(name, {}).get(backend)


def list_backends(name: str) -> List[str]:
    """List all registered backends for an op.

    Args:
        name: Op identifier.

    Returns:
        Sorted list of backend name strings.
    """
    return sorted(NATIVE_REGISTRY.get(name, {}).keys())


def resolve(
    name: str, priority: List[str]
) -> Callable[..., Any]:
    """Resolve an op name to the best available implementation.

    Returns the first available backend in *priority* order, falling back
    to the Python function in :data:`OP_REGISTRY`.

    Args:
        name: Op identifier.
        priority: Ordered list of backend names to try.

    Returns:
        The resolved callable.

    Raises:
        KeyError: If *name* is not registered in :data:`OP_REGISTRY`.
    """
    for backend in priority:
        native_fn: Optional[Callable[..., Any]] = get_native(name, backend)
        if native_fn is not None:
            return native_fn

    if name in OP_REGISTRY:
        return OP_REGISTRY.get(name)

    raise KeyError(
        f"Op '{name}' not found in OP_REGISTRY or NATIVE_REGISTRY."
    )
