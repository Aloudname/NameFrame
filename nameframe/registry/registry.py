"""Component registry system for pluggable model/dataset/loss/metric components."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T", bound=Callable[..., Any])


class Registry:
    """A named registry mapping string keys to callables or classes.

    Enables zero-code-change extensibility: new components are discovered
    automatically when their module is imported and the decorator runs.

    Attributes:
        name: Human-readable name for this registry (e.g. "model", "loss").
    """

    def __init__(self, name: str) -> None:
        """Initialize an empty registry.

        Args:
            name: Human-readable identifier for diagnostics.
        """
        self.name: str = name
        self._registry: Dict[str, Callable[..., Any]] = {}

    def register(self, name: Optional[str] = None) -> Callable[[T], T]:
        """Decorator to register a callable or class under a given name.

        Args:
            name: Key to register under. If None, uses ``cls_or_fn.__name__``.

        Returns:
            A decorator that registers the target and returns it unchanged.

        Example:
            >>> @MODEL_REGISTRY.register("my_model")
            ... class MyModel(BaseModel): ...
        """

        def decorator(obj: T) -> T:
            key: str = name if name is not None else obj.__name__
            if key in self._registry:
                raise KeyError(
                    f"'{key}' is already registered in {self.name} registry."
                )
            self._registry[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        """Retrieve a registered component by name.

        Args:
            name: The registered key.

        Returns:
            The registered callable or class.

        Raises:
            KeyError: If *name* is not found.
        """
        if name not in self._registry:
            available: str = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: [{available}]"
            )
        return self._registry[name]

    def get_or_build(self, name: str, **kwargs: Any) -> Any:
        """Get a registered component or attempt dynamic import.

        Args:
            name: The registered key or dotted import path.
            **kwargs: Forwarded to the callable on instantiation.

        Returns:
            The instantiated component.
        """
        if name in self._registry:
            return self._registry[name](**kwargs)
        # fallback: dynamic import
        parts: List[str] = name.split(".")
        module_path: str = ".".join(parts[:-1])
        attr_name: str = parts[-1]
        import importlib

        module = importlib.import_module(module_path)
        cls_or_fn = getattr(module, attr_name)
        return cls_or_fn(**kwargs)

    def list(self) -> List[str]:
        """Return sorted list of all registered names.

        Returns:
            Alphabetically sorted name list.
        """
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check whether *name* is registered."""
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, entries={len(self._registry)})"


# ---- global registries ----

MODEL_REGISTRY: Registry = Registry("model")
"""Global registry for model classes (subclasses of BaseModel)."""

DATASET_REGISTRY: Registry = Registry("dataset")
"""Global registry for dataset classes (subclasses of BaseDataset)."""

LOSS_REGISTRY: Registry = Registry("loss")
"""Global registry for loss classes (subclasses of BaseLoss)."""

METRIC_REGISTRY: Registry = Registry("metric")
"""Global registry for metric classes (subclasses of BaseMetric)."""

OPTIMIZER_REGISTRY: Registry = Registry("optimizer")
"""Global registry for custom optimizer builders."""

SCHEDULER_REGISTRY: Registry = Registry("scheduler")
"""Global registry for custom scheduler builders."""
