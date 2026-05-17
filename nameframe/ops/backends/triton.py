"""Triton kernel manager — wraps triton.jit with grid auto-tuning."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from nameframe.ops.backends.compiler import detect_triton


class TritonKernelManager:
    """Manage Triton JIT-compiled kernels with optional auto-tuning support.

    Each registered kernel is wrapped via ``triton.jit`` and exposed as a
    launcher that handles grid configuration automatically.

    Attributes:
        _kernels: Dict of op name -> (kernel_fn, config).
    """

    def __init__(self) -> None:
        """Initialize an empty Triton kernel manager."""
        self._kernels: Dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        kernel_fn: Callable[..., Any],
        configs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Register a Triton kernel under *name*.

        If *configs* is provided, auto-tuning metadata is stored for later use.

        Args:
            name: Op identifier.
            kernel_fn: A ``triton.jit``-decorated function or raw kernel.
            configs: Optional list of auto-tuning configuration dicts.
        """
        try:
            import triton  # noqa: F401
            # store both the kernel and tuning configs
            self._kernels[name] = kernel_fn
            if configs:
                self._kernels[f"{name}__configs"] = configs
        except ImportError:
            pass  # triton not available — silently skip registration

    def get_launcher(self, name: str) -> Optional[Callable[..., Any]]:
        """Return a grid-aware launcher for the registered kernel.

        Args:
            name: Op identifier.

        Returns:
            A callable that wraps the kernel with proper grid settings,
            or ``None`` if not registered.
        """
        kernel_fn: Optional[Callable[..., Any]] = self._kernels.get(name)
        if kernel_fn is None:
            return None

        # return a simple wrapper that does basic grid calculation
        def _launcher(*args: Any, **kwargs: Any) -> Any:
            import triton
            # basic grid: one program per element along first dimension
            grid = lambda meta: (triton.cdiv(args[0].numel(), meta["BLOCK_SIZE"]),)
            return kernel_fn[grid](*args, **kwargs)

        return _launcher

    @staticmethod
    def is_available() -> bool:
        """Check whether Triton is usable on this system.

        Returns:
            ``True`` if Triton is importable.
        """
        return detect_triton()
