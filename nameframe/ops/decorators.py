"""Core decorator API for transparent acceleration dispatch.

The ``@accelerated`` decorator is the primary entry point. It transforms
a pure-Python function into one that auto-dispatches to the fastest
available native backend (Triton > CUDA > C++/Cython > Python fallback).
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar

from munch import Munch

from nameframe.ops.backends import (
    TritonKernelManager,
    detect_cython,
    detect_triton,
)
from nameframe.ops.backends.cuda import CUDAExtensionBuilder
from nameframe.ops.registry import (
    NATIVE_REGISTRY,
    OP_REGISTRY,
    register_native_fn,
    register_op,
)

F = TypeVar("F", bound=Callable[..., Any])

_triton_manager: TritonKernelManager = TritonKernelManager()
"""Module-level Triton kernel registry."""

# default backend priority (configurable via config or decorator params)
_DEFAULT_PRIORITY: List[str] = ["triton", "cuda", "cython", "python"]


def accelerated(
    name: str,
    backends: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator that makes a function auto-dispatch to the fastest backend.

    The decorated function's body serves as the pure-Python fallback.
    On first call, the decorator checks available backends in priority
    order and replaces itself with the best implementation.

    Args:
        name: Unique op identifier.
        backends: Override backend priority order. Default: ``["triton",
            "cuda", "cython", "python"]``.
        config: Backend-specific configuration dict.

    Returns:
        A wrapped function that transparently dispatches to native code
        when available, falling back to the decorated Python body.

    Example:
        >>> @accelerated("fused_gelu_dropout")
        ... def fused_gelu_dropout(x, p=0.5, training=True):
        ...     x = F.gelu(x)
        ...     if training:
        ...         x = F.dropout(x, p=p, training=training)
        ...     return x
    """
    priority: List[str] = backends or _DEFAULT_PRIORITY

    def decorator(fn: F) -> F:
        register_op(name, fn)
        _resolved: Optional[Callable[..., Any]] = None

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal _resolved
            if _resolved is not None:
                # already resolved -> fast path
                return _resolved(*args, **kwargs)

            # try backends in priority order
            for backend in priority:
                native_fn: Optional[Callable[..., Any]] = (
                    NATIVE_REGISTRY.get(name, {}).get(backend)
                )
                if native_fn is not None:
                    # verify on first use if configured
                    if config and config.get("verify_on_first_call", False):
                        _verify_op(name, fn, native_fn, config.get("verify_tolerance", 1e-4))
                    _resolved = native_fn
                    return _resolved(*args, **kwargs)

            # python fallback (always available)
            _resolved = fn
            return _resolved(*args, **kwargs)

        # expose metadata
        wrapper._op_name = name  # type: ignore[attr-defined]
        wrapper._op_fallback = fn  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def cuda_kernel(
    name: str,
    sources: List[str],
    headers: Optional[List[str]] = None,
    build_dir: str = "/tmp/nameframe_ops",
) -> Callable[[F], F]:
    """Decorator that binds a CUDA kernel (from .cu source files) to *name*.

    The decorated function body is ignored at runtime (it serves as
    documentation). The CUDA kernel is JIT-compiled on first use.

    Args:
        name: Op identifier.
        sources: List of ``.cu`` source file paths.
        headers: Optional header file paths.
        build_dir: JIT compilation working directory.

    Returns:
        A decorator that registers the CUDA native function.

    Example:
        >>> @cuda_kernel("fused_gelu_dropout", sources=["ops/fused.cu"])
        ... def _native_fused_gelu_dropout():
        ...     pass  # body is replaced by the compiled CUDA function
    """
    _builder: CUDAExtensionBuilder = CUDAExtensionBuilder(
        name=name + "_cuda",
        sources=sources,
        headers=headers,
        build_dir=build_dir,
    )

    def decorator(fn: F) -> F:
        if _builder.is_available():
            try:
                module = _builder.jit_build()
                # look for exported functions matching the op name
                native_fn: Any = getattr(module, name, None) or getattr(module, fn.__name__, None)
                if native_fn is not None:
                    register_native_fn(name, "cuda", native_fn)
            except Exception:
                pass  # build failed — keep Python fallback
        return fn

    return decorator


def triton_kernel(
    name: str,
    autotune_configs: Optional[List[Dict[str, Any]]] = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> Callable[[F], F]:
    """Decorator that binds a Triton kernel to *name*.

    The decorated function should be a ``triton.jit``-decorated kernel
    function. The decorator registers it with auto-tuning.

    Args:
        name: Op identifier.
        autotune_configs: Optional list of tuning config dicts.
        num_warps: Default number of warps.
        num_stages: Default number of pipeline stages.

    Returns:
        A decorator that registers the Triton kernel.
    """
    def decorator(fn: F) -> F:
        if _triton_manager.is_available():
            _triton_manager.register(name, fn, autotune_configs)
            launcher: Optional[Callable[..., Any]] = _triton_manager.get_launcher(name)
            if launcher is not None:
                register_native_fn(name, "triton", launcher)
        return fn

    return decorator


def cython_op(
    name: str,
    pyx_source: str,
    build_dir: str = "/tmp/nameframe_ops",
) -> Callable[[F], F]:
    """Decorator that compiles and binds Cython code to *name*.

    Args:
        name: Op identifier.
        pyx_source: Path to a ``.pyx`` file or inline Cython source string.
        build_dir: Build output directory.

    Returns:
        A decorator that registers the compiled Cython function.
    """
    def decorator(fn: F) -> F:
        if detect_cython():
            from nameframe.ops.backends.cython import CythonBuilder

            builder: CythonBuilder = CythonBuilder(
                name=name + "_cython",
                pyx_source=pyx_source,
                build_dir=build_dir,
            )
            try:
                module = builder.build()
                native_fn: Any = getattr(module, name, None) or getattr(module, fn.__name__, None)
                if native_fn is not None:
                    register_native_fn(name, "cython", native_fn)
            except Exception:
                pass
        return fn

    return decorator


def register_native(name: str, backend: str) -> Callable[[F], F]:
    """Decorator to register a pre-built native function for an op.

    The decorated function itself is stored as the native implementation.

    Args:
        name: Op identifier.
        backend: Backend name (``"cuda"``, ``"triton"``, ``"cpp"``, ``"cython"``).

    Returns:
        A decorator that stores the function in :data:`NATIVE_REGISTRY`.

    Example:
        >>> @register_native("my_op", backend="cuda")
        ... def _cuda_my_op(x):
        ...     return _compiled_module.my_op(x)
    """
    def decorator(fn: F) -> F:
        register_native_fn(name, backend, fn)
        return fn

    return decorator


def build_all(config: Munch) -> Dict[str, str]:
    """Pre-build all registered native ops (warm-start / eager mode).

    Iterates through :data:`NATIVE_REGISTRY` and attempts to JIT-compile
    or load each registered op. This is called by :class:`Pipeline` when
    ``config.ops.build_strategy == "eager"``.

    Args:
        config: The ``config.ops`` subtree.

    Returns:
        Dict mapping op name -> resolved backend (``"cuda"``, ``"python"``, etc.).
    """
    priority: List[str] = config.get("backend_priority", _DEFAULT_PRIORITY)
    status_map: Dict[str, str] = {}

    for op_name in OP_REGISTRY.list():
        resolved: str = "python"
        for backend in priority:
            native_fn = NATIVE_REGISTRY.get(op_name, {}).get(backend)
            if native_fn is not None:
                resolved = backend
                break
        status_map[op_name] = resolved

    return status_map


def status() -> Dict[str, Dict[str, Any]]:
    """Report the status of all registered ops.

    Returns:
        Dict of ``{op_name: {"backend": str, "built": bool}}``.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for op_name in OP_REGISTRY.list():
        backends_available: List[str] = list(NATIVE_REGISTRY.get(op_name, {}).keys())
        result[op_name] = {
            "backend": backends_available[0] if backends_available else "python",
            "built": len(backends_available) > 0,
        }
    return result


def _verify_op(
    name: str,
    fallback_fn: Callable[..., Any],
    native_fn: Callable[..., Any],
    tolerance: float,
) -> None:
    """Compare native and fallback outputs for correctness (best-effort).

    Only logs a warning on mismatch; does not raise.

    Args:
        name: Op identifier.
        fallback_fn: The Python reference implementation.
        native_fn: The native implementation.
        tolerance: Allowed relative error.
    """
    import logging

    import torch

    logger: logging.Logger = logging.getLogger(__name__)
    # generate a simple test input
    test_input: torch.Tensor = torch.randn(2, 64, 32, 32, device="cuda" if torch.cuda.is_available() else "cpu")
    try:
        ref: torch.Tensor = fallback_fn(test_input.to("cpu"))
        nat: torch.Tensor = native_fn(test_input)
        diff: float = float((ref.to(nat.device) - nat).abs().max().item())
        if diff > tolerance:
            logger.warning(
                "Op '%s' verification failed: max diff = %.6f (tolerance = %.6f)",
                name, diff, tolerance,
            )
    except Exception as exc:
        logger.debug("Op '%s' verification skipped: %s", name, exc)
