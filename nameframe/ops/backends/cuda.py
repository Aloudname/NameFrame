"""CUDA extension builder — JIT and AOT compilation of .cu source files."""

from __future__ import annotations

import hashlib
import os
from typing import Any, List, Optional

from nameframe.ops.backends.compiler import detect_cuda, get_cuda_flags


class CUDAExtensionBuilder:
    """Build CUDA kernels from ``.cu`` source files.

    Supports JIT compilation (via ``torch.utils.cpp_extension.load_inline``)
    and AOT compilation for deployment. Caches compiled binaries keyed by
    source + compiler hash.

    Attributes:
        name: Extension module name.
        sources: List of ``.cu`` source file paths.
        headers: List of header file paths.
        build_dir: Directory for intermediate build artifacts.
    """

    def __init__(
        self,
        name: str,
        sources: List[str],
        headers: Optional[List[str]] = None,
        build_dir: str = "/tmp/nameframe_ops",
    ) -> None:
        """Initialize the CUDA extension builder.

        Args:
            name: Module name for the compiled extension.
            sources: Paths to ``.cu`` source files.
            headers: Optional paths to ``.h`` / ``.cuh`` header files.
            build_dir: Working directory for JIT compilation.
        """
        self.name: str = name
        self.sources: List[str] = sources
        self.headers: List[str] = headers or []
        self.build_dir: str = build_dir
        self._module: Optional[Any] = None

    def jit_build(self, arch: Optional[List[str]] = None) -> Any:
        """JIT-compile and return the extension module.

        Uses ``torch.utils.cpp_extension.load_inline``.

        Args:
            arch: CUDA SM architectures to target.

        Returns:
            The loaded extension module.

        Raises:
            ImportError: If torch is not installed.
            RuntimeError: If nvcc is not available.
        """
        if self._module is not None:
            return self._module

        import torch.utils.cpp_extension

        cuda_flags: List[str] = get_cuda_flags(arch)
        cpp_flags: List[str] = ["-O3", "-fPIC"]

        # read sources for inline compilation
        cuda_sources: List[str] = []
        for src in self.sources:
            with open(src, "r") as f:
                cuda_sources.append(f.read())

        self._module = torch.utils.cpp_extension.load_inline(
            name=self.name,
            cpp_sources="",
            cuda_sources=cuda_sources,
            functions=[],
            extra_cflags=cpp_flags,
            extra_cuda_cflags=cuda_flags,
            build_directory=self.build_dir,
            verbose=False,
        )
        return self._module

    def aot_build(self, output_dir: str, arch: Optional[List[str]] = None) -> Any:
        """Ahead-of-time build suitable for production deployment.

        Delegates to :meth:`jit_build` as a simplified path; a full AOT
        setup would use ``torch.utils.cpp_extension.CUDAExtension`` in
        ``setup.py``.

        Args:
            output_dir: Where to place compiled artifact.
            arch: CUDA SM architectures.

        Returns:
            The loaded extension module.
        """
        self.build_dir = output_dir
        return self.jit_build(arch=arch)

    def get_cached(self, cache_dir: str) -> Optional[Any]:
        """Check for a cached pre-compiled binary.

        Cache key is SHA256 of concatenated source contents + compiler version.

        Args:
            cache_dir: Directory to search for cached binaries.

        Returns:
            The cached module if found and loadable, else ``None``.
        """
        if not os.path.isdir(cache_dir):
            return None

        key: str = self._cache_key()
        cache_path: str = os.path.join(cache_dir, f"{self.name}_{key}.pt")
        if not os.path.exists(cache_path):
            return None

        try:
            import torch
            self._module = torch.jit.load(cache_path)
            return self._module
        except Exception:
            return None

    def is_available(self) -> bool:
        """Check whether CUDA compilation is possible.

        Returns:
            ``True`` if nvcc is available and PyTorch has CUDA support.
        """
        if detect_cuda() is None:
            return False
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _cache_key(self) -> str:
        """Generate a deterministic cache key from sources + nvcc version."""
        hasher = hashlib.sha256()
        for src in sorted(self.sources):
            with open(src, "rb") as f:
                hasher.update(f.read())
        nvcc: Optional[str] = detect_cuda()
        hasher.update((nvcc or "no-cuda").encode())
        return hasher.hexdigest()[:20]
