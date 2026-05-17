"""C++ extension builder via PyTorch's cpp_extension or pybind11."""

from __future__ import annotations

from typing import Any, List, Optional

from nameframe.ops.backends.compiler import detect_cpp


class CppExtensionBuilder:
    """Build C++ source files into a loadable Python extension module.

    Uses ``torch.utils.cpp_extension.load`` (JIT) for development; a full
    pybind11-based AOT setup can be configured in ``setup.py``.

    Attributes:
        name: Module name.
        sources: List of ``.cpp`` source paths.
    """

    def __init__(
        self,
        name: str,
        sources: List[str],
        build_dir: str = "/tmp/nameframe_ops",
    ) -> None:
        """Initialize the C++ extension builder.

        Args:
            name: Module name for the compiled extension.
            sources: Paths to ``.cpp`` source files.
            build_dir: Working directory for compilation.
        """
        self.name: str = name
        self.sources: List[str] = sources
        self.build_dir: str = build_dir
        self._module: Optional[Any] = None

    def build(self) -> Any:
        """JIT-compile and return the extension module.

        Returns:
            The loaded extension module.

        Raises:
            ImportError: If torch is not installed.
        """
        if self._module is not None:
            return self._module

        import torch.utils.cpp_extension

        self._module = torch.utils.cpp_extension.load(
            name=self.name,
            sources=self.sources,
            build_directory=self.build_dir,
            verbose=False,
        )
        return self._module

    def is_available(self) -> bool:
        """Check whether C++ compilation is possible.

        Returns:
            ``True`` if a C++ compiler is accessible and torch is installed.
        """
        return detect_cpp() is not None
