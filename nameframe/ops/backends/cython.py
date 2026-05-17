"""Cython extension builder — compiles .pyx source to .so."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from nameframe.ops.backends.compiler import detect_cython


class CythonBuilder:
    """Compile Cython ``.pyx`` source files into a loadable ``.so`` module.

    Attributes:
        name: Module name.
        pyx_source: Path to the ``.pyx`` file, or inline Cython source string.
        build_dir: Working directory for the Cython build.
    """

    def __init__(
        self,
        name: str,
        pyx_source: str,
        build_dir: str = "/tmp/nameframe_ops",
        language_level: int = 3,
        compiler_directives: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the Cython builder.

        Args:
            name: Module name.
            pyx_source: Path to ``.pyx`` or raw Cython source.
            build_dir: Build output directory.
            language_level: Python language level for Cython.
            compiler_directives: Cython compiler directives dict.
        """
        self.name: str = name
        self.pyx_source: str = pyx_source
        self.build_dir: str = build_dir
        self.language_level: int = language_level
        self.compiler_directives: Dict[str, Any] = compiler_directives or {
            "boundscheck": False,
            "wraparound": False,
        }
        self._module: Optional[Any] = None

    def build(self) -> Any:
        """Compile the Cython source and return the loaded module.

        Returns:
            The loaded extension module.

        Raises:
            ImportError: If Cython is not installed.
        """
        if self._module is not None:
            return self._module

        from Cython.Build import cythonize
        from setuptools import Distribution, Extension

        os.makedirs(self.build_dir, exist_ok=True)

        # determine if pyx_source is a file path or inline source
        if os.path.isfile(self.pyx_source):
            pyx_path: str = self.pyx_source
        else:
            pyx_path = os.path.join(self.build_dir, f"{self.name}.pyx")
            with open(pyx_path, "w") as f:
                f.write(self.pyx_source)

        ext: Extension = Extension(
            name=self.name,
            sources=[pyx_path],
            language="c",
        )

        # set compiler directives
        for k, v in self.compiler_directives.items():
            setattr(ext, f"cython_{k}", v)

        cythonize(
            [ext],
            language_level=self.language_level,
            build_dir=self.build_dir,
            quiet=True,
        )

        dist: Distribution = Distribution({"ext_modules": [ext]})
        dist.run_command("build_ext")
        # the built module path depends on platform; simplified import
        import importlib

        self._module = importlib.import_module(self.name)
        return self._module

    def is_available(self) -> bool:
        """Check whether Cython compilation is possible.

        Returns:
            ``True`` if Cython is importable.
        """
        return detect_cython()
