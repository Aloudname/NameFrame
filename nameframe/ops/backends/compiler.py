"""Compiler and toolchain detection for native opextension building."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CompilerConfig:
    """Compiler detection results and flags.

    Attributes:
        cxx: C++ compiler path (or ``None``).
        nvcc: CUDA compiler path (or ``None``).
        extra_flags: Extra compilation flags.
        include_dirs: Additional include directories.
    """

    cxx: Optional[str] = None
    nvcc: Optional[str] = None
    extra_flags: List[str] = field(default_factory=list)
    include_dirs: List[str] = field(default_factory=list)


def detect_cpp() -> Optional[str]:
    """Locate a C++ compiler (g++, clang++, c++).

    Returns:
        Compiler path or ``None``.
    """
    for candidate in ("g++", "c++", "clang++"):
        path: Optional[str] = shutil.which(candidate)
        if path is not None:
            return path
    return None


def detect_cuda() -> Optional[str]:
    """Locate the nvcc CUDA compiler.

    Returns:
        ``nvcc`` path or ``None``.
    """
    return shutil.which("nvcc") or shutil.which("nvcc.exe")


def detect_triton() -> bool:
    """Check whether the Triton language package is importable.

    Returns:
        ``True`` if ``import triton`` succeeds.
    """
    try:
        import triton
        return True
    except ImportError:
        return False


def detect_cython() -> bool:
    """Check whether Cython build tools are importable.

    Returns:
        ``True`` if ``import Cython.Build`` succeeds.
    """
    try:
        import Cython.Build
        return True
    except ImportError:
        return False


def get_cuda_flags(arch: Optional[List[str]] = None) -> List[str]:
    """Generate nvcc flags for target SM architectures.

    Args:
        arch: List of SM architecture strings (e.g. ``["8.0", "9.0"]``).

    Returns:
        List of nvcc flag strings.
    """
    if arch is None:
        arch = ["8.0", "9.0"]
    flags: List[str] = []
    for a in arch:
        flags.extend(["-gencode", f"arch=compute_{a},code=sm_{a}"])
    return flags


def get_cpp_flags(std: str = "c++17") -> List[str]:
    """Generate standard C++ compiler flags.

    Args:
        std: C++ standard version string.

    Returns:
        List of compiler flag strings.
    """
    return [f"-std={std}", "-O3", "-fPIC"]
