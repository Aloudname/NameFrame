"""Backend discovery, priority resolution, and native extension builders."""

from nameframe.ops.backends.compiler import (
    CompilerConfig,
    detect_cpp,
    detect_cuda,
    detect_cython,
    detect_triton,
    get_cpp_flags,
    get_cuda_flags,
)
from nameframe.ops.backends.cpp import CppExtensionBuilder
from nameframe.ops.backends.cuda import CUDAExtensionBuilder
from nameframe.ops.backends.cython import CythonBuilder
from nameframe.ops.backends.triton import TritonKernelManager

__all__ = [
    "CompilerConfig",
    "detect_cpp",
    "detect_cuda",
    "detect_cython",
    "detect_triton",
    "get_cpp_flags",
    "get_cuda_flags",
    "CUDAExtensionBuilder",
    "CppExtensionBuilder",
    "CythonBuilder",
    "TritonKernelManager",
]
