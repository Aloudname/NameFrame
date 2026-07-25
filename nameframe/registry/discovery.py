"""
automatic module discovery is the third method of registry.

scans a directory tree and imports every module found,
triggering side-effect registrations from `@reg.*.register` decorator calls.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path


def discover(directory: Path) -> None:
    """
    recursively imports all modules in a directory tree.

    each imported module can be triggered by `@reg.*.register`.

    params:
    - `directory`: `Path` type, root directory to scan.

    raises:
    - `NotADirectoryError`: if directory does not exist or is a file.
    """
    directory = directory.resolve()
    if not directory.is_dir():
        raise NotADirectoryError(
            f"'{directory}' is not a valid directory."
        )

    for py_file in directory.rglob("*.py"):
        # skip init
        if py_file.name == "__init__.py":
            continue

        # construct module path relative to directory's parent
        # e.g. my_project/model/my_vit.py -> my_project.model.my_vit
        relative: Path = py_file.relative_to(directory.parent)
        module_path: str = (
            str(relative.with_suffix(""))
            .replace("/", ".")
            .replace("\\", ".")
        )

        _import_module_from_path(module_path, py_file)


def _import_module_from_path(module_path: str, file_path: Path) -> None:
    """
    imports a single module from a file path.

    uses `importlib` to load and execute the module,
    which triggers any module-level side effects including registry decorators.

    params:
    - `module_path`: `str` type as dotted module name.
    - `file_path`: `Path` type as absolute path to `*.py`.
    """
    try:
        spec = importlib.util.spec_from_file_location(
            module_path, str(file_path)
        )
        if spec is None or spec.loader is None:
            warnings.warn(
                f"could not create module spec for '{file_path}'"
            )
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path] = module
        spec.loader.exec_module(module)

    except Exception as e:  # noqa: BLE001
        warnings.warn(
            f"failed to import '{module_path}' from '{file_path}': {e}"
        )
