"""import component modules deterministically, preserving python package semantics."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True)
class DiscoveryFailure:
    """a component module that could not be imported, with its original error."""

    module: str
    path: Path
    error: Exception


@dataclass(frozen=True)
class DiscoveryReport:
    """successful module names (including cached imports) and failed imports."""

    imported: tuple[str, ...]
    failures: tuple[DiscoveryFailure, ...]


class DiscoveryError(ImportError):
    """strict discovery failed; the report retains all attempted import results."""

    def __init__(self, report: DiscoveryReport) -> None:
        self.report = report
        details = "\n".join(
            f"  - {failure.module} ({failure.path}): {failure.error}"
            for failure in report.failures
        )
        super().__init__(f"component discovery failed:\n{details}")


def _check_origin(
    module_name: str,
    origin: str | None,
    locations: Iterable[str] | None,
    expected: Path,
) -> None:
    """validate a module or import spec before running component code."""
    expected = expected.resolve()
    if expected.is_dir():
        if (
            origin is None
            and locations is not None
            and expected in {Path(path).resolve() for path in locations}
        ):
            return
    elif origin is not None and Path(origin).resolve() == expected:
        return
    raise ImportError(
        f"module '{module_name}' belongs to {origin or locations!r}, "
        f"expected '{expected}'."
    )


def _checked_import(
    module_name: str, source: Path, package_root: Path, package_name: str
) -> ModuleType:
    """check each parent before importing descendants from its search path."""
    parts = module_name.split(".")
    search_path = sys.path
    module: ModuleType | None = None
    for index in range(len(parts)):
        name = ".".join(parts[: index + 1])
        if name == module_name:
            expected = source
        else:
            directory = package_root.joinpath(*parts[1 : index + 1])
            init = directory / "__init__.py"
            expected = init if init.is_file() else directory
        # The synthetic package has no source file and preserves relative imports.
        if name == package_name and not (package_root / "__init__.py").is_file():
            expected = package_root
        if name in sys.modules:
            module = sys.modules[name]
            if module is None:
                raise ImportError(f"module '{name}' is blocked in the import cache.")
            _check_origin(
                name,
                getattr(module, "__file__", None),
                getattr(module, "__path__", None),
                expected,
            )
        else:
            spec = importlib.machinery.PathFinder.find_spec(name, search_path)
            if spec is None:
                raise ModuleNotFoundError(f"cannot find component module '{name}'.")
            _check_origin(name, spec.origin, spec.submodule_search_locations, expected)
            module = importlib.import_module(name)
            _check_origin(
                name,
                getattr(module, "__file__", None),
                getattr(module, "__path__", None),
                expected,
            )
        search_path = getattr(module, "__path__", [])
    assert module is not None
    return module


def discover(directory: Path | str, *, strict: bool = False) -> DiscoveryReport:
    """import component files and package initializers, returning an import report.

    normal packages retain their qualified names and relative imports. standalone
    directories use a path-derived private package so equal directory names do not
    collide. successful modules are cached by python, failed imports can be retried.
    conflicting package origins are rejected before importing their descendants.

    default mode warns on failures and continues. strict mode attempts every file
    then raises DiscoveryError with the complete report. imports execute code;
    registrations and other side effects before an import failure are not rolled
    back, so discovery should run before constructing runtime components.
    """
    if not isinstance(strict, bool):
        raise TypeError("discovery strict must be a boolean.")
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a valid directory.")

    package_root = directory
    if (directory / "__init__.py").is_file():
        while (package_root.parent / "__init__.py").is_file():
            package_root = package_root.parent
        package_name = package_root.name
        search_path = str(package_root.parent)
    else:
        digest = hashlib.sha256(str(directory).encode()).hexdigest()[:16]
        package_name = f"_nameframe_discovery_{digest}"
        search_path = None
        if package_name not in sys.modules:
            spec = importlib.util.spec_from_loader(
                package_name, loader=None, is_package=True
            )
            package = importlib.util.module_from_spec(spec)
            package.__path__ = [str(directory)]
            sys.modules[package_name] = package

    imported: list[str] = []
    failures: list[DiscoveryFailure] = []
    importlib.invalidate_caches()
    # Restore the original list, even if imported component code changes sys.path.
    original_path = list(sys.path)
    if search_path is not None:
        sys.path.insert(0, search_path)
    try:
        for py_file in sorted(directory.rglob("*.py")):
            if "__pycache__" in py_file.relative_to(directory).parts:
                continue
            relative = py_file.relative_to(package_root).with_suffix("")
            if py_file.name == "__init__.py":
                relative = relative.parent
            module_path = ".".join((package_name, *relative.parts))
            try:
                _checked_import(module_path, py_file, package_root, package_name)
                imported.append(module_path)
            except Exception as exc:  # imports execute arbitrary component code.
                failures.append(DiscoveryFailure(module_path, py_file, exc))
                if not strict:
                    warnings.warn(
                        f"failed to import '{module_path}' from '{py_file}': {exc}",
                        stacklevel=2,
                    )
    finally:
        sys.path[:] = original_path

    report = DiscoveryReport(tuple(imported), tuple(failures))
    if strict and failures:
        raise DiscoveryError(report) from failures[0].error
    return report
