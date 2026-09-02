"""
config freezer.

`FrozenConfigError` is raised when a frozen config node is changed.
"""

from __future__ import annotations

import traceback
from pathlib import Path


class FrozenConfigError(Exception):
    """
    raised when a frozen config node is changed.

    params:
    - `key`: `str` type of changed config key.
    - `location`: `str` type of changed `filepath:line`.
    """

    def __init__(self, key: str, location: str) -> None:
        self.key = key
        self.location = location
        super().__init__(
            f"attempted to change frozen '{key}' at {location}."
        )


def _get_caller_location() -> str:
    """
    extracts the file and line changed.

    returns:
    - `str` type of `path/to/file.py:line`.
    """
    config_dir = str(Path(__file__).resolve().parent).replace("\\", "/")

    for frame in traceback.extract_stack():
        filename = frame.filename.replace("\\", "/")
        # skip config internals and non-file frames (e.g. <frozen runpy>)
        if config_dir not in filename and not filename.startswith("<"):
            return f"{filename}:{frame.lineno}"

    return "<unknown location>"
