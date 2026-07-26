"""
variable resolver for config values.

converts placeholder strings e.g. `${search: grid(1, 5, 10)}` into `SearchSpace`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SEARCH_PATTERN: re.Pattern = re.compile(r"^\$\{search:\s*(.+)\}$")


@dataclass
class SearchSpace:
    """
    placeholder of a hyperparameter search space to the sweep engine.

    params:
    - `expression`: `str` type, the raw expression inside `${search: ...}`.
    """

    expression: str


def _resolve_search_vars(node: object) -> None:
    """
    NOTE: an in-place operation of config tree.

    converts `${search: ...}` in tree to `SearchSpace` objects.

    params:
    - `node`: `_ConfigNode` or `list`, config tree root.
    """
    from nameframe.config.loader import _ConfigNode

    if isinstance(node, _ConfigNode):
        for key, value in list(node.__dict__.items()):
            if key == "_frozen":
                continue
            if isinstance(value, str):
                m = _SEARCH_PATTERN.match(value)
                if m:
                    node.__dict__[key] = SearchSpace(m.group(1).strip())
            elif isinstance(value, (_ConfigNode, list)):
                _resolve_search_vars(value)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            if isinstance(value, str):
                m = _SEARCH_PATTERN.match(value)
                if m:
                    node[i] = SearchSpace(m.group(1).strip())
            elif isinstance(value, (_ConfigNode, list)):
                _resolve_search_vars(value)
    else:
        raise TypeError(f"unexpected type {type(node)} for parameter resolution.")
