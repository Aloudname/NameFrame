"""
config schema check.

compares actual param dicts with `FieldSchema`.
"""

from __future__ import annotations

from nameframe.utils.typing import FieldSchema


class ConfigConflictError(ValueError):
    """
    raised when config values conflict with schema check.

    params:
    - `errors`: `list[str]` type, one message per conflict.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        msg = "config conflicts:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)


def validate_schema(params: dict, schema: dict[str, FieldSchema]) -> None:
    """
    compares param dict with `FieldSchema`.

    checks:
    - unknown keys
    - missing required keys
    - type mismatches
    - value not in allowed choices
    - value not in allowed numeric range

    params:
    - `params`: `dict` type, the actual parameter values.
    - `schema`: `dict[str, FieldSchema]` type, FieldSchema values.

    raises:
    - `ConfigConflictError`: if any rule fails.
    """
    errors: list[str] = []
    schema_keys = set(schema.keys())
    param_keys = set(params.keys())

    # unknown keys & suggestions
    for key in sorted(param_keys - schema_keys):
        suggestion = _suggest(key, list(schema_keys))
        if suggestion:
            errors.append(f"unknown field '{key}', did you mean '{suggestion}'?")
        else:
            errors.append(f"unknown field '{key}', available: {sorted(schema_keys)}")

    # each-field check
    for key, field in schema.items():
        if key not in params:
            if "default" not in field:
                errors.append(f"missing required field '{key}'.")
            continue

        value = params[key]

        # null check
        if value is None:
            if not field.get("nullable", False):
                errors.append(f"field '{key}' does not allow None.")
            continue

        # type check
        declared_type = field.get("type")
        if declared_type is not None and not isinstance(value, declared_type):
            errors.append(
                f"field '{key}' expects {declared_type.__name__}, "
                f"got {type(value).__name__} (value={value!r})."
            )

        # choices check
        choices = field.get("choices")
        if choices is not None and value not in choices:
            errors.append(
                f"field '{key}' has value {value!r}, but must be one of {choices}."
            )

        # range check
        value_range = field.get("range")
        if value_range is not None and isinstance(value, (int, float)):
            lo, hi = value_range
            if not (lo <= value <= hi):
                errors.append(
                    f"field '{key}' value {value} out of range [{lo}, {hi}]."
                )

    # error delivery
    if errors:
        raise ConfigConflictError(errors)


def _suggest(target: str, candidates: list[str]) -> str | None:
    """
    finds the closest match `candidates` of given `target`.

    params:
    - `target`: `str` type, mistyped key.
    - `candidates`: `list[str]` type, valid keys.

    returns:
    - `str` match, or `None` if no good match.
    """
    if not candidates:
        return None

    best = min(candidates, key=lambda c: _levenshtein(target, c))

    # threshold = max(3, len(target) // 2)
    if _levenshtein(target, best) <= max(3, len(target) // 2):
        return best
    return None


def _levenshtein(a: str, b: str) -> int:
    """
    computes the Levenshtein distance between two strings.

    params:
    - `a`: `str` type.
    - `b`: `str` type.

    returns:
    - `int` type, the minimum number of single-character edits.
    """
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)

    prev_row: list[int] = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr_row = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr_row.append(
                min(
                    prev_row[j] + 1,  # delete
                    curr_row[j - 1] + 1,  # insert
                    prev_row[j - 1] + cost,  # substitute
                )
            )
        prev_row = curr_row
    return prev_row[-1]
