"""
config schema check.

compares actual param dicts with `FieldSchema`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

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


def validate_schema_definition(schema: dict[str, FieldSchema]) -> None:
    """Reject malformed field definitions and invalid defaults before use."""
    if not isinstance(schema, Mapping):
        raise ConfigConflictError(["schema must be a mapping with string keys."])
    errors: list[str] = []
    allowed = {"type", "default", "help", "choices", "range", "nullable"}
    for key, field in schema.items():
        prefix = f"schema field {key!r}"
        if not isinstance(key, str):
            errors.append(f"{prefix} must have a string name.")
            continue
        if not isinstance(field, Mapping):
            errors.append(f"{prefix} must be a mapping.")
            continue
        field_errors: list[str] = []
        unknown = set(field) - allowed
        if unknown:
            field_errors.append(
                f"{prefix} has unknown rules: {sorted(map(repr, unknown))}."
            )
        declared_type = field.get("type")
        if declared_type is not None:
            if not isinstance(declared_type, type):
                field_errors.append(f"{prefix} type must be a Python runtime type.")
            else:
                try:
                    isinstance(None, declared_type)
                except TypeError:
                    field_errors.append(
                        f"{prefix} type cannot be used for runtime validation."
                    )
        if "nullable" in field and not isinstance(field["nullable"], bool):
            field_errors.append(f"{prefix} nullable must be a bool.")
        if "help" in field and not isinstance(field["help"], str):
            field_errors.append(f"{prefix} help must be a string.")
        choices = field.get("choices")
        if choices is not None and not isinstance(choices, list):
            field_errors.append(f"{prefix} choices must be a list or None.")
        bounds = field.get("range")
        if bounds is not None:
            if (
                not isinstance(bounds, (list, tuple))
                or len(bounds) != 2
                or any(
                    isinstance(bound, bool)
                    or not isinstance(bound, (int, float))
                    or (isinstance(bound, float) and math.isnan(bound))
                    for bound in bounds
                )
            ):
                field_errors.append(
                    f"{prefix} range needs two numeric, non-NaN bounds."
                )
            elif bounds[0] > bounds[1]:
                field_errors.append(f"{prefix} range lower bound exceeds upper bound.")
            if declared_type is not None and declared_type not in (int, float):
                field_errors.append(f"{prefix} range requires int or float type.")
        errors.extend(field_errors)
        if not field_errors and "default" in field:
            errors.extend(_value_errors(f"{key} (default)", field["default"], field))
    if errors:
        raise ConfigConflictError(errors)


def _value_errors(key: str, value: Any, field: FieldSchema) -> list[str]:
    """Check one value after its field definition has been validated."""
    if value is None:
        return (
            []
            if field.get("nullable", False)
            else [f"field '{key}' does not allow None."]
        )
    declared_type = field.get("type")
    if declared_type is not None and (
        not isinstance(value, declared_type)
        or (declared_type in (int, float) and isinstance(value, bool))
    ):
        return [
            f"field '{key}' expects {declared_type.__name__}, "
            f"got {type(value).__name__} (value={value!r})."
        ]
    if isinstance(value, float) and not math.isfinite(value):
        return [f"field '{key}' must be finite, got {value!r}."]
    errors: list[str] = []
    choices = field.get("choices")
    if choices is not None and value not in choices:
        errors.append(
            f"field '{key}' has value {value!r}, but must be one of {choices}."
        )
    bounds = field.get("range")
    if bounds is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"field '{key}' must be numeric to apply range {bounds}.")
        elif not bounds[0] <= value <= bounds[1]:
            errors.append(
                f"field '{key}' value {value} out of range [{bounds[0]}, {bounds[1]}]."
            )
    return errors


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
    validate_schema_definition(schema)
    if not isinstance(params, Mapping) or any(not isinstance(k, str) for k in params):
        raise ConfigConflictError(["params must be a mapping with string keys."])
    errors: list[str] = []
    schema_keys = set(schema.keys())
    param_keys = set(params.keys())

    # unknown keys & suggestions
    for key in sorted(param_keys - schema_keys):
        suggestion = _suggest(key, sorted(schema_keys))
        if suggestion:
            errors.append(f"unknown field '{key}', did you mean '{suggestion}'?")
        else:
            errors.append(f"unknown field '{key}', available: {sorted(schema_keys)}")

    # each-field check
    for key, field in schema.items():
        if key not in params and "default" not in field:
            errors.append(f"missing required field '{key}'.")
            continue

        value = params[key] if key in params else field["default"]

        errors.extend(_value_errors(key, value, field))

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
