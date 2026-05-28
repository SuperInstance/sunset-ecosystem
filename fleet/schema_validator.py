"""schema_validator.py — JSON-like schema validation.

Provides:
1. Type checking (str, int, float, bool, list, dict)
2. Required vs optional fields
3. Nested schema validation
4. Custom validators
5. Coercion hints

Usage:
    schema = {
        "name": {"type": str, "required": True},
        "age": {"type": int, "min": 0, "max": 150},
        "tags": {"type": list, "item_type": str},
    }
    validator = SchemaValidator(schema)
    errors = validator.validate({"name": "Alice", "age": 30})
    assert errors == []
"""
from __future__ import annotations

__all__ = [
    "SchemaValidator",
    "ValidationError",
]

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ValidationError:
    """A single validation error."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message

    def __repr__(self) -> str:
        return f"ValidationError({self.path}: {self.message})"


class SchemaValidator:
    """Validate dicts against a schema definition."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema

    def validate(self, data: dict[str, Any]) -> list[ValidationError]:
        """Validate data against schema, return list of errors."""
        errors: list[ValidationError] = []
        self._validate_dict(data, self._schema, "", errors)
        return errors

    def is_valid(self, data: dict[str, Any]) -> bool:
        """Quick check: True if no errors."""
        return len(self.validate(data)) == 0

    def _validate_dict(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
        path: str,
        errors: list[ValidationError],
    ) -> None:
        # Check for extra keys
        for key in data:
            if key not in schema:
                errors.append(ValidationError(f"{path}.{key}" if path else key, "Unknown field"))

        for key, spec in schema.items():
            current_path = f"{path}.{key}" if path else key
            required = spec.get("required", False)
            if key not in data or data[key] is None:
                if required:
                    errors.append(ValidationError(current_path, "Required field missing"))
                continue

            value = data[key]
            self._validate_value(value, spec, current_path, errors)

    def _validate_value(
        self,
        value: Any,
        spec: dict[str, Any],
        path: str,
        errors: list[ValidationError],
    ) -> None:
        expected_type = spec.get("type")
        if expected_type is not None and not isinstance(value, expected_type):
            errors.append(
                ValidationError(path, f"Expected {expected_type.__name__}, got {type(value).__name__}")
            )
            return

        # Range checks for numbers
        if isinstance(value, (int, float)):
            if "min" in spec and value < spec["min"]:
                errors.append(ValidationError(path, f"Value {value} < minimum {spec['min']}"))
            if "max" in spec and value > spec["max"]:
                errors.append(ValidationError(path, f"Value {value} > maximum {spec['max']}"))

        # String checks
        if isinstance(value, str):
            if "min_len" in spec and len(value) < spec["min_len"]:
                errors.append(ValidationError(path, f"Length {len(value)} < minimum {spec['min_len']}"))
            if "max_len" in spec and len(value) > spec["max_len"]:
                errors.append(ValidationError(path, f"Length {len(value)} > maximum {spec['max_len']}"))
            if "pattern" in spec:
                import re
                if not re.match(spec["pattern"], value):
                    errors.append(ValidationError(path, f"Value does not match pattern {spec['pattern']}"))

        # List checks
        if isinstance(value, list):
            if "min_len" in spec and len(value) < spec["min_len"]:
                errors.append(ValidationError(path, f"Length {len(value)} < minimum {spec['min_len']}"))
            if "max_len" in spec and len(value) > spec["max_len"]:
                errors.append(ValidationError(path, f"Length {len(value)} > maximum {spec['max_len']}"))
            item_type = spec.get("item_type")
            item_schema = spec.get("item_schema")
            for i, item in enumerate(value):
                item_path = f"{path}[{i}]"
                if item_type is not None and not isinstance(item, item_type):
                    errors.append(
                        ValidationError(item_path, f"Expected {item_type.__name__}, got {type(item).__name__}")
                    )
                if item_schema is not None and isinstance(item, dict):
                    self._validate_dict(item, item_schema, item_path, errors)

        # Dict / nested schema
        if isinstance(value, dict) and "schema" in spec:
            self._validate_dict(value, spec["schema"], path, errors)

        # Custom validator
        custom = spec.get("validator")
        if custom is not None and callable(custom):
            try:
                if not custom(value):
                    errors.append(ValidationError(path, "Custom validator failed"))
            except Exception as e:
                errors.append(ValidationError(path, f"Custom validator error: {e}"))

    def __repr__(self) -> str:
        return f"SchemaValidator(fields={len(self._schema)})"
