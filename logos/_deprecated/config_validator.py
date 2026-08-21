"""config_validator.py — Schema-based configuration validation with type checking.

Provides declarative config schemas:
- Type checking (int, float, str, bool, list, dict)
- Range validation (min, max for numeric)
- Pattern validation (regex for strings)
- Required/optional fields
- Default values
- Nested schema support
- Custom validators

Usage:
    schema = Schema({
        "name": Field(str, required=True, min_length=1),
        "port": Field(int, required=True, min=1, max=65535, default=8080),
        "nodes": Field(list, required=True, item_schema=Field(str)),
    })
    result = schema.validate(config_dict)
    # result.errors or result.value
"""

from __future__ import annotations

__all__ = [
    "Schema",
    "Field",
    "ValidationResult",
    "ValidationError",
]

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ValidationError:
    """Single validation error."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass
class ValidationResult:
    """Result of schema validation."""

    value: dict[str, Any] | None
    errors: list[ValidationError] = field(default_factory=list)
    valid: bool = True

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


@dataclass
class Field:
    """Schema field definition."""

    type: type | None = None
    required: bool = True
    default: Any = None
    min: int | float | None = None
    max: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    choices: list[Any] | None = None
    item_schema: "Field" | None = None  # for list items
    dict_schema: dict[str, "Field"] | None = None  # for dict values
    custom_validator: Callable[[Any], str | None] | None = None
    allow_none: bool = False

    def _validate_value(self, value: Any, path: str) -> list[ValidationError]:
        errors: list[ValidationError] = []

        if value is None:
            if self.allow_none:
                return errors
            if self.required:
                errors.append(ValidationError(path, "required field is None"))
            return errors

        # Type check
        if self.type is not None and not isinstance(value, self.type):
            errors.append(
                ValidationError(
                    path, f"expected {self.type.__name__}, got {type(value).__name__}"
                )
            )
            return errors  # further checks would fail with wrong type

        # Range check
        if self.min is not None and value < self.min:
            errors.append(ValidationError(path, f"value {value} < min {self.min}"))
        if self.max is not None and value > self.max:
            errors.append(ValidationError(path, f"value {value} > max {self.max}"))

        # Length check
        if self.min_length is not None and len(value) < self.min_length:
            errors.append(
                ValidationError(
                    path, f"length {len(value)} < min_length {self.min_length}"
                )
            )
        if self.max_length is not None and len(value) > self.max_length:
            errors.append(
                ValidationError(
                    path, f"length {len(value)} > max_length {self.max_length}"
                )
            )

        # Pattern check
        if self.pattern is not None and isinstance(value, str):
            if not re.match(self.pattern, value):
                errors.append(
                    ValidationError(
                        path, f"value '{value}' does not match pattern '{self.pattern}'"
                    )
                )

        # Choices check
        if self.choices is not None and value not in self.choices:
            errors.append(
                ValidationError(path, f"value {value} not in choices {self.choices}")
            )

        # List item schema
        if (
            self.type is list
            and self.item_schema is not None
            and isinstance(value, list)
        ):
            for i, item in enumerate(value):
                item_errors = self.item_schema._validate_value(item, f"{path}[{i}]")
                errors.extend(item_errors)

        # Dict schema
        if (
            self.type is dict
            and self.dict_schema is not None
            and isinstance(value, dict)
        ):
            for key, field_def in self.dict_schema.items():
                if key in value:
                    dict_errors = field_def._validate_value(value[key], f"{path}.{key}")
                    errors.extend(dict_errors)
                elif field_def.required:
                    errors.append(
                        ValidationError(f"{path}.{key}", "required field missing")
                    )

        # Custom validator
        if self.custom_validator is not None:
            msg = self.custom_validator(value)
            if msg:
                errors.append(ValidationError(path, msg))

        return errors


@dataclass
class Schema:
    """Schema definition for dict validation."""

    fields: dict[str, Field]

    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate a dict against this schema."""
        errors: list[ValidationError] = []
        result: dict[str, Any] = {}

        # Check for unknown fields
        known = set(self.fields.keys())
        for key in data:
            if key not in known:
                errors.append(ValidationError(key, f"unknown field '{key}'"))

        # Validate each field
        for key, field_def in self.fields.items():
            if key in data:
                field_errors = field_def._validate_value(data[key], key)
                errors.extend(field_errors)
                if not field_errors:
                    result[key] = data[key]
            elif (
                field_def.required
                and field_def.default is None
                and not field_def.allow_none
            ):
                errors.append(ValidationError(key, "required field missing"))
            elif field_def.default is not None:
                result[key] = field_def.default
            elif field_def.allow_none:
                result[key] = None

        return ValidationResult(
            value=result if not errors else None,
            errors=errors,
            valid=not errors,
        )

    def validate_or_raise(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate and return value, or raise ValueError with errors."""
        result = self.validate(data)
        if not result.is_valid:
            raise ValueError(
                "validation failed: " + "; ".join(str(e) for e in result.errors)
            )
        return result.value

    def __repr__(self) -> str:
        return f"Schema(fields={list(self.fields.keys())})"
