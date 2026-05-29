from __future__ import annotations

import re
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ValidationError:
    """A validation error."""
    field: str
    message: str
    severity: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity,
        }


class SchemaValidator:
    """
    Schema validator for breeding configurations and fleet state.

    Validates dicts against JSON-like schemas with type checking.
    """

    def __init__(self):
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register(self, schema_name: str, schema: Dict[str, Any]):
        """Register a schema."""
        self._schemas[schema_name] = schema

    def validate(self, data: Dict[str, Any], schema_name: str) -> List[ValidationError]:
        """Validate data against a named schema."""
        if schema_name not in self._schemas:
            return [ValidationError("_schema", f"Schema '{schema_name}' not found")]
        schema = self._schemas[schema_name]
        return self._validate_object(data, schema, "")

    def _validate_object(self, data: Dict[str, Any], schema: Dict[str, Any],
                         path: str) -> List[ValidationError]:
        errors = []
        for key, spec in schema.items():
            current_path = f"{path}.{key}" if path else key
            if key not in data:
                if spec.get("required", False):
                    errors.append(ValidationError(current_path, f"Missing required field: {key}"))
                continue

            value = data[key]
            errors.extend(self._validate_value(value, spec, current_path))
        return errors

    def _validate_value(self, value: Any, spec: Dict[str, Any],
                        path: str) -> List[ValidationError]:
        errors = []
        expected_type = spec.get("type")

        if expected_type == "string" and not isinstance(value, str):
            errors.append(ValidationError(path, f"Expected string, got {type(value).__name__}"))
        elif expected_type == "number" and not isinstance(value, (int, float)):
            errors.append(ValidationError(path, f"Expected number, got {type(value).__name__}"))
        elif expected_type == "integer" and not isinstance(value, int):
            errors.append(ValidationError(path, f"Expected integer, got {type(value).__name__}"))
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(ValidationError(path, f"Expected boolean, got {type(value).__name__}"))
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(ValidationError(path, f"Expected array, got {type(value).__name__}"))
        elif expected_type == "object" and not isinstance(value, dict):
            errors.append(ValidationError(path, f"Expected object, got {type(value).__name__}"))

        # Range validation
        if expected_type in ("number", "integer") and isinstance(value, (int, float)):
            if "min" in spec and value < spec["min"]:
                errors.append(ValidationError(path, f"Value {value} < minimum {spec['min']}"))
            if "max" in spec and value > spec["max"]:
                errors.append(ValidationError(path, f"Value {value} > maximum {spec['max']}"))

        # String pattern validation
        if expected_type == "string" and isinstance(value, str):
            if "pattern" in spec and not re.match(spec["pattern"], value):
                errors.append(ValidationError(path, f"Value does not match pattern {spec['pattern']}"))
            if "min_length" in spec and len(value) < spec["min_length"]:
                errors.append(ValidationError(path, f"String too short: {len(value)} < {spec['min_length']}"))

        # Nested object validation
        if expected_type == "object" and isinstance(value, dict):
            if "properties" in spec:
                errors.extend(self._validate_object(value, spec["properties"], path))

        # Array item validation
        if expected_type == "array" and isinstance(value, list):
            if "items" in spec:
                for i, item in enumerate(value):
                    item_path = f"{path}[{i}]"
                    errors.extend(self._validate_value(item, spec["items"], item_path))

        return errors

    def is_valid(self, data: Dict[str, Any], schema_name: str) -> bool:
        """Check if data is valid."""
        return len(self.validate(data, schema_name)) == 0

    def get_registered_schemas(self) -> List[str]:
        """Get list of registered schema names."""
        return list(self._schemas.keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schemas": len(self._schemas),
        }
