"""Data validation with schema and type checking.

Validates data structures against schemas with type checking, range
validation, and custom validators. Used for fleet API request validation,
configuration validation, and data pipeline quality checks.

Usage:
    validator = DataValidator()
    validator.add_schema("user", {
        "name": {"type": str, "required": True},
        "age": {"type": int, "min": 0, "max": 150},
    })
    errors = validator.validate("user", {"name": "Alice", "age": 30})
    assert errors == []  # Valid
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DataValidator:
    """
    Data validator with schema and type checking.
    """

    def __init__(self):
        self._schemas: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def add_schema(self, name: str, schema: Dict[str, Any]) -> bool:
        """
        Register a validation schema.

        :param name: Schema name.
        :param schema: Schema definition dict.
        :returns: True if registered, False if already exists.
        """
        if name in self._schemas:
            return False
        self._schemas[name] = schema
        return True

    def remove_schema(self, name: str) -> bool:
        """Remove a schema."""
        if name in self._schemas:
            del self._schemas[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, schema_name: str, data: Dict[str, Any]) -> List[str]:
        """
        Validate data against a schema.

        :param schema_name: Schema name.
        :param data: Data to validate.
        :returns: List of error messages (empty if valid).
        """
        schema = self._schemas.get(schema_name)
        if not schema:
            return [f"Schema '{schema_name}' not found"]

        errors = []

        for field_name, field_schema in schema.items():
            value = data.get(field_name)

            # Check required
            if field_schema.get("required") and value is None:
                errors.append(f"Field '{field_name}' is required")
                continue

            # Skip if not required and missing
            if value is None and not field_schema.get("required"):
                continue

            # Type check
            expected_type = field_schema.get("type")
            if expected_type and not isinstance(value, expected_type):
                errors.append(f"Field '{field_name}' must be {expected_type.__name__}")
                continue

            # Range check
            if expected_type in (int, float):
                min_val = field_schema.get("min")
                max_val = field_schema.get("max")
                if min_val is not None and value < min_val:
                    errors.append(f"Field '{field_name}' must be >= {min_val}")
                if max_val is not None and value > max_val:
                    errors.append(f"Field '{field_name}' must be <= {max_val}")

            # Length check
            if expected_type in (str, list, tuple):
                min_len = field_schema.get("min_length")
                max_len = field_schema.get("max_length")
                if min_len is not None and len(value) < min_len:
                    errors.append(f"Field '{field_name}' length must be >= {min_len}")
                if max_len is not None and len(value) > max_len:
                    errors.append(f"Field '{field_name}' length must be <= {max_len}")

            # Custom validator
            custom = field_schema.get("validator")
            if custom and not custom(value):
                errors.append(f"Field '{field_name}' failed custom validation")

        return errors

    def is_valid(self, schema_name: str, data: Dict[str, Any]) -> bool:
        """Check if data is valid against schema."""
        return len(self.validate(schema_name, data)) == 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def schemas(self) -> List[str]:
        """List all registered schema names."""
        return list(self._schemas.keys())

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get schema definition."""
        return self._schemas.get(name)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "schemas": len(self._schemas),
        }

    def __repr__(self) -> str:
        return f"<DataValidator schemas={len(self._schemas)}>"
