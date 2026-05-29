"""Configuration file validation with schema support.

Validates configuration files (dicts) against declarative schemas with
type checking, required fields, and custom validators. Used for fleet
service configuration, deployment manifests, and settings validation.

Usage:
    validator = ConfigValidator()
    validator.add_schema("service", {
        "name": {"required": True, "type": str},
        "port": {"required": True, "type": int, "min": 1, "max": 65535},
    })
    ok, errors = validator.validate("service", {"name": "api", "port": 8080})
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class ConfigValidator:
    """
    Configuration validator with schemas.
    """

    def __init__(self):
        self._schemas: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Schema definition
    # ------------------------------------------------------------------

    def add_schema(self, name: str, fields: Dict[str, Dict[str, Any]]) -> None:
        """
        Register a schema.

        :param name: Schema name.
        :param fields: Field definitions with required, type, min, max, regex.
        """
        self._schemas[name] = fields

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, schema_name: str, data: Dict[str, Any]) -> tuple:
        """
        Validate data against a schema.

        :returns: (valid: bool, errors: list of str)
        """
        schema = self._schemas.get(schema_name)
        if not schema:
            return False, [f"Unknown schema: {schema_name}"]

        errors: List[str] = []
        for field_name, rules in schema.items():
            if field_name not in data or data[field_name] is None:
                if rules.get("required"):
                    errors.append(f"{field_name}: required")
                continue

            value = data[field_name]

            # Type check
            expected_type = rules.get("type")
            if expected_type and not isinstance(value, expected_type):
                errors.append(
                    f"{field_name}: expected {expected_type.__name__}, got {type(value).__name__}"
                )
                continue

            # Range
            if "min" in rules and value < rules["min"]:
                errors.append(f"{field_name}: must be >= {rules['min']}")
            if "max" in rules and value > rules["max"]:
                errors.append(f"{field_name}: must be <= {rules['max']}")

            # Regex (for strings)
            import re
            if "regex" in rules:
                if not re.match(rules["regex"], str(value)):
                    errors.append(f"{field_name}: does not match pattern {rules['regex']}")

            # Custom validator
            custom = rules.get("custom")
            if custom:
                err = custom(value)
                if err:
                    errors.append(f"{field_name}: {err}")

        # Check for extra fields (if strict)
        for key in data:
            if key not in schema:
                errors.append(f"{key}: unknown field")

        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def schemas(self) -> List[str]:
        return list(self._schemas.keys())

    def has_schema(self, name: str) -> bool:
        return name in self._schemas

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {"schemas": len(self._schemas)}

    def __repr__(self) -> str:
        return f"<ConfigValidator schemas={len(self._schemas)}>"
