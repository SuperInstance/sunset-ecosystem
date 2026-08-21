"""Schema validation engine with rules and constraints.

Validates data structures against declarative rules. Supports type
checking, range validation, regex matching, and custom validators.
Used for fleet configuration validation, API input checking, and
data integrity enforcement.

Usage:
    engine = ValidationEngine()
    engine.add_rule("name", required=True, type=str, min_len=1)
    engine.add_rule("age", required=True, type=int, min=0, max=120)
    ok, errors = engine.validate({"name": "Alice", "age": 30})
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional


class ValidationEngine:
    """
    Declarative validation engine.
    """

    def __init__(self):
        self._rules: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Rule definition
    # ------------------------------------------------------------------

    def add_rule(
        self,
        field: str,
        required: bool = False,
        type: Optional[type] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        min_len: Optional[int] = None,
        max_len: Optional[int] = None,
        regex: Optional[str] = None,
        custom: Optional[Callable[[Any], Optional[str]]] = None,
    ) -> None:
        """
        Add a validation rule for a field.

        :param field: Field name.
        :param required: Field must be present.
        :param type: Expected type.
        :param min: Minimum numeric value.
        :param max: Maximum numeric value.
        :param min_len: Minimum length.
        :param max_len: Maximum length.
        :param regex: Regex pattern string.
        :param custom: Custom validator returning error string or None.
        """
        self._rules[field] = {
            "required": required,
            "type": type,
            "min": min,
            "max": max,
            "min_len": min_len,
            "max_len": max_len,
            "regex": regex,
            "custom": custom,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, data: Dict[str, Any]) -> tuple:
        """
        Validate data against rules.

        :returns: (valid: bool, errors: list of str)
        """
        errors: List[str] = []
        for field, rules in self._rules.items():
            if field not in data or data[field] is None:
                if rules["required"]:
                    errors.append(f"{field}: required")
                continue
            value = data[field]
            # Type check
            if rules["type"] and not isinstance(value, rules["type"]):
                errors.append(
                    f"{field}: expected {rules['type'].__name__}, got {type(value).__name__}"
                )
                continue
            # Range check
            if rules["min"] is not None and value < rules["min"]:
                errors.append(f"{field}: must be >= {rules['min']}")
            if rules["max"] is not None and value > rules["max"]:
                errors.append(f"{field}: must be <= {rules['max']}")
            # Length check
            if rules["min_len"] is not None:
                length = len(value) if hasattr(value, "__len__") else 0
                if length < rules["min_len"]:
                    errors.append(f"{field}: length must be >= {rules['min_len']}")
            if rules["max_len"] is not None:
                length = len(value) if hasattr(value, "__len__") else 0
                if length > rules["max_len"]:
                    errors.append(f"{field}: length must be <= {rules['max_len']}")
            # Regex check
            if rules["regex"]:
                if not re.match(rules["regex"], str(value)):
                    errors.append(f"{field}: does not match pattern {rules['regex']}")
            # Custom validator
            if rules["custom"]:
                err = rules["custom"](value)
                if err:
                    errors.append(f"{field}: {err}")
        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {"rules": len(self._rules)}

    def __repr__(self) -> str:
        return f"<ValidationEngine rules={len(self._rules)}>"
