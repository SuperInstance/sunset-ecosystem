"""Grammar Engine Security Hardening — input validation for rule ingestion.

Provides `RuleValidator` which sanitizes rule creation to prevent the
4 chaos vectors discovered in the April 22 audit:

  1. Path Traversal   (../../../etc/passwd in rule name)
  2. XSS              (<script> in production.tagline)
  3. SQL Injection    ('; DROP TABLE in production.condition)
  4. Code Injection   (__import__('os').system in name/exec)

Usage::

    from grammar.security_hardening import RuleValidator
    validator = RuleValidator()
    validator.validate_rule_name("safe_rule")           # OK
    validator.validate_rule_name("../../../etc/passwd") # raises ValidationError
"""

from __future__ import annotations

__all__ = ["RuleValidator", "ValidationError"]

import re
from dataclasses import dataclass
from typing import Any


class ValidationError(ValueError):
    """Raised when a rule fails security validation."""

    pass


# ── Sanitization patterns ─────────────────────────────────────────

# Rule names: alphanumeric + underscore only
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# XSS detection: any HTML tag-like content
_XSS_RE = re.compile(r"<[^>]+>")

# SQL injection detection: common SQL metacharacters in suspicious context
_SQLI_RE = re.compile(
    r"('|--|;|DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO)", re.IGNORECASE
)

# Code injection detection: import or system calls
_CODE_INJECTION_RE = re.compile(
    r"(__import__|import\s+os|os\.system|subprocess\.call|eval\(|exec\()"
)

# Path traversal detection
_PATH_TRAVERSAL_RE = re.compile(r"\.\./|/\.\./|\.\.\\|/etc/|/var/|/home/")


@dataclass(frozen=True)
class RuleProvenance:
    """Immutable record of who created a rule and when."""

    creator: str
    creator_type: str  # "GrammarEvolver", "external", "human", etc.
    timestamp: float
    source_ip: str | None = None


class RuleValidator:
    """Security gate for the Grammar Engine rule ingestion pipeline.

    Every rule that enters the system must pass through this validator.
    Invalid rules raise `ValidationError` with a descriptive message.
    """

    def __init__(self) -> None:
        self._provenance_log: list[RuleProvenance] = []

    # ── Public API ─────────────────────────────────────────────────

    def validate_rule_name(self, name: str) -> str:
        """Sanitize and validate a rule name.

        Rules:
          - Must match ``^[a-zA-Z_][a-zA-Z0-9_]*$``
          - No path traversal sequences
          - No Python code injection
        """
        if not isinstance(name, str):
            raise ValidationError(f"Rule name must be str, got {type(name).__name__}")
        if len(name) > 128:
            raise ValidationError(f"Rule name too long: {len(name)} > 128")
        if not _SAFE_NAME_RE.match(name):
            raise ValidationError(
                f"Rule name '{name}' contains invalid characters. "
                "Allowed: alphanumeric + underscore, must start with letter or _"
            )
        if _PATH_TRAVERSAL_RE.search(name):
            raise ValidationError(
                f"Rule name '{name}' contains path traversal sequence"
            )
        if _CODE_INJECTION_RE.search(name):
            raise ValidationError(f"Rule name '{name}' contains code injection payload")
        return name

    def validate_production_fields(self, production: dict[str, Any]) -> dict[str, Any]:
        """Sanitize all fields in a production block.

        Checks:
          - tagline: no HTML/script tags
          - condition: no SQL metacharacters
          - exec: sandboxed execution only
          - Any string value: no code injection patterns
        """
        if not isinstance(production, dict):
            raise ValidationError("Production must be a dict")

        sanitized: dict[str, Any] = {}
        for key, value in production.items():
            if isinstance(value, str):
                value = self._sanitize_string(value, context=f"production.{key}")
            sanitized[key] = value

        # Extra checks for known sensitive fields
        tagline = sanitized.get("tagline", "")
        if _XSS_RE.search(tagline):
            raise ValidationError(
                f"production.tagline contains HTML/script: {tagline[:50]}..."
            )

        condition = sanitized.get("condition", "")
        if _SQLI_RE.search(condition):
            raise ValidationError(
                f"production.condition contains SQL injection payload: {condition[:50]}..."
            )

        # Sandboxed exec validation
        exec_code = sanitized.get("exec", "")
        if exec_code:
            sanitized["exec"] = self._sandbox_exec(exec_code)

        return sanitized

    def track_provenance(
        self,
        creator: str,
        creator_type: str = "external",
        timestamp: float | None = None,
        source_ip: str | None = None,
    ) -> RuleProvenance:
        """Record who created a rule for audit trails.

        Returns the immutable provenance record.
        """
        import time

        prov = RuleProvenance(
            creator=creator,
            creator_type=creator_type,
            timestamp=timestamp or time.time(),
            source_ip=source_ip,
        )
        self._provenance_log.append(prov)
        return prov

    def validate_full_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        """Validate an entire rule dict before ingestion.

        Raises `ValidationError` if any security check fails.
        """
        name = rule.get("name", "")
        self.validate_rule_name(name)

        production = rule.get("production", {})
        if production:
            rule["production"] = self.validate_production_fields(production)

        # Track provenance if provided
        provenance = rule.get("provenance")
        if provenance:
            self.track_provenance(**provenance)
        else:
            self.track_provenance(creator="unknown", creator_type="external")

        return rule

    def get_provenance_log(self) -> list[RuleProvenance]:
        """Return all recorded provenance entries."""
        return list(self._provenance_log)

    # ── Internal ────────────────────────────────────────────────────

    def _sanitize_string(self, value: str, context: str = "field") -> str:
        """Basic string sanitization: strip null bytes, limit length."""
        value = value.replace("\x00", "")
        if len(value) > 4096:
            raise ValidationError(f"{context} exceeds 4096 characters")
        if _CODE_INJECTION_RE.search(value):
            raise ValidationError(f"{context} contains code injection pattern")
        return value

    def _sandbox_exec(self, code: str) -> str:
        """Validate that exec code is safe for sandboxed execution.

        Current sandbox policy: block all imports, os, subprocess, eval, exec.
        Future: replace with restricted execution environment.
        """
        forbidden = [
            "__import__",
            "import os",
            "import sys",
            "import subprocess",
            "os.system",
            "os.path",
            "subprocess.call",
            "subprocess.Popen",
            "eval(",
            "exec(",
            "compile(",
            "open(",
            "file(",
        ]
        lower = code.lower()
        for token in forbidden:
            if token in lower:
                raise ValidationError(
                    f"production.exec contains forbidden token '{token}'"
                )
        return code


def create_rule_from_dict(rule: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper: validate a rule dict and return it.

    Raises `ValidationError` if validation fails.
    """
    validator = RuleValidator()
    return validator.validate_full_rule(rule)
