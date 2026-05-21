"""Grammar Engine — Rule ingestion with input validation.

Provides create_rule() with strict sanitization on all string fields.
Blocks path traversal, XSS, SQL injection, and arbitrary code execution.
"""

import ast
import html
import re
from dataclasses import dataclass, field
from typing import List, Optional

# ── Validation Constants ─────────────────────────────────────────────

RULE_NAME_MAX_LEN = 64
TAGLINE_MAX_LEN = 256
CONDITION_MAX_LEN = 1024
EXEC_MAX_LEN = 512

# Allow alphanumerics, underscores, hyphens. No dots, slashes, backslashes.
RULE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# SQLi blacklist — semicolons, comment dashes, and dangerous keywords.
SQLI_BLACKLIST = re.compile(
    r";|--|\b(DROP|DELETE|INSERT|UPDATE|ALTER|EXEC|EXECUTE|UNION|SELECT)\b",
    re.IGNORECASE,
)

# HTML tag stripper — removes <script> and any other dangerous tags.
# A full HTML parser is overkill; we strip all angle-bracket tags and escape.
HTML_TAG_PATTERN = re.compile(r"<[^>]*>")


# ── Data Classes ─────────────────────────────────────────────────────

@dataclass
class Production:
    tagline: str = ""
    condition: str = ""
    exec_field: Optional[str] = field(default=None, repr=False)  # renamed from 'exec'


@dataclass
class Rule:
    name: str
    production: Production


# ── Validation Exceptions ──────────────────────────────────────────

class ValidationError(ValueError):
    """Raised when a rule field fails security validation."""

    pass


# ── Core Validation Functions ──────────────────────────────────────

def validate_rule_name(name: str) -> str:
    """Sanitize rule name.

    - Alphanumeric + underscore + hyphen only.
    - Max 64 characters.
    - Rejects path traversal sequences.
    """
    if not isinstance(name, str):
        raise ValidationError("Rule name must be a string.")
    if len(name) > RULE_NAME_MAX_LEN:
        raise ValidationError(f"Rule name exceeds {RULE_NAME_MAX_LEN} characters.")
    if not RULE_NAME_PATTERN.match(name):
        raise ValidationError(
            "Rule name contains illegal characters. "
            "Allowed: a-z, A-Z, 0-9, _, -."
        )
    return name


def validate_tagline(tagline: str) -> str:
    """Sanitize production tagline.

    - Strip all HTML tags (especially <script>).
    - HTML-escape remaining text.
    - Max 256 characters.
    """
    if not isinstance(tagline, str):
        raise ValidationError("Tagline must be a string.")
    if len(tagline) > TAGLINE_MAX_LEN:
        raise ValidationError(f"Tagline exceeds {TAGLINE_MAX_LEN} characters.")
    tagline = HTML_TAG_PATTERN.sub("", tagline)  # strip tags
    tagline = html.escape(tagline)  # escape ampersands, quotes, etc.
    return tagline


def validate_condition(condition: str) -> str:
    """Sanitize production condition.

    - Blacklist SQLi patterns: ;, --, DROP, DELETE, etc.
    - Max 1024 characters.
    """
    if not isinstance(condition, str):
        raise ValidationError("Condition must be a string.")
    if len(condition) > CONDITION_MAX_LEN:
        raise ValidationError(f"Condition exceeds {CONDITION_MAX_LEN} characters.")
    if SQLI_BLACKLIST.search(condition):
        raise ValidationError("Condition contains blocked SQL injection patterns.")
    return condition


def validate_exec_field(exec_code: Optional[str]) -> Optional[str]:
    """Sandbox or disable production.exec entirely.

    **Option A (recommended):** Return None — disable exec fields in rules.
    **Option B (if exec is required):** Parse with ast.literal_eval only.
    **Never use eval(), exec(), or compile() on untrusted input.**
    """
    if exec_code is None:
        return None
    if not isinstance(exec_code, str):
        raise ValidationError("Exec field must be a string or None.")
    if len(exec_code) > EXEC_MAX_LEN:
        raise ValidationError(f"Exec field exceeds {EXEC_MAX_LEN} characters.")

    # ── Recommended: disable exec entirely ───────────────────────
    # Uncomment the next line to forbid exec fields completely.
    # raise ValidationError("Exec fields are disabled for security reasons.")

    # ── Option B: ast.literal_eval sandbox ─────────────────────────
    try:
        ast.literal_eval(exec_code)
    except (ValueError, SyntaxError) as exc:
        raise ValidationError(
            f"Exec field is not a safe literal expression: {exc}"
        ) from exc

    return exec_code


# ── Rule Creation API ────────────────────────────────────────────────

def create_rule(
    name: str,
    tagline: str = "",
    condition: str = "",
    exec_field: Optional[str] = None,
) -> Rule:
    """Create a validated Rule.

    All inputs are strictly sanitized before the Rule is returned.
    Raises ValidationError on any security violation.
    """
    clean_name = validate_rule_name(name)
    clean_tagline = validate_tagline(tagline)
    clean_condition = validate_condition(condition)
    clean_exec = validate_exec_field(exec_field)

    return Rule(
        name=clean_name,
        production=Production(
            tagline=clean_tagline,
            condition=clean_condition,
            exec_field=clean_exec,
        ),
    )


# ── Batch / JSON ingestion helper ──────────────────────────────────

def create_rule_from_dict(data: dict) -> Rule:
    """Convenience wrapper for JSON/rule-dict ingestion."""
    return create_rule(
        name=data.get("name", ""),
        tagline=data.get("production", {}).get("tagline", ""),
        condition=data.get("production", {}).get("condition", ""),
        exec_field=data.get("production", {}).get("exec"),
    )
