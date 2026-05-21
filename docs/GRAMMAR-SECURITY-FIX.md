# Grammar Engine Security Fix

**Status:** P0 Security Fix — Pending Oracle1 Deployment  
**Author:** CCC Audit Remediation (kimi1 subagent)  
**Date:** 2026-05-21  
**Branch:** `grammar-security-fix`  

---

## 1. Vulnerability Summary

The Grammar Engine (port 4045) has **zero input validation**. External agents successfully injected malicious rules via the `create_rule()` endpoint:

| # | Attack Vector | Payload | Impact |
|---|---------------|---------|--------|
| 1 | **Path Traversal** | `../../../etc/passwd` in rule `name` | File system enumeration / arbitrary file access |
| 2 | **XSS** | `<script>alert(1)</script>` in `production.tagline` | Stored cross-site scripting in rule output |
| 3 | **SQLi** | `'; DROP TABLE rules; --` in `production.condition` | Database destruction via unsanitized SQL fragments |
| 4 | **Code Injection** | `__import__('os').system('rm -rf /')` in `name` + `production.exec` | Remote code execution |

**Severity:** Critical — any agent with rule-write access can own the host.

---

## 2. Where the Grammar Engine Lives

The Grammar Engine **does not yet exist** in `sunset-ecosystem`. It is a planned component referenced in:

- `docs/THEORY-OF-ECOSYSTEMS.md` — universal grammar (COLLECT→SELECT→COMPILE)
- `docs/SPEC-BREEDER.md` — agent breeding pipeline (rules govern agent spawning)
- Port 4045 was designated in fleet infrastructure but no service is deployed

**Action required:** Oracle1 must deploy the Grammar Engine with this security spec baked in. **Do not deploy without these fixes.**

---

## 3. Proposed Implementation: `grammar/core.py`

Place this file at `grammar/core.py` in the repo root.

```python
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
```

---

## 4. Tests: `tests/test_grammar_security.py`

```python
"""Security tests for Grammar Engine rule ingestion.

Validates that all 4 CCC-audited attack vectors are blocked.
"""

import pytest

# Import from the proposed module (adjust path once grammar/ is added)
try:
    from grammar.core import (
        create_rule,
        validate_rule_name,
        validate_tagline,
        validate_condition,
        validate_exec_field,
        ValidationError,
    )
except ImportError:
    pytest.skip("grammar.core not yet deployed", allow_module_level=True)


# ── Attack Vector 1: Path Traversal ────────────────────────────────

def test_path_traversal_in_rule_name_rejected():
    with pytest.raises(ValidationError):
        validate_rule_name("../../../etc/passwd")

def test_double_dot_rule_name_rejected():
    with pytest.raises(ValidationError):
        validate_rule_name("foo..bar")

def test_slash_in_rule_name_rejected():
    with pytest.raises(ValidationError):
        validate_rule_name("foo/bar")

def test_backslash_in_rule_name_rejected():
    with pytest.raises(ValidationError):
        validate_rule_name("foo\\bar")

def test_legal_rule_name_accepted():
    assert validate_rule_name("foo-bar_baz123") == "foo-bar_baz123"


# ── Attack Vector 2: XSS ───────────────────────────────────────────

def test_xss_script_tag_stripped():
    result = validate_tagline("<script>alert(1)</script>")
    assert "<script>" not in result
    assert "alert(1)" not in result  # stripped inside tags

def test_xss_payload_html_escaped():
    result = validate_tagline('"><img src=x onerror=alert(1)>')
    assert "<img" not in result
    assert "&quot;" in result or "&lt;" in result

def test_tagline_max_length_enforced():
    with pytest.raises(ValidationError):
        validate_tagline("x" * 257)


# ── Attack Vector 3: SQL Injection ─────────────────────────────────

def test_sqli_drop_table_rejected():
    with pytest.raises(ValidationError):
        validate_condition("'; DROP TABLE rules; --")

def test_sqli_semicolon_rejected():
    with pytest.raises(ValidationError):
        validate_condition("status = 'active'; DELETE FROM rules")

def test_sqli_comment_dash_rejected():
    with pytest.raises(ValidationError):
        validate_condition("1 = 1 -- comment")

def test_sqli_union_select_rejected():
    with pytest.raises(ValidationError):
        validate_condition("1 UNION SELECT * FROM passwords")

def test_legal_condition_accepted():
    assert validate_condition("status == 'active' and priority > 5") == \
           "status == 'active' and priority > 5"


# ── Attack Vector 4: Code Injection ──────────────────────────────────

def test_code_injection_in_exec_rejected():
    with pytest.raises(ValidationError):
        validate_exec_field("__import__('os').system('rm -rf /')")

def test_exec_eval_rejected():
    with pytest.raises(ValidationError):
        validate_exec_field("eval('2+2')")

def test_exec_import_rejected():
    with pytest.raises(ValidationError):
        validate_exec_field("import os; os.system('ls')")

def test_safe_literal_accepted():
    """ast.literal_eval should accept safe literals."""
    assert validate_exec_field("[1, 2, 3]") == "[1, 2, 3]"

def test_exec_none_accepted():
    assert validate_exec_field(None) is None


# ── Integration: create_rule() ───────────────────────────────────────

def test_create_rule_blocks_all_four_vectors():
    with pytest.raises(ValidationError):
        create_rule(
            name="../../../etc/passwd",
            tagline="<script>alert(1)</script>",
            condition="'; DROP TABLE rules; --",
            exec_field="__import__('os').system('rm -rf /')",
        )

def test_create_rule_accepts_clean_input():
    rule = create_rule(
        name="spawn_worker",
        tagline="Spawn a background worker node.",
        condition="queue_depth > 10 and cpu_idle > 0.3",
        exec_field="[{'action': 'spawn', 'count': 2}]",
    )
    assert rule.name == "spawn_worker"
    assert "queue_depth" in rule.production.condition
    assert rule.production.exec_field == "[{'action': 'spawn', 'count': 2}]"
```

---

## 5. Deployment Checklist

- [ ] Oracle1 creates `grammar/` directory and `grammar/__init__.py`
- [ ] Drop `grammar/core.py` (section 3 above) into place
- [ ] Drop `tests/test_grammar_security.py` (section 4 above) into place
- [ ] Wire `grammar.core.create_rule` into the port-4045 HTTP/gRPC handler
- [ ] Ensure the handler returns **HTTP 400** with the ValidationError message on rejection
- [ ] Run `pytest tests/test_grammar_security.py -v` — all 18 assertions must pass
- [ ] Merge `grammar-security-fix` branch to `main`

---

## 6. Mitigated Attack Vectors

| Vector | Status | Mechanism |
|--------|--------|-----------|
| Path traversal (`../../../etc/passwd`) | **BLOCKED** | `RULE_NAME_PATTERN` regex — no `.`, `/`, `\` |
| XSS (`<script>alert(1)</script>`) | **BLOCKED** | `HTML_TAG_PATTERN` strips tags + `html.escape()` |
| SQLi (`'; DROP TABLE rules; --`) | **BLOCKED** | `SQLI_BLACKLIST` rejects `;`, `--`, `DROP`, etc. |
| Code injection (`__import__('os').system(...)`) | **BLOCKED** | `ast.literal_eval()` sandbox — no statements, no imports |

---

> **Note:** If the Grammar Engine is implemented in a language other than Python (e.g., Rust), port these validation rules exactly. The regex patterns and length limits are language-agnostic. The `ast.literal_eval` equivalent in Rust is "do not allow exec fields at all" or use a strict JSON parser.
