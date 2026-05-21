"""Security tests for Grammar Engine rule ingestion.

Validates that all 4 CCC-audited attack vectors are blocked.
"""

import pytest

from grammar.core import (
    create_rule,
    validate_condition,
    validate_exec_field,
    validate_rule_name,
    validate_tagline,
    ValidationError,
)


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
    # Inner text remains after tag stripping, but is harmless without tags
    assert result == "alert(1)"

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


# ── Integration: create_rule() ─────────────────────────────────────

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
