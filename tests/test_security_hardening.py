"""Tests for RuleValidator — security hardening for rule ingestion.

Covers path traversal, XSS, SQL injection, and code injection detection.
"""

from __future__ import annotations

import pytest

from grammar.security_hardening import (
    RuleValidator,
    ValidationError,
    RuleProvenance,
    create_rule_from_dict,
)


# ---------------------------------------------------------------------------
# RuleValidator — rule name validation
# ---------------------------------------------------------------------------


class TestRuleNameValidation:
    def test_safe_name(self):
        v = RuleValidator()
        assert v.validate_rule_name("safe_rule") == "safe_rule"
        assert v.validate_rule_name("_private") == "_private"
        assert v.validate_rule_name("CamelCase123") == "CamelCase123"

    def test_invalid_chars(self):
        v = RuleValidator()
        with pytest.raises(ValidationError):
            v.validate_rule_name("rule-with-dash")
        with pytest.raises(ValidationError):
            v.validate_rule_name("rule.with.dots")
        with pytest.raises(ValidationError):
            v.validate_rule_name("123starts_with_number")

    def test_path_traversal(self):
        v = RuleValidator()
        with pytest.raises(ValidationError):
            v.validate_rule_name("../../../etc/passwd")
        with pytest.raises(ValidationError):
            v.validate_rule_name("..\\windows\\system32")

    def test_code_injection_in_name(self):
        v = RuleValidator()
        with pytest.raises(ValidationError):
            v.validate_rule_name("__import__('os').system")

    def test_non_string(self):
        v = RuleValidator()
        with pytest.raises(ValidationError):
            v.validate_rule_name(123)

    def test_too_long(self):
        v = RuleValidator()
        with pytest.raises(ValidationError):
            v.validate_rule_name("a" * 129)

    def test_empty(self):
        v = RuleValidator()
        with pytest.raises(ValidationError):
            v.validate_rule_name("")


# ---------------------------------------------------------------------------
# Production fields
# ---------------------------------------------------------------------------


class TestProductionFields:
    def test_valid_production(self):
        v = RuleValidator()
        result = v.validate_production_fields(
            {"tagline": "Hello world", "condition": "x > 5"}
        )
        assert result["tagline"] == "Hello world"

    def test_xss_in_tagline(self):
        v = RuleValidator()
        with pytest.raises(ValidationError):
            v.validate_production_fields({"tagline": "<script>alert(1)</script>"})

    def test_sqli_in_condition(self):
        v = RuleValidator()
        with pytest.raises(ValidationError):
            v.validate_production_fields({"condition": "'; DROP TABLE users; --"})

    def test_code_injection_in_exec(self):
        v = RuleValidator()
        with pytest.raises(ValidationError):
            v.validate_production_fields({"exec": "import os; os.system('rm -rf /')"})

    def test_safe_exec(self):
        v = RuleValidator()
        result = v.validate_production_fields({"exec": "print('hello')"})
        assert result["exec"] == "print('hello')"

    def test_not_a_dict(self):
        v = RuleValidator()
        with pytest.raises(ValidationError):
            v.validate_production_fields("not a dict")

    def test_long_string(self):
        v = RuleValidator()
        with pytest.raises(ValidationError):
            v.validate_production_fields({"tagline": "a" * 4097})

    def test_null_bytes(self):
        v = RuleValidator()
        result = v.validate_production_fields({"tagline": "hello\x00world"})
        assert "\x00" not in result["tagline"]


# ---------------------------------------------------------------------------
# Full rule validation
# ---------------------------------------------------------------------------


class TestFullRuleValidation:
    def test_valid_rule(self):
        v = RuleValidator()
        rule = {
            "name": "safe_rule",
            "production": {"tagline": "Hello", "condition": "x > 0"},
        }
        result = v.validate_full_rule(rule)
        assert result["name"] == "safe_rule"
        assert "provenance" not in rule  # auto-tracked

    def test_invalid_name_in_rule(self):
        v = RuleValidator()
        rule = {"name": "../../../etc/passwd", "production": {}}
        with pytest.raises(ValidationError):
            v.validate_full_rule(rule)

    def test_provenance_tracking(self):
        v = RuleValidator()
        rule = {
            "name": "rule_with_provenance",
            "production": {},
            "provenance": {
                "creator": "test",
                "creator_type": "human",
                "timestamp": 1234567890.0,
            },
        }
        v.validate_full_rule(rule)
        log = v.get_provenance_log()
        assert len(log) == 1
        assert log[0].creator == "test"
        assert log[0].creator_type == "human"

    def test_default_provenance(self):
        v = RuleValidator()
        rule = {"name": "default_prov", "production": {}}
        v.validate_full_rule(rule)
        log = v.get_provenance_log()
        assert len(log) == 1
        assert log[0].creator == "unknown"
        assert log[0].creator_type == "external"

    def test_provenance_log_grows(self):
        v = RuleValidator()
        for i in range(3):
            rule = {"name": f"rule_{i}", "production": {}}
            v.validate_full_rule(rule)
        assert len(v.get_provenance_log()) == 3


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


class TestCreateRuleFromDict:
    def test_valid(self):
        rule = {"name": "ok", "production": {"tagline": "OK"}}
        result = create_rule_from_dict(rule)
        assert result["name"] == "ok"

    def test_invalid(self):
        with pytest.raises(ValidationError):
            create_rule_from_dict({"name": "<script>bad</script>"})


# ---------------------------------------------------------------------------
# RuleProvenance
# ---------------------------------------------------------------------------


class TestRuleProvenance:
    def test_fields(self):
        prov = RuleProvenance(
            creator="test", creator_type="human", timestamp=1.0, source_ip="127.0.0.1"
        )
        assert prov.creator == "test"
        assert prov.source_ip == "127.0.0.1"

    def test_immutable(self):
        prov = RuleProvenance(creator="test", creator_type="human", timestamp=1.0)
        with pytest.raises(AttributeError):
            prov.creator = "other"

    def test_optional_ip(self):
        prov = RuleProvenance(creator="test", creator_type="human", timestamp=1.0)
        assert prov.source_ip is None
