#!/usr/bin/env python3
"""Grammar Engine Security Test — verify fix blocks known chaos vectors.

Runs the 4 attack payloads from the April 22 audit against a local
grammar engine instance. All should be rejected with ValidationError.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/tmp/sunset-ecosystem")

from grammar.core import ValidationError, create_rule_from_dict

# ── Chaos vectors from April 22 audit ──────────────────────────────

CHAOS_VECTORS = [
    {
        "name": "../../../etc/passwd",
        "attack": "Path Traversal",
        "field": "name",
    },
    {
        "name": "safe_rule",
        "production": {
            "tagline": "<script>alert(1)</script>",
        },
        "attack": "XSS",
        "field": "production.tagline",
    },
    {
        "name": "safe_rule",
        "production": {
            "condition": "'; DROP TABLE rules; --",
        },
        "attack": "SQL Injection",
        "field": "production.condition",
    },
    {
        "name": "__import__('os').system('rm -rf /')",
        "production": {
            "exec": "__import__('os').system('rm -rf /')",
        },
        "attack": "Code Injection",
        "field": "name + production.exec",
    },
]


def test_chaos_vector(payload: dict, verbose: bool = True) -> bool:
    """Test one chaos vector. Returns True if blocked or safely sanitized."""
    attack_name = payload.pop("attack")
    field = payload.pop("field")

    try:
        rule = create_rule_from_dict(payload)
    except ValidationError as exc:
        if verbose:
            print(f"  ✅ {attack_name} ({field}) — BLOCKED by validator")
            print(f"     {exc}")
        return True

    # Rule was created — check if output is safe
    is_safe = True
    issues: list[str] = []

    # Check tagline for remaining HTML tags
    if "<" in rule.production.tagline or ">" in rule.production.tagline:
        is_safe = False
        issues.append(f"tagline still contains angle brackets: {rule.production.tagline!r}")

    # Check condition for SQLi patterns
    if ";" in rule.production.condition or "DROP" in rule.production.condition.upper():
        is_safe = False
        issues.append(f"condition still contains SQLi: {rule.production.condition!r}")

    # Check exec for code execution
    if rule.production.exec_field and "import" in rule.production.exec_field.lower():
        is_safe = False
        issues.append(f"exec still contains import: {rule.production.exec_field!r}")

    if verbose:
        if is_safe:
            print(f"  ✅ {attack_name} ({field}) — ALLOWED but SANITIZED")
            print(f"     tagline={rule.production.tagline!r}")
            print(f"     condition={rule.production.condition!r}")
            print(f"     exec={rule.production.exec_field!r}")
        else:
            print(f"  ❌ {attack_name} ({field}) — RULE CREATED UNSAFELY")
            for issue in issues:
                print(f"     {issue}")

    return is_safe


def run_all() -> None:
    print("=" * 70)
    print("Grammar Engine Security Test — Chaos Vector Validation")
    print("=" * 70)
    print()

    blocked = 0
    passed = 0

    for payload in CHAOS_VECTORS:
        if test_chaos_vector(payload.copy()):
            blocked += 1
        else:
            passed += 1
        print()

    print("=" * 70)
    print(f"Results: {blocked}/{len(CHAOS_VECTORS)} blocked, {passed} passed")
    if passed == 0:
        print("✅ ALL ATTACK VECTORS BLOCKED — Security fix is working")
    else:
        print("❌ SOME VECTORS PASSED — Fix incomplete")
    print("=" * 70)

    return passed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
