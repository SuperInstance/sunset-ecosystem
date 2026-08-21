"""Tests for Code Review Personas.

Covers 5 AST-based personas: Correctness, Security, Performance, Simplicity, Adversarial.
"""

import pytest

from fleet.review_code import (
    ReviewPersona,
    ReviewFinding,
    review_code,
    review_all,
    CorrectnessVisitor,
    SecurityVisitor,
    PerformanceVisitor,
    SimplicityVisitor,
    AdversarialVisitor,
)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


class TestCorrectness:
    def test_bare_except(self):
        code = """
try:
    x = 1
except:
    pass
"""
        findings = review_code(code, ReviewPersona.CORRECTNESS)
        assert any("Bare except" in f.message for f in findings)

    def test_mutable_default(self):
        code = """
def foo(a=[]):
    return a
"""
        findings = review_code(code, ReviewPersona.CORRECTNESS)
        assert any("Mutable default" in f.message for f in findings)
        assert any(f.severity == "critical" for f in findings)

    def test_is_with_literal(self):
        code = """
x = "hello" is "world"
"""
        findings = review_code(code, ReviewPersona.CORRECTNESS)
        assert any("'is' with literal" in f.message for f in findings)

    def test_dict_call(self):
        code = """
x = dict()
"""
        findings = review_code(code, ReviewPersona.CORRECTNESS)
        assert any("dict()" in f.message for f in findings)

    def test_no_issues_clean_code(self):
        code = """
def foo(a):
    return a + 1
"""
        findings = review_code(code, ReviewPersona.CORRECTNESS)
        assert len(findings) == 0 or all(f.severity == "info" for f in findings)


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_eval_call(self):
        code = """
x = eval(user_input)
"""
        findings = review_code(code, ReviewPersona.SECURITY)
        assert any("eval" in f.message for f in findings)
        assert any(f.severity == "critical" for f in findings)

    def test_exec_call(self):
        code = """
exec("import os")
"""
        findings = review_code(code, ReviewPersona.SECURITY)
        assert any("exec" in f.message for f in findings)

    def test_sql_formatting(self):
        code = """
query = "SELECT * FROM users WHERE id = %s" % user_id
"""
        findings = review_code(code, ReviewPersona.SECURITY)
        assert any("SQL injection" in f.message for f in findings)

    def test_sql_fstring(self):
        code = """
query = f"SELECT * FROM users WHERE id = {user_id}"
"""
        findings = review_code(code, ReviewPersona.SECURITY)
        assert any("SQL injection" in f.message for f in findings)

    def test_hardcoded_password(self):
        code = """
password = "secret123"
"""
        findings = review_code(code, ReviewPersona.SECURITY)
        assert any("Hardcoded password" in f.message for f in findings)

    def test_hardcoded_api_key(self):
        code = """
api_key = "sk-1234567890"
"""
        findings = review_code(code, ReviewPersona.SECURITY)
        assert any("Hardcoded API key" in f.message for f in findings)

    def test_pickle_import(self):
        code = """
from pickle import loads
"""
        findings = review_code(code, ReviewPersona.SECURITY)
        assert any("Pickle" in f.message for f in findings)

    def test_input_in_loop(self):
        code = """
for i in range(10):
    x = input()
"""
        findings = review_code(code, ReviewPersona.SECURITY)
        assert any("input() inside loop" in f.message for f in findings)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_nested_loop(self):
        code = """
for i in range(10):
    for j in range(10):
        for k in range(10):
            pass
"""
        findings = review_code(code, ReviewPersona.PERFORMANCE)
        assert any("Nested loop" in f.message for f in findings)

    def test_list_comp_in_loop(self):
        code = """
for i in range(10):
    x = [j for j in range(10)]
"""
        findings = review_code(code, ReviewPersona.PERFORMANCE)
        assert any("List comprehension inside loop" in f.message for f in findings)

    def test_many_generators(self):
        code = """
x = [i for i in range(10) for j in range(10) for k in range(10)]
"""
        findings = review_code(code, ReviewPersona.PERFORMANCE)
        assert any("generators" in f.message for f in findings)

    def test_repeated_function_calls(self):
        code = """
def foo():
    a = bar()
    b = bar()
    c = bar()
    d = bar()
    e = bar()
    f = bar()
    g = bar()
"""
        findings = review_code(code, ReviewPersona.PERFORMANCE)
        assert any("bar" in f.message and "caching" in f.message for f in findings)


# ---------------------------------------------------------------------------
# Simplicity
# ---------------------------------------------------------------------------


class TestSimplicity:
    def test_long_function(self):
        code = "\n".join(["def foo():"] + ["    x = 1"] * 60 + ["    return x"])
        findings = review_code(code, ReviewPersona.SIMPLICITY)
        assert any("lines" in f.message for f in findings)

    def test_very_long_function(self):
        code = "\n".join(["def foo():"] + ["    x = 1"] * 110 + ["    return x"])
        findings = review_code(code, ReviewPersona.SIMPLICITY)
        assert any(f.severity == "critical" for f in findings)

    def test_many_parameters(self):
        code = """
def foo(a, b, c, d, e, f, g, h, i):
    pass
"""
        findings = review_code(code, ReviewPersona.SIMPLICITY)
        assert any("parameters" in f.message for f in findings)

    def test_static_only_class(self):
        code = """
class Foo:
    @staticmethod
    def bar():
        return 1
    @staticmethod
    def baz():
        return 2
"""
        findings = review_code(code, ReviewPersona.SIMPLICITY)
        assert any("static methods" in f.message for f in findings)

    def test_empty_class(self):
        code = """
class Foo:
    pass
"""
        findings = review_code(code, ReviewPersona.SIMPLICITY)
        assert any("Empty class" in f.message for f in findings)

    def test_deep_if_chain(self):
        code = """
if x == 1:
    pass
elif x == 2:
    pass
elif x == 3:
    pass
elif x == 4:
    pass
elif x == 5:
    pass
"""
        findings = review_code(code, ReviewPersona.SIMPLICITY)
        assert any("Deep if-elif" in f.message for f in findings)

    def test_while_true_no_break(self):
        code = """
while True:
    pass
"""
        findings = review_code(code, ReviewPersona.SIMPLICITY)
        assert any("infinite loop" in f.message for f in findings)

    def test_bare_pass_in_except(self):
        code = """
try:
    x = 1
except Exception:
    pass
"""
        findings = review_code(code, ReviewPersona.SIMPLICITY)
        assert any("Bare pass in except" in f.message for f in findings)


# ---------------------------------------------------------------------------
# Adversarial
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_no_validation(self):
        code = """
def foo(a, b):
    return a / b
"""
        findings = review_code(code, ReviewPersona.ADVERSARIAL)
        assert any("input validation" in f.message for f in findings)

    def test_division_by_variable(self):
        code = """
def foo(a, b):
    return a / b
"""
        findings = review_code(code, ReviewPersona.ADVERSARIAL)
        assert any("ZeroDivisionError" in f.message for f in findings)

    def test_negative_index(self):
        code = """
x = arr[-1]
"""
        findings = review_code(code, ReviewPersona.ADVERSARIAL)
        assert any("Negative index" in f.message for f in findings)

    def test_no_all_export(self):
        code = """
def foo():
    pass
class Bar:
    pass
"""
        findings = review_code(code, ReviewPersona.ADVERSARIAL)
        assert any("__all__" in f.message for f in findings)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_review_all(self):
        code = """
def foo(a=[]):
    x = eval("1")
    return x
"""
        results = review_all(code)
        assert "Correctness" in results
        assert "Security" in results
        assert len(results) == 5

    def test_review_all_counts(self):
        code = """
def foo(a=[]):
    x = eval("1")
    return x
"""
        results = review_all(code)
        correctness = results["Correctness"]
        security = results["Security"]
        assert any("Mutable default" in f.message for f in correctness)
        assert any("eval" in f.message for f in security)

    def test_syntax_error(self):
        code = "def foo(\n"
        findings = review_code(code, ReviewPersona.CORRECTNESS)
        assert any("Syntax error" in f.message for f in findings)

    def test_empty_code(self):
        code = ""
        findings = review_code(code, ReviewPersona.CORRECTNESS)
        assert len(findings) == 0 or all(f.severity == "info" for f in findings)


# ---------------------------------------------------------------------------
# Visitor direct
# ---------------------------------------------------------------------------


class TestVisitorsDirect:
    def test_correctness_visitor(self):
        v = CorrectnessVisitor()
        findings = v.run("def foo(a=[]): pass")
        assert any("Mutable default" in f.message for f in findings.findings)

    def test_security_visitor(self):
        v = SecurityVisitor()
        findings = v.run("x = eval('1')")
        assert any("eval" in f.message for f in findings.findings)

    def test_performance_visitor(self):
        v = PerformanceVisitor()
        code = """
for i in range(10):
    for j in range(10):
        for k in range(10):
            pass
"""
        findings = v.run(code)
        assert any("Nested loop" in f.message for f in findings.findings)

    def test_simplicity_visitor(self):
        v = SimplicityVisitor()
        code = "\n".join(["def foo():"] + ["    x = 1"] * 60 + ["    return x"])
        findings = v.run(code)
        assert any("lines" in f.message for f in findings.findings)

    def test_adversarial_visitor(self):
        v = AdversarialVisitor()
        findings = v.run("def foo(a, b): return a / b")
        assert any("ZeroDivisionError" in f.message for f in findings.findings)
