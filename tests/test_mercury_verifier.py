"""Tests for Mercury formula verifier.

Covers code generation, syntax validation, mock verification, batch
processing, and determinism classification.  Tests that require
`mmc` (Mercury compiler) are skipped if not installed.
"""

import pytest

from fleet.mercury_verifier import (
    FormulaToMercury,
    MercuryVerifier,
    BatchMercuryVerifier,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

class TestCodeGeneration:
    def test_number(self):
        gen = FormulaToMercury()
        code = gen.compile("=42")
        assert "42.0" in code
        assert ":- module" in code

    def test_string(self):
        gen = FormulaToMercury()
        code = gen.compile('="hello"')
        assert '"hello"' in code

    def test_if_then_else(self):
        gen = FormulaToMercury()
        code = gen.compile('=IF(1 < 2, 10, 20)')
        assert "if" in code
        assert "then" in code
        assert "else" in code

    def test_infix_add(self):
        gen = FormulaToMercury()
        code = gen.compile("=1 + 2")
        assert "(1.0 + 2.0)" in code

    def test_infix_comparison(self):
        gen = FormulaToMercury()
        code = gen.compile("=3 < 5")
        assert "(3.0 < 5.0)" in code

    def test_spawn_action(self):
        gen = FormulaToMercury()
        code = gen.compile('=SPAWN("scout")')
        assert '"SPAWN:scout"' in code

    def test_fleet_health_call(self):
        gen = FormulaToMercury()
        code = gen.compile("=FLEET_HEALTH()")
        assert "fleet_health_value()" in code

    def test_module_name_sanitized(self):
        gen = FormulaToMercury()
        code = gen.compile("=A+B*C")
        assert ":- module fleet_abc." in code

    def test_complex_policy(self):
        gen = FormulaToMercury()
        code = gen.compile(
            '=IF(AND(FLEET_HEALTH() > 0.5, THERMAL_AVG() < 0.8), SPAWN("worker"), IDLE())'
        )
        assert "fleet_health_value()" in code
        assert "thermal_avg_value()" in code
        assert '"SPAWN:worker"' in code
        assert '"IDLE"' in code

    def test_average_list(self):
        gen = FormulaToMercury()
        code = gen.compile("=AVERAGE(1, 2, 3)")
        assert "list.foldl" in code
        assert "list.length" in code

    def test_and_or_not(self):
        gen = FormulaToMercury()
        code = gen.compile("=AND(1, 0)")
        assert "and" in code
        code = gen.compile("=OR(1, 0)")
        assert "or" in code
        code = gen.compile("=NOT(1)")
        assert "not" in code

    def test_module_interface(self):
        gen = FormulaToMercury()
        code = gen.compile("=1")
        assert ":- interface." in code
        assert ":- pred evaluate(float::out) is det." in code
        assert ":- implementation." in code

    def test_compile_with_mode(self):
        gen = FormulaToMercury()
        code = gen.compile_with_mode("=1", mode="semidet")
        assert "is semidet." in code


# ---------------------------------------------------------------------------
# Mercury syntax validation
# ---------------------------------------------------------------------------

class TestSyntaxValidation:
    def test_valid_mercury_module_structure(self):
        gen = FormulaToMercury()
        code = gen.compile("=42")
        lines = code.splitlines()
        assert lines[0].startswith(":- module")
        assert any(":- interface." in line for line in lines)
        assert any(":- implementation." in line for line in lines)
        assert any("evaluate(Result)" in line for line in lines)

    def test_no_unclosed_parentheses(self):
        gen = FormulaToMercury()
        code = gen.compile("=IF(1 < 2, 10, 20)")
        assert code.count("(") == code.count(")")

    def test_quoting_in_strings(self):
        gen = FormulaToMercury()
        code = gen.compile('="hello world"')
        assert '"hello world"' in code


# ---------------------------------------------------------------------------
# MercuryVerifier (mock / without mmc)
# ---------------------------------------------------------------------------

class TestMercuryVerifier:
    def test_is_available_false_when_mmc_missing(self):
        verifier = MercuryVerifier(mmc_path="/nonexistent/mmc")
        assert not verifier.is_available()

    def test_analyze_returns_error_when_mmc_missing(self):
        verifier = MercuryVerifier(mmc_path="/nonexistent/mmc")
        gen = FormulaToMercury()
        code = gen.compile("=42")
        result = verifier.analyze(code)
        assert not result.success
        assert len(result.errors) >= 1
        assert "not found" in result.errors[0].lower()

    def test_check_formula_returns_result(self):
        verifier = MercuryVerifier(mmc_path="/nonexistent/mmc")
        result = verifier.check_formula("=42", expected="det")
        assert isinstance(result, VerificationResult)
        assert result.determinism == "unknown"

    def test_extract_determinism_det(self):
        raw = "some line is det other stuff"
        assert MercuryVerifier._extract_determinism(raw) == "det"

    def test_extract_determinism_semidet(self):
        raw = "predicate is semidet."
        assert MercuryVerifier._extract_determinism(raw) == "semidet"

    def test_extract_determinism_multi(self):
        raw = "predicate is multi."
        assert MercuryVerifier._extract_determinism(raw) == "multi"

    def test_extract_determinism_none(self):
        raw = "no determinism here"
        assert MercuryVerifier._extract_determinism(raw) is None

    def test_extract_errors(self):
        raw = "foo.m:42: error: undefined predicate\nbar.m:7: warning: unused var"
        errors = MercuryVerifier._extract_errors(raw)
        assert len(errors) == 1
        assert "undefined predicate" in errors[0]

    def test_extract_warnings(self):
        raw = "foo.m:7: warning: unused variable X\nfoo.m:8: error: bad type"
        warnings = MercuryVerifier._extract_warnings(raw)
        assert len(warnings) == 1
        assert "unused variable" in warnings[0]


# ---------------------------------------------------------------------------
# Batch verifier
# ---------------------------------------------------------------------------

class TestBatchVerifier:
    def test_verify_batch(self):
        verifier = MercuryVerifier(mmc_path="/nonexistent/mmc")
        batch = BatchMercuryVerifier(verifier)
        formulas = ["=42", "=1 + 2"]
        results = batch.verify_batch(formulas)
        assert len(results) == 2
        assert "=42" in results

    def test_classify_formulas(self):
        verifier = MercuryVerifier(mmc_path="/nonexistent/mmc")
        batch = BatchMercuryVerifier(verifier)
        formulas = ["=42", "=1 + 2"]
        classification = batch.classify_formulas(formulas)
        assert classification["=42"] == "unknown"

    def test_filter_safe_empty(self):
        verifier = MercuryVerifier(mmc_path="/nonexistent/mmc")
        batch = BatchMercuryVerifier(verifier)
        safe = batch.filter_safe(["=42"])
        assert safe == []  # all unknown, none safe


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_formula(self):
        gen = FormulaToMercury()
        with pytest.raises(Exception):  # SyntaxError from parser
            gen.compile("=")

    def test_division_by_zero_formula(self):
        gen = FormulaToMercury()
        code = gen.compile("=1 / 0")
        assert "(1.0 / 0.0)" in code

    def test_nested_if(self):
        gen = FormulaToMercury()
        code = gen.compile('=IF(1 < 2, IF(3 < 4, "a", "b"), "c")')
        # Count only in the body, not the module name
        body = code.split("evaluate(Result) :-")[-1]
        assert body.count("if") == 2
        assert body.count("then") == 2
        assert body.count("else") == 2

    def test_long_formula_module_name_truncated(self):
        gen = FormulaToMercury()
        long_formula = "=" + "A" * 100
        code = gen.compile(long_formula)
        # Module name should be truncated
        module_line = [l for l in code.splitlines() if l.startswith(":- module")][0]
        assert len(module_line) < 50

    def test_special_chars_in_module_name(self):
        gen = FormulaToMercury()
        code = gen.compile("=IF(1+2*3/4-5<6)")
        assert ":- module" in code

    def test_result_data_class(self):
        r = VerificationResult(
            determinism="det",
            errors=[],
            warnings=[],
            mercury_code="dummy",
            success=True,
        )
        assert r.determinism == "det"
        assert r.success
