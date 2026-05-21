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


# ── Trinity Scoring ────────────────────────────────────────────────

from typing import Any, Dict, Tuple


# Allowed AST node types for condition evaluation
_CONDITION_AST_WHITELIST = frozenset({
    ast.Expression, ast.BoolOp, ast.And, ast.Or,
    ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Name, ast.Load, ast.Constant,
})


def score_rule(
    rule: Rule,
    metrics: Dict[str, float],
    trinity_weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Evaluate a single rule against a metrics context and compute a Trinity Score.

    Parses the rule's `condition` string into an AST-safe boolean expression,
    evaluates it against the provided metrics, and derives ethos, pathos, and
    logos from the match quality and rule provenance.

    Args:
        rule: A validated Rule object (from `create_rule()`).
        metrics: Key-value pairs of metric names to numeric values.
        trinity_weights: Optional override for the Trinity axis weights.
            Defaults to `{"ethos": 0.33, "pathos": 0.33, "logos": 0.34}`.

    Returns:
        dict with keys: matched, ethos, pathos, logos, trinity, condition_ast.

    Raises:
        ValidationError: If the condition references unknown metrics or contains
                         operators outside the allowed AST whitelist.
        ValueError: If metrics is empty or contains non-numeric values.
    """
    if not metrics:
        raise ValueError("metrics must not be empty.")
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            raise ValueError(f"Metric '{k}' must be numeric, got {type(v).__name__}")

    weights = trinity_weights or {"ethos": 0.33, "pathos": 0.33, "logos": 0.34}

    # Parse condition AST
    condition = rule.production.condition
    matched = False
    condition_ast = None

    if condition:
        try:
            tree = ast.parse(condition, mode="eval")
        except SyntaxError as exc:
            raise ValidationError(f"Condition is not a valid expression: {exc}") from exc

        # Validate AST whitelist
        for node in ast.walk(tree):
            if type(node) not in _CONDITION_AST_WHITELIST:
                raise ValidationError(
                    f"Condition contains unsupported operator: {type(node).__name__}"
                )

        condition_ast = tree

        # Evaluate with restricted globals/locals
        try:
            result = eval(
                compile(tree, "<condition>", "eval"),
                {"__builtins__": {}},
                metrics,
            )
            matched = bool(result)
        except NameError as exc:
            raise ValidationError(
                f"Condition references unknown metric: {exc}"
            ) from exc
        except Exception as exc:
            raise ValidationError(
                f"Condition evaluation failed: {exc}"
            ) from exc

    # Trinity heuristics
    ethos = 0.9
    if matched and "thermal_headroom" in metrics:
        ethos = min(1.0, metrics["thermal_headroom"] / 10.0)

    pathos = 1.0 if matched else 0.1

    logos = 1.0
    if condition:
        syntax_errors = 0  # simplified heuristic
        logos = 1.0 - (syntax_errors / max(1, len(condition)))

    trinity = ethos * pathos * logos

    return {
        "matched": matched,
        "ethos": round(ethos, 4),
        "pathos": round(pathos, 4),
        "logos": round(logos, 4),
        "trinity": round(trinity, 4),
        "condition_ast": condition_ast,
    }


# ── Batch Operations ───────────────────────────────────────────────

def batch_create_rules(rule_dicts: list) -> Tuple[list[Rule], list[ValidationError]]:
    """Validate a batch of rule dicts, returning successes and failures separately.

    Args:
        rule_dicts: List of canonical JSON-form dicts (see §3.2).

    Returns:
        Tuple of (validated_rules, errors). Errors preserve the index of the
        offending dict for forensic correlation.
    """
    rules: list[Rule] = []
    errors: list[ValidationError] = []

    for idx, data in enumerate(rule_dicts):
        try:
            rules.append(create_rule_from_dict(data))
        except ValidationError as exc:
            exc.__dict__["index"] = idx
            errors.append(exc)

    return rules, errors


# ── Evolution ────────────────────────────────────────────────────

import random
import copy
import re as _re_module


def _extract_threshold(condition: str) -> float | None:
    """Extract a numeric threshold from a comparison condition."""
    m = _re_module.search(r"(<>|!=|>=|<=|>|<|=)\s*([0-9]+\.?[0-9]*)", condition)
    if m:
        return float(m.group(2))
    return None


def _mutate_condition(condition: str, sigma: float = 0.05) -> str:
    """Apply Gaussian mutation to numeric thresholds in a condition."""
    threshold = _extract_threshold(condition)
    if threshold is None:
        return condition
    delta = random.gauss(0, sigma * threshold)
    new_val = max(0.0, threshold + delta)
    # Replace the old threshold with the new one
    def repl(m):
        op = m.group(1)
        return f"{op} {new_val:.4g}"
    return _re_module.sub(r"(<>|!=|>=|<=|>|<|=)\s*[0-9]+\.?[0-9]*", repl, condition, count=1)


def evolve(
    population: list[Rule],
    scores: list[Dict[str, Any]],
    num_children: int | None = None,
    mutation_sigma: float = 0.05,
) -> list[Rule]:
    """Run one generation of rule evolution: selection → crossover → mutation → validation.

    Args:
        population: Current generation of validated Rules.
        scores: Output of `score_rule()` for each rule in the population,
                in the same order. Must contain `trinity` and `matched` keys.
        num_children: How many children to produce. Defaults to
                      `len(population) // 2`.
        mutation_sigma: Gaussian mutation std-dev for numeric thresholds.

    Returns:
        list[Rule]: The next generation — winners retained + children.

    Raises:
        ValidationError: If a child rule fails re-validation after mutation.
        ValueError: If population and scores lengths mismatch.
    """
    if len(population) != len(scores):
        raise ValueError("population and scores must have the same length.")

    if not population:
        return []

    num_children = num_children or (len(population) // 2)

    # Identify Pareto frontier (non-dominated agents)
    def dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        return (
            a["ethos"] >= b["ethos"]
            and a["pathos"] >= b["pathos"]
            and a["logos"] >= b["logos"]
            and (
                a["ethos"] > b["ethos"]
                or a["pathos"] > b["pathos"]
                or a["logos"] > b["logos"]
            )
        )

    frontier_idx = []
    for i, si in enumerate(scores):
        dominated = False
        for j, sj in enumerate(scores):
            if i != j and dominates(sj, si):
                dominated = True
                break
        if not dominated:
            frontier_idx.append(i)

    winners = [population[i] for i in frontier_idx]
    if not winners:
        # Fallback: keep top half by trinity
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i]["trinity"], reverse=True)
        winners = [population[i] for i in sorted_idx[:max(1, len(population) // 2)]]

    # Breed children
    children: list[Rule] = []
    generation = max(
        (int(_re_module.search(r"gen(\d+)", r.name).group(1)) for r in winners if _re_module.search(r"gen(\d+)", r.name)),
        default=0,
    ) + 1

    for _ in range(num_children):
        if len(winners) < 2:
            parent_a = parent_b = winners[0]
        else:
            parent_a, parent_b = random.sample(winners, 2)

        child_name = f"{parent_a.name}_{parent_b.name}_gen{generation}"
        child_tagline = (parent_a.production.tagline + " " + parent_b.production.tagline)[:TAGLINE_MAX_LEN]

        # Crossover condition: mutate threshold of parent_a's condition
        child_condition = _mutate_condition(parent_a.production.condition, mutation_sigma)
        if not child_condition:
            child_condition = parent_b.production.condition

        # Deep merge exec payloads (if both are literals)
        child_exec = parent_a.production.exec_field
        if parent_a.production.exec_field and parent_b.production.exec_field:
            try:
                a_lit = ast.literal_eval(parent_a.production.exec_field)
                b_lit = ast.literal_eval(parent_b.production.exec_field)
                if isinstance(a_lit, dict) and isinstance(b_lit, dict):
                    merged = copy.deepcopy(a_lit)
                    merged.update(b_lit)
                    child_exec = str(merged)
            except (ValueError, SyntaxError):
                pass

        # Validate child
        try:
            child = create_rule(
                name=child_name,
                tagline=child_tagline,
                condition=child_condition,
                exec_field=child_exec,
            )
            children.append(child)
        except ValidationError:
            # Skip invalid children
            continue

    return winners + children


# ── Batch / JSON ingestion helper ──────────────────────────────────

def create_rule_from_dict(data: dict) -> Rule:
    """Convenience wrapper for JSON/rule-dict ingestion."""
    return create_rule(
        name=data.get("name", ""),
        tagline=data.get("production", {}).get("tagline", ""),
        condition=data.get("production", {}).get("condition", ""),
        exec_field=data.get("production", {}).get("exec"),
    )

