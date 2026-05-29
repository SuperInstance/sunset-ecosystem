"""Lightweight template engine for fleet configuration rendering.

Simple string templating with variable substitution, conditionals,
and loops. No external dependencies. Used for rendering fleet config
files, room descriptions, and agent prompt templates.

Usage:
    engine = TemplateEngine()
    result = engine.render("Hello {{ name }}!", {"name": "Fleet"})
    # result == "Hello Fleet!"
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


class TemplateError(Exception):
    pass


class TemplateEngine:
    """
    Lightweight template renderer.

    Supports:
    - {{ var }} — variable interpolation
    - {% if var %} ... {% endif %} — conditionals
    - {% for item in list %} ... {% endfor %} — loops
    """

    def __init__(self):
        self._block_re = re.compile(
            r"{%\s*(if|for)\s+(.+?)\s*%}(.*?){%\s*end\1\s*%}",
            re.DOTALL,
        )
        self._var_re = re.compile(r"{{\s*(\w+)\s*}}")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, template: str, context: Dict[str, Any]) -> str:
        """Render a template with a context dictionary."""
        result = self._render_blocks(template, context)
        result = self._render_vars(result, context)
        return result

    def render_file(self, path: str, context: Dict[str, Any]) -> str:
        """Read a file and render it."""
        with open(path, "r", encoding="utf-8") as f:
            return self.render(f.read(), context)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _render_blocks(self, template: str, context: Dict[str, Any]) -> str:
        result = template
        for _ in range(100):  # safety limit
            new_result = self._block_re.sub(
                lambda m: self._replace_block(m, context), result
            )
            if new_result == result:
                break
            result = new_result
        return result

    def _replace_block(self, match: Any, context: Dict[str, Any]) -> str:
        kind = match.group(1)
        expr = match.group(2).strip()
        body = match.group(3)

        if kind == "if":
            condition = self._eval_condition(expr, context)
            return body if condition else ""

        if kind == "for":
            parts = expr.split(" in ")
            if len(parts) != 2:
                raise TemplateError(f"Invalid for loop: {expr}")
            var_name = parts[0].strip()
            list_name = parts[1].strip()
            items = context.get(list_name, [])
            if not isinstance(items, list):
                raise TemplateError(f"Expected list for {list_name}, got {type(items)}")
            rendered = []
            for item in items:
                sub_ctx = dict(context)
                sub_ctx[var_name] = item
                rendered.append(self._render_vars(body, sub_ctx))
            return "".join(rendered)

        return ""

    def _render_vars(self, template: str, context: Dict[str, Any]) -> str:
        def replace_var(match: re.Match) -> str:
            var = match.group(1)
            if var in context:
                return str(context[var])
            return match.group(0)  # leave unreplaced

        return self._var_re.sub(replace_var, template)

    def _eval_condition(self, expr: str, context: Dict[str, Any]) -> bool:
        """Evaluate a simple conditional expression."""
        expr = expr.strip()
        # not var
        if expr.startswith("not "):
            return not bool(context.get(expr[4:].strip()))
        # var
        return bool(context.get(expr, False))

    def __repr__(self) -> str:
        return "<TemplateEngine>"
