"""Superinstance Runtime — COLLECT → SELECT → COMPILE event bus.

The event bus implements a three-phase pipeline:
    1. COLLECT — gather raw artifacts from plugins (signals, constraints, tiles)
    2. SELECT  — filter and rank artifacts using plugin-specific logic
    3. COMPILE — transform selected artifacts into actionable outputs

Plugins register for one or more phases. The bus is intentionally
phase-separated so that plugins cannot skip the SELECT bottleneck.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── plugin interface ──────────────────────────────────────────


class Plugin(ABC):
    """Base class for all superinstance-runtime plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin identifier."""

    def health(self) -> dict[str, Any]:
        """Optional health check — return empty dict if healthy."""
        return {}


class CollectorPlugin(Plugin):
    """Phase-1 plugin: gather raw artifacts."""

    @abstractmethod
    def collect(self, context: dict[str, Any]) -> list[Any]:
        """Return a list of raw artifacts for the given context."""


class SelectorPlugin(Plugin):
    """Phase-2 plugin: filter and rank collected artifacts."""

    @abstractmethod
    def select(self, artifacts: list[Any], context: dict[str, Any]) -> list[Any]:
        """Return a filtered / ranked subset of *artifacts*."""


class CompilerPlugin(Plugin):
    """Phase-3 plugin: transform selected artifacts into outputs."""

    @abstractmethod
    def compile(self, artifacts: list[Any], context: dict[str, Any]) -> list[Any]:
        """Return compiled outputs."""


# ── data structures ───────────────────────────────────────────


@dataclass
class EventResult:
    """Immutable result of running the event bus."""

    collected: list[Any] = field(default_factory=list)
    selected: list[Any] = field(default_factory=list)
    compiled: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.collected or self.selected or self.compiled)


# ── event bus ─────────────────────────────────────────────────


class EventBus:
    """COLLECT → SELECT → COMPILE event bus.

    Usage::

        bus = EventBus()
        bus.register_collector(ConstraintCollector())
        bus.register_selector(ConstraintSelector())
        bus.register_compiler(ConstraintCompiler())

        result = bus.run({"room_id": 42, "signal": x})
    """

    def __init__(self) -> None:
        self._collectors: list[CollectorPlugin] = []
        self._selectors: list[SelectorPlugin] = []
        self._compilers: list[CompilerPlugin] = []

    # ── registration ────────────────────────────────────────

    def register_collector(self, plugin: CollectorPlugin) -> None:
        """Add a collector plugin."""
        self._collectors.append(plugin)
        logger.debug("Registered collector %s", plugin.name)

    def register_selector(self, plugin: SelectorPlugin) -> None:
        """Add a selector plugin."""
        self._selectors.append(plugin)
        logger.debug("Registered selector %s", plugin.name)

    def register_compiler(self, plugin: CompilerPlugin) -> None:
        """Add a compiler plugin."""
        self._compilers.append(plugin)
        logger.debug("Registered compiler %s", plugin.name)

    def plugin_names(self) -> dict[str, list[str]]:
        """Summary of registered plugins per phase."""
        return {
            "collect": [p.name for p in self._collectors],
            "select": [p.name for p in self._selectors],
            "compile": [p.name for p in self._compilers],
        }

    # ── execution ───────────────────────────────────────────

    def run(self, context: dict[str, Any] | None = None) -> EventResult:
        """Execute the full COLLECT → SELECT → COMPILE pipeline.

        Each phase runs all registered plugins in registration order.
        Errors from individual plugins are captured in *result.errors*
        and do not stop the pipeline.

        Args:
            context: Arbitrary dict passed to every plugin.

        Returns:
            :class:`EventResult` with artifacts from each phase.
        """
        context = context or {}
        errors: list[str] = []

        # 1. COLLECT
        collected: list[Any] = []
        for plugin in self._collectors:
            try:
                batch = plugin.collect(context)
                if batch:
                    collected.extend(batch)
            except Exception as exc:
                msg = f"Collector {plugin.name!r} failed: {exc}"
                logger.warning(msg)
                errors.append(msg)

        # 2. SELECT
        selected: list[Any] = []
        for plugin in self._selectors:
            try:
                batch = plugin.select(collected, context)
                if batch:
                    selected.extend(batch)
            except Exception as exc:
                msg = f"Selector {plugin.name!r} failed: {exc}"
                logger.warning(msg)
                errors.append(msg)

        # 3. COMPILE
        compiled: list[Any] = []
        for plugin in self._compilers:
            try:
                batch = plugin.compile(selected, context)
                if batch:
                    compiled.extend(batch)
            except Exception as exc:
                msg = f"Compiler {plugin.name!r} failed: {exc}"
                logger.warning(msg)
                errors.append(msg)

        return EventResult(
            collected=collected,
            selected=selected,
            compiled=compiled,
            errors=errors,
        )

    def run_collect(self, context: dict[str, Any] | None = None) -> list[Any]:
        """Run only the COLLECT phase."""
        context = context or {}
        collected: list[Any] = []
        for plugin in self._collectors:
            try:
                batch = plugin.collect(context)
                if batch:
                    collected.extend(batch)
            except Exception as exc:
                logger.warning("Collector %s failed: %s", plugin.name, exc)
        return collected

    def run_select(
        self,
        artifacts: list[Any],
        context: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Run only the SELECT phase."""
        context = context or {}
        selected: list[Any] = []
        for plugin in self._selectors:
            try:
                batch = plugin.select(artifacts, context)
                if batch:
                    selected.extend(batch)
            except Exception as exc:
                logger.warning("Selector %s failed: %s", plugin.name, exc)
        return selected

    def run_compile(
        self,
        artifacts: list[Any],
        context: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Run only the COMPILE phase."""
        context = context or {}
        compiled: list[Any] = []
        for plugin in self._compilers:
            try:
                batch = plugin.compile(artifacts, context)
                if batch:
                    compiled.extend(batch)
            except Exception as exc:
                logger.warning("Compiler %s failed: %s", plugin.name, exc)
        return compiled
