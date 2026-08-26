"""Tests for superinstance-runtime event bus and plugin system.

Covers:
    - Plugin registration (collector, selector, compiler)
    - Full COLLECT → SELECT → COMPILE pipeline
    - Error isolation (one plugin failing does not stop the bus)
    - Constraint plugin end-to-end
    - PLATO plugin end-to-end
"""

from __future__ import annotations

import pytest

from superinstance.runtime import (
    CompilerPlugin,
    CollectorPlugin,
    EventBus,
    EventResult,
    SelectorPlugin,
)
from superinstance.plugins.constraint import (
    ConstraintArtifact,
    ConstraintCollector,
    ConstraintCompiler,
    ConstraintSelector,
)
from superinstance.plugins.plato import (
    TileArtifact,
    PlatoCollector,
    PlatoCompiler,
    PlatoSelector,
)


# ═══════════════════════════════════════════════════════════════
# Minimal test plugins
# ═══════════════════════════════════════════════════════════════


class EchoCollector(CollectorPlugin):
    name = "echo-collector"

    def collect(self, context):
        return context.get("items", [])


class EchoSelector(SelectorPlugin):
    name = "echo-selector"

    def select(self, artifacts, context):
        threshold = context.get("threshold", 0)
        return [a for a in artifacts if isinstance(a, int) and a > threshold]


class EchoCompiler(CompilerPlugin):
    name = "echo-compiler"

    def compile(self, artifacts, context):
        return [{"value": a, "doubled": a * 2} for a in artifacts if isinstance(a, int)]


class BrokenCollector(CollectorPlugin):
    name = "broken-collector"

    def collect(self, context):
        raise RuntimeError("simulated failure")


# ═══════════════════════════════════════════════════════════════
# EventBus registration
# ═══════════════════════════════════════════════════════════════


class TestRegistration:
    def test_register_collector(self):
        bus = EventBus()
        bus.register_collector(EchoCollector())
        assert bus.plugin_names()["collect"] == ["echo-collector"]

    def test_register_selector(self):
        bus = EventBus()
        bus.register_selector(EchoSelector())
        assert bus.plugin_names()["select"] == ["echo-selector"]

    def test_register_compiler(self):
        bus = EventBus()
        bus.register_compiler(EchoCompiler())
        assert bus.plugin_names()["compile"] == ["echo-compiler"]

    def test_multiple_plugins_same_phase(self):
        bus = EventBus()
        bus.register_collector(EchoCollector())
        bus.register_collector(EchoCollector())
        assert len(bus.plugin_names()["collect"]) == 2

    def test_empty_bus_is_empty(self):
        bus = EventBus()
        result = bus.run()
        assert result.empty


# ═══════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════


class TestPipeline:
    def test_full_pipeline(self):
        bus = EventBus()
        bus.register_collector(EchoCollector())
        bus.register_selector(EchoSelector())
        bus.register_compiler(EchoCompiler())

        result = bus.run({"items": [1, 5, 10], "threshold": 3})

        assert result.collected == [1, 5, 10]
        assert result.selected == [5, 10]
        assert result.compiled == [
            {"value": 5, "doubled": 10},
            {"value": 10, "doubled": 20},
        ]
        assert not result.errors

    def test_no_selectors_passes_all(self):
        bus = EventBus()
        bus.register_collector(EchoCollector())
        bus.register_compiler(EchoCompiler())

        result = bus.run({"items": [1, 2]})
        assert result.selected == []
        # Compilers receive selected artifacts; with no selectors selected is empty
        assert result.compiled == []

    def test_no_compilers_returns_empty_compiled(self):
        bus = EventBus()
        bus.register_collector(EchoCollector())
        bus.register_selector(EchoSelector())

        result = bus.run({"items": [1, 2]})
        assert result.compiled == []


# ═══════════════════════════════════════════════════════════════
# Error isolation
# ═══════════════════════════════════════════════════════════════


class TestErrorIsolation:
    def test_broken_collector_does_not_stop_pipeline(self):
        bus = EventBus()
        bus.register_collector(BrokenCollector())
        bus.register_collector(EchoCollector())
        bus.register_selector(EchoSelector())
        bus.register_compiler(EchoCompiler())

        result = bus.run({"items": [7]})
        assert len(result.errors) == 1
        assert "broken-collector" in result.errors[0]
        assert result.collected == [7]
        assert result.compiled == [{"value": 7, "doubled": 14}]

    def test_broken_selector_caught_in_errors(self):
        class BrokenSelector(SelectorPlugin):
            name = "broken-selector"

            def select(self, artifacts, context):
                raise ValueError("boom")

        bus = EventBus()
        bus.register_collector(EchoCollector())
        bus.register_selector(BrokenSelector())

        result = bus.run({"items": [1, 2]})
        assert len(result.errors) == 1
        assert result.collected == [1, 2]


# ═══════════════════════════════════════════════════════════════
# Constraint plugin
# ═══════════════════════════════════════════════════════════════


class TestConstraintPlugin:
    def test_collector_extracts_artifacts(self):
        coll = ConstraintCollector()
        ctx = {
            "constraints": [
                {
                    "field": "chaos",
                    "value": 0.5,
                    "lower_bound": 0.0,
                    "upper_bound": 1.0,
                },
                {"field": "temp", "value": 1.2, "lower_bound": 0.0, "upper_bound": 1.0},
            ]
        }
        arts = coll.collect(ctx)
        assert len(arts) == 2
        assert arts[0].field == "chaos"
        assert not arts[0].violated
        assert arts[1].violated

    def test_selector_filters_violated(self):
        coll = ConstraintCollector()
        sel = ConstraintSelector()
        arts = coll.collect(
            {
                "constraints": [
                    {"field": "a", "value": 0.5, "lower_bound": 0, "upper_bound": 1},
                    {"field": "b", "value": 2.0, "lower_bound": 0, "upper_bound": 1},
                ]
            }
        )
        violated = sel.select(arts, {})
        assert len(violated) == 1
        assert violated[0].field == "b"

    def test_compiler_produces_directives(self):
        coll = ConstraintCollector()
        comp = ConstraintCompiler()
        arts = coll.collect(
            {
                "constraints": [
                    {
                        "field": "temp",
                        "value": 1.5,
                        "lower_bound": 0,
                        "upper_bound": 1.0,
                    },
                ]
            }
        )
        dirs = comp.compile(arts, {})
        assert len(dirs) == 1
        assert dirs[0]["action"] == "decrease"
        assert dirs[0]["field"] == "temp"

    def test_constraint_end_to_end(self):
        bus = EventBus()
        bus.register_collector(ConstraintCollector())
        bus.register_selector(ConstraintSelector())
        bus.register_compiler(ConstraintCompiler())

        result = bus.run(
            {
                "constraints": [
                    {"field": "x", "value": 0.5, "lower_bound": 0, "upper_bound": 1},
                    {"field": "y", "value": 2.0, "lower_bound": 0, "upper_bound": 1},
                    {"field": "z", "value": -1.0, "lower_bound": 0, "upper_bound": 1},
                ]
            }
        )

        assert len(result.collected) == 3
        assert len(result.selected) == 2  # y and z violated
        assert len(result.compiled) == 2
        assert all(d["plugin"] == "constraint" for d in result.compiled)


# ═══════════════════════════════════════════════════════════════
# PLATO plugin
# ═══════════════════════════════════════════════════════════════


class TestPlatoPlugin:
    def test_collector_extracts_tiles(self):
        coll = PlatoCollector()
        ctx = {
            "tiles": [
                {
                    "tile_id": "t1",
                    "room_id": 0,
                    "content": "hello",
                    "tags": ["greeting"],
                    "entropy": 0.9,
                },
            ]
        }
        arts = coll.collect(ctx)
        assert len(arts) == 1
        assert arts[0].tile_id == "t1"
        assert arts[0].entropy == 0.9

    def test_selector_filters_by_entropy(self):
        coll = PlatoCollector()
        sel = PlatoSelector()
        arts = coll.collect(
            {
                "tiles": [
                    {
                        "tile_id": "t1",
                        "room_id": 0,
                        "content": "a",
                        "tags": [],
                        "entropy": 0.9,
                    },
                    {
                        "tile_id": "t2",
                        "room_id": 1,
                        "content": "b",
                        "tags": [],
                        "entropy": 0.1,
                    },
                ]
            }
        )
        selected = sel.select(arts, {"entropy_threshold": 0.5})
        assert len(selected) == 1
        assert selected[0].tile_id == "t1"

    def test_compiler_produces_directives(self):
        coll = PlatoCollector()
        comp = PlatoCompiler()
        arts = coll.collect(
            {
                "tiles": [
                    {
                        "tile_id": "t1",
                        "room_id": 0,
                        "content": "hello world",
                        "tags": ["greeting"],
                        "entropy": 0.9,
                    },
                ]
            }
        )
        dirs = comp.compile(arts, {})
        assert len(dirs) == 1
        assert dirs[0]["action"] == "ingest"
        assert dirs[0]["tile_id"] == "t1"

    def test_plato_end_to_end(self):
        bus = EventBus()
        bus.register_collector(PlatoCollector())
        bus.register_selector(PlatoSelector())
        bus.register_compiler(PlatoCompiler())

        result = bus.run(
            {
                "tiles": [
                    {
                        "tile_id": "t1",
                        "room_id": 0,
                        "content": "high entropy",
                        "tags": ["a"],
                        "entropy": 0.9,
                    },
                    {
                        "tile_id": "t2",
                        "room_id": 1,
                        "content": "low entropy",
                        "tags": ["b"],
                        "entropy": 0.2,
                    },
                ],
                "entropy_threshold": 0.5,
            }
        )

        assert len(result.collected) == 2
        assert len(result.selected) == 1
        assert result.selected[0].tile_id == "t1"
        assert len(result.compiled) == 1
        assert result.compiled[0]["plugin"] == "plato"


# ═══════════════════════════════════════════════════════════════
# Partial phase execution
# ═══════════════════════════════════════════════════════════════


class TestPartialExecution:
    def test_run_collect_only(self):
        bus = EventBus()
        bus.register_collector(EchoCollector())
        assert bus.run_collect({"items": [1, 2]}) == [1, 2]

    def test_run_select_only(self):
        bus = EventBus()
        bus.register_selector(EchoSelector())
        assert bus.run_select([1, 5, 10], {"threshold": 3}) == [5, 10]

    def test_run_compile_only(self):
        bus = EventBus()
        bus.register_compiler(EchoCompiler())
        assert bus.run_compile([3, 4]) == [
            {"value": 3, "doubled": 6},
            {"value": 4, "doubled": 8},
        ]
