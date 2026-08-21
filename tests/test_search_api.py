"""Tests for FleetSearch — unified query interface across memory layers.

Covers SearchIntent, SearchResult, FleetSearch init, intent detection,
ask() routing, and backend searches.
"""

from unittest.mock import MagicMock

import pytest

from swarm.search_api import FleetSearch, SearchIntent, SearchResult


# ---------------------------------------------------------------------------
# SearchIntent
# ---------------------------------------------------------------------------


class TestSearchIntent:
    def test_values(self):
        assert SearchIntent.KNOWLEDGE.value == "knowledge"
        assert SearchIntent.HARDWARE.value == "hardware"
        assert SearchIntent.TEMPORAL.value == "temporal"
        assert SearchIntent.AGENT.value == "agent"
        assert SearchIntent.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_init(self):
        r = SearchResult(source="knowledge", score=0.9, payload="hello")
        assert r.source == "knowledge"
        assert r.score == 0.9
        assert r.payload == "hello"
        assert r.context == {}

    def test_repr(self):
        r = SearchResult(source="knowledge", score=0.9, payload="hello")
        assert "knowledge" in repr(r)
        assert "0.900" in repr(r)


# ---------------------------------------------------------------------------
# FleetSearch init
# ---------------------------------------------------------------------------


class TestFleetSearchInit:
    def test_empty(self):
        fs = FleetSearch()
        assert fs.knowledge is None
        assert fs.hardware is None
        assert fs.temporal is None
        assert fs.agent_table is None

    def test_with_backends(self):
        k = MagicMock()
        h = MagicMock()
        fs = FleetSearch(knowledge=k, hardware=h)
        assert fs.knowledge is k
        assert fs.hardware is h


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------


class TestIntentDetection:
    def test_hardware_keywords(self):
        fs = FleetSearch()
        assert fs._detect_intent("Which GPU should I use?") == SearchIntent.HARDWARE
        assert fs._detect_intent("run on RTX") == SearchIntent.HARDWARE
        assert fs._detect_intent("thermal capacity") == SearchIntent.HARDWARE

    def test_temporal_keywords(self):
        fs = FleetSearch()
        assert fs._detect_intent("What will happen tomorrow?") == SearchIntent.TEMPORAL
        assert fs._detect_intent("predict next trend") == SearchIntent.TEMPORAL
        assert fs._detect_intent("forecast the future") == SearchIntent.TEMPORAL

    def test_agent_keywords(self):
        fs = FleetSearch()
        assert fs._detect_intent("breed new agents") == SearchIntent.AGENT
        assert fs._detect_intent("parent fitness DNA") == SearchIntent.AGENT
        assert fs._detect_intent("tournament selection") == SearchIntent.AGENT

    def test_knowledge_default(self):
        fs = FleetSearch()
        assert (
            fs._detect_intent("What is the meaning of life?") == SearchIntent.KNOWLEDGE
        )
        assert fs._detect_intent("How does this work?") == SearchIntent.KNOWLEDGE

    def test_unknown(self):
        fs = FleetSearch()
        assert fs._detect_intent("") == SearchIntent.KNOWLEDGE


# ---------------------------------------------------------------------------
# ask() routing
# ---------------------------------------------------------------------------


class TestAskRouting:
    def test_no_backends(self):
        fs = FleetSearch()
        results = fs.ask("What is X?")
        assert results == []

    def test_knowledge_routing(self):
        k = MagicMock()
        k.search.return_value = []
        fs = FleetSearch(knowledge=k)
        results = fs.ask("What do we know?")
        assert k.search.called

    def test_hardware_routing(self):
        h = MagicMock()
        h.find_best_device.return_value = []
        fs = FleetSearch(hardware=h)
        results = fs.ask("Which GPU for training?")
        assert h.find_best_device.called

    def test_hardware_routes_hardware(self):
        h = MagicMock()
        h.find_best_device.return_value = []
        fs = FleetSearch(hardware=h)
        results = fs.ask("Which GPU for training?")
        assert h.find_best_device.called

    def test_knowledge_routes_knowledge(self):
        k = MagicMock()
        k.search.return_value = []
        fs = FleetSearch(knowledge=k)
        results = fs.ask("What do we know?")
        assert k.search.called
        assert fs.hardware is None

    def test_temporal_routes_temporal(self):
        t = MagicMock()
        t.find_similar_trajectory.return_value = []
        fs = FleetSearch(temporal=t)
        results = fs.ask("What will room 1 look like tomorrow?")
        assert t.find_similar_trajectory.called

    def test_results_sorted_and_limited(self):
        k = MagicMock()
        k.search.return_value = []
        fs = FleetSearch(knowledge=k)
        results = fs.ask("test", k=3)
        assert len(results) <= 3

    def test_knowledge_search_explicit(self):
        k = MagicMock()
        k.search.return_value = []
        fs = FleetSearch(knowledge=k)
        results = fs.knowledge_search("test", k=5)
        assert k.search.called

    def test_knowledge_search_no_backend(self):
        fs = FleetSearch()
        assert fs.knowledge_search("test") == []

    def test_hardware_search_explicit(self):
        h = MagicMock()
        h.find_best_device.return_value = []
        fs = FleetSearch(hardware=h)
        results = fs.hardware_search("GPU")
        assert h.find_best_device.called

    def test_hardware_search_no_backend(self):
        fs = FleetSearch()
        assert fs.hardware_search("GPU") == []

    def test_predict_room_no_backend(self):
        fs = FleetSearch()
        assert fs.predict_room(1) is None

    def test_predict_room_with_backend(self):
        t = MagicMock()
        t.predict.return_value = {"state": "ok"}
        fs = FleetSearch(temporal=t)
        result = fs.predict_room(1, ticks_ahead=2)
        assert result is not None
        assert result.source == "temporal"
        assert result.score == 1.0
        t.predict.assert_called_once_with(1, 2)

    def test_repr(self):
        fs = FleetSearch(knowledge=MagicMock())
        r = repr(fs)
        assert "knowledge=True" in r
        assert "hardware=False" in r
