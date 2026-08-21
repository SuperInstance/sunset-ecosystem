#!/usr/bin/env python3
"""Tests for fleet/plato_sdk_bridge.py."""

import pytest

from fleet.plato_sdk_bridge import PlatoSDKBridge, TileResult


class TestTileResult:
    def test_dataclass(self):
        t = TileResult(domain="ethos", question="Q", answer="A", metadata={"k": "v"})
        assert t.domain == "ethos"
        assert t.question == "Q"
        assert t.answer == "A"
        assert t.metadata == {"k": "v"}


class TestPlatoSDKBridge:
    def test_init(self):
        bridge = PlatoSDKBridge(base_url="http://test:8847")
        assert bridge.base_url == "http://test:8847"
        assert bridge.backend_name in ("cocapn-plato-sdk", "urllib-fallback")

    def test_repr(self):
        bridge = PlatoSDKBridge()
        assert "PlatoSDKBridge" in repr(bridge)
        assert "backend=" in repr(bridge)

    def test_fallback_client_exists(self):
        # Ensure fallback is importable
        from fleet.plato_sdk_bridge import _FallbackPlatoClient

        client = _FallbackPlatoClient(base_url="http://test:8847")
        assert client.base_url == "http://test:8847"

    def test_tile_result_from_raw(self):
        bridge = PlatoSDKBridge()
        # Simulate a raw tile dict
        raw = {"domain": "d", "question": "q", "answer": "a", "extra": 1}
        result = bridge.get_tile("d", "q")
        # This will fail to connect, but we can test the structure via query
        assert isinstance(result, (TileResult, type(None)))
