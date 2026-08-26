"""Tests for Spread Integration bridge.

Mocks HTTP calls to spread viewer. Tests local functionality and
connection logic.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from fleet.spread_integration import SpreadBridge
from fleet.deckboss import DeckbossGrid, FleetFormulaEnv


class TestSpreadBridge:
    def test_init(self):
        bridge = SpreadBridge(fleet_node_id="alpha")
        assert bridge.fleet_node_id == "alpha"
        assert bridge.is_connected is False

    def test_connect(self):
        bridge = SpreadBridge(fleet_node_id="alpha")
        bridge.connect_to_spread(host="127.0.0.1", port=8080)
        assert bridge.is_connected is True
        assert bridge.flight_location is not None
        bridge.disconnect()
        assert bridge.is_connected is False

    def test_disconnect_without_connect(self):
        bridge = SpreadBridge(fleet_node_id="alpha")
        bridge.disconnect()  # Should not raise
        assert bridge.is_connected is False

    def test_push_sheet_not_connected(self):
        bridge = SpreadBridge(fleet_node_id="alpha")
        result = bridge.push_sheet("test", [["a", "b"], ["1", "2"]])
        assert result is False

    @patch("urllib.request.urlopen")
    def test_push_sheet_connected(self, mock_urlopen):
        bridge = SpreadBridge(fleet_node_id="alpha")
        bridge.connect_to_spread()
        mock_urlopen.return_value.__enter__.return_value.read.return_value = (
            b'{"ok": true}'
        )
        result = bridge.push_sheet("test", [["a", "b"], ["1", "2"]])
        assert result is True
        bridge.disconnect()

    @patch("urllib.request.urlopen")
    def test_push_grid(self, mock_urlopen):
        bridge = SpreadBridge(fleet_node_id="alpha")
        bridge.connect_to_spread()
        grid = DeckbossGrid(FleetFormulaEnv())
        grid.set_cell("A1", "hello")
        grid.set_cell("B1", "42")
        mock_urlopen.return_value.__enter__.return_value.read.return_value = (
            b'{"ok": true}'
        )
        result = bridge.push_grid("sheet1", grid)
        assert result is True
        bridge.disconnect()

    @patch("urllib.request.urlopen")
    def test_push_formula(self, mock_urlopen):
        bridge = SpreadBridge(fleet_node_id="alpha")
        bridge.connect_to_spread()
        mock_urlopen.return_value.__enter__.return_value.read.return_value = (
            b'{"ok": true}'
        )
        result = bridge.push_formula("A1", "=FLEET_HEALTH()")
        assert result is True
        bridge.disconnect()

    @patch("urllib.request.urlopen")
    def test_push_fleet_snapshot(self, mock_urlopen):
        bridge = SpreadBridge(fleet_node_id="alpha")
        bridge.connect_to_spread()
        mock_urlopen.return_value.__enter__.return_value.read.return_value = (
            b'{"ok": true}'
        )
        snapshot = {"agent_count": 50, "thermal_avg": 0.75}
        result = bridge.push_fleet_snapshot(snapshot)
        assert result is True
        bridge.disconnect()

    @patch("urllib.request.urlopen")
    def test_get_spread_status(self, mock_urlopen):
        bridge = SpreadBridge(fleet_node_id="alpha")
        bridge.connect_to_spread()
        mock_urlopen.return_value.__enter__.return_value.read.return_value = (
            b'{"sheets": 3, "rows": 1000}'
        )
        status = bridge.get_spread_status()
        assert status == {"sheets": 3, "rows": 1000}
        bridge.disconnect()

    @patch("urllib.request.urlopen")
    def test_get_spread_status_failure(self, mock_urlopen):
        bridge = SpreadBridge(fleet_node_id="alpha")
        bridge.connect_to_spread()
        mock_urlopen.side_effect = Exception("connection refused")
        status = bridge.get_spread_status()
        assert status is None
        bridge.disconnect()

    @patch("urllib.request.urlopen")
    def test_notification_failure(self, mock_urlopen):
        bridge = SpreadBridge(fleet_node_id="alpha")
        bridge.connect_to_spread()
        mock_urlopen.side_effect = Exception("connection refused")
        result = bridge.push_formula("A1", "=1+1")
        assert result is False
        bridge.disconnect()
