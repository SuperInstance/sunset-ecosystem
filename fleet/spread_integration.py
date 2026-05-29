"""fleet/spread_integration.py — Python bridge to SuperInstance/spread Rust viewer.

Provides integration between our fleet systems and the spread Rust
spreadsheet viewer.  This module runs in Python; the Rust side is in
the separate `cocapn-spread` crate (see rust/ directory).

Usage
-----
    from fleet.spread_integration import SpreadBridge

    bridge = SpreadBridge(fleet_node_id="alpha")
    bridge.connect_to_spread(host="127.0.0.1", port=8080)
    bridge.push_sheet("fleet_status", deckboss_grid.to_rows())
    bridge.push_formula("A1", "=FLEET_HEALTH()")

    # On the Rust side:
    # spread loads the sheet via Arrow Flight and renders it

Architecture
------------
- Python side: this module — serializes fleet data to Arrow/CSV/JSON
- Rust side: `cocapn-spread` crate — receives data, renders in GPUI
- Transport: Arrow Flight (gRPC) or HTTP JSON fallback
- Data format: Arrow IPC streams for live updates, CSV for snapshots
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fleet.deckboss import DeckbossGrid
from fleet.parquet_bridge import ParquetBridge
from swarm.arrow_flight_mesh import ArrowFlightMeshNode, MeshPeer

logger = logging.getLogger(__name__)


@dataclass
class SpreadBridge:
    """Bridge between fleet systems and the spread Rust viewer."""

    fleet_node_id: str
    _flight_node: Optional[ArrowFlightMeshNode] = field(default=None, repr=False)
    _spread_host: str = "127.0.0.1"
    _spread_port: int = 8080
    _connected: bool = False

    def connect_to_spread(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Establish connection to the spread viewer."""
        self._spread_host = host
        self._spread_port = port
        self._flight_node = ArrowFlightMeshNode(
            node_id=self.fleet_node_id,
            host=host,
            listen_port=0,  # auto-assign
        )
        self._flight_node.start()
        self._connected = True
        logger.info("SpreadBridge connected to %s:%d", host, port)

    def disconnect(self) -> None:
        """Close connection."""
        if self._flight_node:
            self._flight_node.stop()
        self._connected = False

    def push_sheet(self, sheet_name: str, rows: List[List[str]]) -> bool:
        """Push a sheet (list of rows) to spread."""
        if not self._connected:
            logger.warning("Not connected to spread")
            return False
        # Store as local table for Arrow Flight retrieval
        self._flight_node.store_table(sheet_name, rows)
        # Also notify spread via HTTP
        return self._notify_spread("sheet", {"name": sheet_name, "rows": len(rows)})

    def push_grid(self, sheet_name: str, grid: DeckbossGrid) -> bool:
        """Push a DeckbossGrid to spread."""
        bridge = ParquetBridge(grid=grid)
        rows = bridge._to_rows()
        return self.push_sheet(sheet_name, rows)

    def push_formula(self, cell_ref: str, formula: str) -> bool:
        """Push a formula definition to spread."""
        return self._notify_spread("formula", {"cell": cell_ref, "formula": formula})

    def push_fleet_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """Push a fleet status snapshot as a sheet."""
        bridge = ParquetBridge()
        bridge.load_fleet_snapshot(snapshot)
        rows = bridge._to_rows()
        return self.push_sheet("fleet_snapshot", rows)

    def _notify_spread(self, endpoint: str, data: Dict[str, Any]) -> bool:
        """Send HTTP notification to spread."""
        try:
            url = f"http://{self._spread_host}:{self._spread_port}/{endpoint}"
            payload = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5.0)
            return True
        except Exception as exc:
            logger.warning("Spread notification failed: %s", exc)
            return False

    def get_spread_status(self) -> Optional[Dict[str, Any]]:
        """Query spread viewer status."""
        try:
            url = f"http://{self._spread_host}:{self._spread_port}/status"
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Spread status query failed: %s", exc)
            return None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def flight_location(self) -> Optional[str]:
        if self._flight_node:
            return self._flight_node.location
        return None
