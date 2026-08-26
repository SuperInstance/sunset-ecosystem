#!/usr/bin/env python3
"""Fleet Bridge — HTTP API exposing sunset-ecosystem modules to Claw.

Runs as a background daemon (or embedded in claw gateway via extension).
Provides REST endpoints for breeding, consensus, mesh, FLUX, and status.

Usage::
    from claw_fleet_bridge import FleetBridgeServer
    server = FleetBridgeServer(port=8850)
    server.start()
"""

from __future__ import annotations

__all__ = ["FleetBridgeServer"]

import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

logger = logging.getLogger(__name__)

# Ensure sunset-ecosystem is on path
_SUNSET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SUNSET_ROOT not in sys.path:
    sys.path.insert(0, _SUNSET_ROOT)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class FleetBridgeServer:
    """HTTP bridge exposing fleet science as a REST API."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8850) -> None:
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        # Lazy-import fleet modules to avoid heavy startup if not used
        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._json_response({"status": "ok", "fleet": "cocapn"})
                elif self.path == "/status":
                    self._status()
                elif self.path == "/flux/presets":
                    self._flux_presets()
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                if self.path == "/breed":
                    self._breed()
                elif self.path == "/flux/check":
                    self._flux_check()
                elif self.path == "/mesh/insert":
                    self._mesh_insert()
                elif self.path == "/mesh/query":
                    self._mesh_query()
                else:
                    self.send_error(404)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                return json.loads(body) if body else {}

            def _json_response(self, data: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _status(self) -> None:
                try:
                    from nexus.fleet_conductor_v2 import FleetConductorV2

                    conductor = FleetConductorV2()
                    data = conductor.get_status()
                except Exception as e:
                    data = {"error": str(e), "note": "FleetConductorV2 not initialized"}
                self._json_response(data)

            def _flux_presets(self) -> None:
                try:
                    from swarm.flux_preset_library import FluxPresetLibrary

                    lib = FluxPresetLibrary()
                    presets = {
                        name: lib.get_preset(name).description
                        for name in lib.list_presets()
                    }
                except Exception as e:
                    presets = {"error": str(e)}
                self._json_response({"presets": presets})

            def _breed(self) -> None:
                payload = self._read_json()
                n_winners = payload.get("n_winners", 3)
                preset_name = payload.get("preset", "diversity")
                try:
                    from swarm.breeder_daemon_v2 import BreederDaemonV2
                    from swarm.flux_preset_library import FluxPresetLibrary

                    breeder = BreederDaemonV2()
                    preset = FluxPresetLibrary().get_preset(preset_name)
                    breeder.flux_preset = preset
                    results = breeder.cycle(n_winners)
                    self._json_response(
                        {
                            "winners": len(results),
                            "preset": preset_name,
                            "agents": [str(r) for r in results],
                        }
                    )
                except Exception as e:
                    self._json_response({"error": str(e)}, status=500)

            def _flux_check(self) -> None:
                payload = self._read_json()
                candidate = payload.get("candidate", {})
                try:
                    from swarm.flux_vm_gating import FluxVMGater

                    gater = FluxVMGater()
                    passed, reason = gater.check(candidate)
                    self._json_response({"passed": passed, "reason": reason})
                except Exception as e:
                    self._json_response({"error": str(e)}, status=500)

            def _mesh_insert(self) -> None:
                payload = self._read_json()
                try:
                    from swarm.mesh_vector_tables import MeshVectorTable

                    table = MeshVectorTable(table_id=payload.get("table_id", "default"))
                    table.insert_signed(
                        vector=payload["vector"],
                        fitness=payload.get("fitness", 0.0),
                        extra=payload.get("extra", {}),
                    )
                    self._json_response({"inserted": True})
                except Exception as e:
                    self._json_response({"error": str(e)}, status=500)

            def _mesh_query(self) -> None:
                payload = self._read_json()
                try:
                    from swarm.mesh_vector_tables import FleetVectorIndex

                    index = FleetVectorIndex()
                    results = index.query_by_fitness(
                        min_fitness=payload.get("min_fitness", 0.0)
                    )
                    self._json_response(
                        {
                            "count": len(results),
                            "results": [
                                {
                                    "vector": r.vector,
                                    "fitness": r.fitness,
                                    "extra": r.extra,
                                }
                                for r in results
                            ],
                        }
                    )
                except Exception as e:
                    self._json_response({"error": str(e)}, status=500)

        return _Handler

    def start(self) -> None:
        handler = self._make_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("Fleet bridge started on http://%s:%s", self.host, self.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def url(self) -> str:
        if self._server is not None:
            host, port = self._server.server_address
            return f"http://{host}:{port}"
        return f"http://{self.host}:{self.port}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fleet Bridge HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8850)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    server = FleetBridgeServer(host=args.host, port=args.port)
    server.start()
    print(f"Fleet bridge running at {server.url}")
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        server.stop()
        print("\nFleet bridge stopped.")
