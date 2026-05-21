"""Local Grammar Engine Service — standalone test server.

Spins up a minimal HTTP grammar engine on localhost:4045
(to match Oracle1's port) with the security fix active.

Purpose: Test the fix against known chaos vectors without
touching Oracle1's live service.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import sys
from pathlib import Path

# Add project root to path for 'grammar' import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grammar.core import (
    Rule,
    ValidationError,
    create_rule,
    create_rule_from_dict,
)

logger = logging.getLogger(__name__)


class GrammarHandler(BaseHTTPRequestHandler):
    """HTTP handler for rule creation and listing."""

    _rules: list[Rule] = []

    def log_message(self, format: str, *args: Any) -> None:
        logger.info(format, *args)

    def do_GET(self) -> None:
        if self.path == "/rules":
            self._send_json(200, {"rules": [r.name for r in self._rules]})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path == "/rules":
            self._handle_create_rule()
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_create_rule(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"Invalid JSON: {exc}"})
            return

        try:
            rule = create_rule_from_dict(data)
            self._rules.append(rule)
            self._send_json(
                201,
                {
                    "status": "created",
                    "name": rule.name,
                    "production": {
                        "tagline": rule.production.tagline,
                        "condition": rule.production.condition,
                        "exec_field": rule.production.exec_field,
                    },
                },
            )
            logger.info("Created rule: %s", rule.name)
        except ValidationError as exc:
            self._send_json(400, {"error": "Validation failed", "detail": str(exc)})
            logger.warning("Blocked rule: %s", exc)

    def _send_json(self, status: int, data: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


def run_server(host: str = "0.0.0.0", port: int = 4045) -> None:
    """Start the grammar engine service."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    server = HTTPServer((host, port), GrammarHandler)
    logger.info("Grammar Engine running on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    run_server()
