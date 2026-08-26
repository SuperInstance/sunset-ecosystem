"""FleetWebServer — HTTP server for fleet dashboard and API.

Serves the fleet dashboard, reports, and health metrics via a web interface.
Uses Python's built-in http.server with custom routing for fleet endpoints.

Endpoints
---------
    GET /            → Fleet dashboard HTML
    GET /api/health  → Health JSON
    GET /api/status  → Status JSON
    GET /api/modules → Modules JSON
    GET /api/metrics → Metrics JSON
    GET /reports/    → Report files
    GET /static/     → Static assets

Usage
-----
    python -m fleet.fleet_web_server --port 8080
"""

from __future__ import annotations

__all__ = [
    "FleetWebServer",
    "FleetRequestHandler",
]

import json
import mimetypes
import os
import socketserver
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from fleet.fleet_cli import FleetCLI


@dataclass
class ServerConfig:
    """Web server configuration."""

    port: int = 8080
    host: str = "0.0.0.0"
    enable_cors: bool = True
    static_dir: str | None = None
    report_dir: str | None = None
    auto_refresh: int = 30  # seconds


class FleetRequestHandler(BaseHTTPRequestHandler):
    """Custom HTTP request handler for fleet endpoints."""

    _cli: FleetCLI | None = None
    _config: ServerConfig | None = None

    @classmethod
    def set_cli(cls, cli: FleetCLI) -> None:
        cls._cli = cli

    @classmethod
    def set_config(cls, config: ServerConfig) -> None:
        cls._config = config

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if self._config and self._config.enable_cors:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def _send_html(self, html: str, status: int = 200) -> None:
        """Send HTML response."""
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        if self._config and self._config.enable_cors:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _send_file(self, path: str, content_type: str | None = None) -> None:
        """Send file response."""
        file_path = Path(path)
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        content = file_path.read_bytes()
        ct = (
            content_type
            or mimetypes.guess_type(str(file_path))[0]
            or "application/octet-stream"
        )

        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging."""
        pass

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/" or path == "/dashboard":
            self._serve_dashboard()
        elif path == "/api/health":
            self._serve_api_health()
        elif path == "/api/status":
            self._serve_api_status()
        elif path == "/api/modules":
            self._serve_api_modules()
        elif path == "/api/metrics":
            self._serve_api_metrics()
        elif path == "/api/benchmark":
            self._serve_api_benchmark()
        elif path.startswith("/reports/"):
            self._serve_report(path)
        elif path.startswith("/static/"):
            self._serve_static(path)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/beat":
            self._serve_api_beat()
        elif path == "/api/report":
            self._serve_api_generate_report()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    # ── Dashboard ───────────────────────────────────────────────

    def _serve_dashboard(self) -> None:
        """Serve the main dashboard HTML."""
        if not self._cli:
            self._send_html("<h1>FleetWebServer: CLI not initialized</h1>")
            return

        refresh = self._config.auto_refresh if self._config else 30

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Sunset Ecosystem Fleet Dashboard</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="{refresh}">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #0d1117; color: #c9d1d9; }}
        h1 {{ color: #58a6ff; margin-bottom: 10px; }}
        h2 {{ color: #79c0ff; font-size: 1.2em; margin-top: 30px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }}
        .badge-healthy {{ background: #238636; color: white; }}
        .badge-degraded {{ background: #9e6a03; color: white; }}
        .badge-critical {{ background: #da3633; color: white; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 20px; }}
        .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
        .card h3 {{ margin: 0 0 10px 0; color: #e6edf3; font-size: 1em; }}
        .metric {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #21262d; }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-label {{ color: #8b949e; font-size: 0.9em; }}
        .metric-value {{ font-weight: 600; color: #e6edf3; }}
        .api-links {{ margin-top: 20px; }}
        .api-links a {{ display: inline-block; margin: 4px 8px 4px 0; padding: 6px 12px; background: #21262d; border: 1px solid #30363d; border-radius: 6px; color: #58a6ff; text-decoration: none; font-size: 0.85em; }}
        .api-links a:hover {{ background: #30363d; }}
        .timestamp {{ color: #6e7681; font-size: 0.8em; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌅 Sunset Ecosystem Fleet Dashboard</h1>
        <span id="status-badge" class="badge badge-healthy">Loading...</span>
    </div>
    <div class="timestamp">Auto-refresh every {refresh}s · Last updated: <span id="timestamp">--</span></div>
    <div class="grid">
        <div class="card">
            <h3>📊 Fleet Health</h3>
            <div class="metric"><span class="metric-label">Modules</span><span class="metric-value" id="modules-count">--</span></div>
            <div class="metric"><span class="metric-label">Tests</span><span class="metric-value" id="tests-count">--</span></div>
            <div class="metric"><span class="metric-label">Healthy</span><span class="metric-value" id="healthy-count">--</span></div>
            <div class="metric"><span class="metric-label">Warning</span><span class="metric-value" id="warning-count">--</span></div>
            <div class="metric"><span class="metric-label">Critical</span><span class="metric-value" id="critical-count">--</span></div>
        </div>
        <div class="card">
            <h3>🔧 API Endpoints</h3>
            <div class="api-links">
                <a href="/api/health">Health</a>
                <a href="/api/status">Status</a>
                <a href="/api/modules">Modules</a>
                <a href="/api/metrics">Metrics</a>
                <a href="/api/benchmark">Benchmark</a>
            </div>
        </div>
    </div>
    <script>
        async function fetchData() {{
            try {{
                const [status, health] = await Promise.all([
                    fetch('/api/status').then(r => r.json()),
                    fetch('/api/health').then(r => r.json())
                ]);
                document.getElementById('status-badge').textContent = status.status || 'UNKNOWN';
                document.getElementById('status-badge').className = 'badge badge-' + (status.status || 'healthy').toLowerCase();
                document.getElementById('modules-count').textContent = status.modules || 0;
                document.getElementById('tests-count').textContent = status.tests || 0;
                document.getElementById('healthy-count').textContent = health.healthy || 0;
                document.getElementById('warning-count').textContent = health.warning || 0;
                document.getElementById('critical-count').textContent = health.critical || 0;
                document.getElementById('timestamp').textContent = new Date().toLocaleString();
            }} catch (e) {{
                console.error('Fetch failed:', e);
            }}
        }}
        fetchData();
    </script>
</body>
</html>"""
        self._send_html(html)

    # ── API Endpoints ─────────────────────────────────────────

    def _serve_api_health(self) -> None:
        """Serve health API."""
        if not self._cli:
            self._send_json({"error": "CLI not initialized"}, 500)
            return
        result = self._cli.health()
        self._send_json(result.data or {"error": "no data"})

    def _serve_api_status(self) -> None:
        """Serve status API."""
        if not self._cli:
            self._send_json({"error": "CLI not initialized"}, 500)
            return
        result = self._cli.status()
        self._send_json(result.data or {"error": "no data"})

    def _serve_api_modules(self) -> None:
        """Serve modules API."""
        if not self._cli:
            self._send_json({"error": "CLI not initialized"}, 500)
            return
        result = self._cli.modules()
        self._send_json(result.data or {"error": "no data"})

    def _serve_api_metrics(self) -> None:
        """Serve metrics API."""
        if not self._cli:
            self._send_json({"error": "CLI not initialized"}, 500)
            return
        result = self._cli.metrics(collect=True)
        self._send_json(result.data or {"error": "no data"})

    def _serve_api_beat(self) -> None:
        """Trigger a fleet beat."""
        if not self._cli:
            self._send_json({"error": "CLI not initialized"}, 500)
            return
        result = self._cli.beat()
        self._send_json(result.data or {"error": "no data"})

    def _serve_api_generate_report(self) -> None:
        """Generate a report."""
        if not self._cli:
            self._send_json({"error": "CLI not initialized"}, 500)
            return
        result = self._cli.report()
        self._send_json({"success": result.success, "message": result.message})

    def _serve_api_benchmark(self) -> None:
        """Serve benchmark summary."""
        from fleet.fleet_benchmark import FleetBenchmark

        bm = FleetBenchmark()
        suite = bm.run_full_suite()
        self._send_json(suite.summary())

    # ── Static Files ──────────────────────────────────────────

    def _serve_report(self, path: str) -> None:
        """Serve report files."""
        report_dir = self._config.report_dir if self._config else None
        if not report_dir:
            report_dir = str(Path(".").resolve() / "docs" / "reports")

        file_path = Path(report_dir) / path.replace("/reports/", "")
        if file_path.exists() and file_path.is_file():
            self._send_file(str(file_path))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Report not found")

    def _serve_static(self, path: str) -> None:
        """Serve static files."""
        static_dir = self._config.static_dir if self._config else None
        if not static_dir:
            self.send_error(HTTPStatus.NOT_FOUND, "Static dir not configured")
            return

        file_path = Path(static_dir) / path.replace("/static/", "")
        if file_path.exists() and file_path.is_file():
            self._send_file(str(file_path))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")


class FleetWebServer:
    """Fleet web server.

    Parameters
    ----------
    workspace : str
        Path to sunset-ecosystem workspace.
    config : ServerConfig | None
        Server configuration.
    """

    def __init__(
        self, workspace: str = ".", config: ServerConfig | None = None
    ) -> None:
        self.workspace = Path(workspace)
        self.config = config or ServerConfig()
        self._cli = FleetCLI(workspace=str(self.workspace))
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self, blocking: bool = False) -> None:
        """Start the web server.

        Parameters
        ----------
        blocking : bool
            If True, block until server is stopped.
        """
        FleetRequestHandler.set_cli(self._cli)
        FleetRequestHandler.set_config(self.config)

        self._server = HTTPServer(
            (self.config.host, self.config.port),
            FleetRequestHandler,
        )

        if blocking:
            print(
                f"FleetWebServer running at http://{self.config.host}:{self.config.port}/"
            )
            self._server.serve_forever()
        else:
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._thread.start()
            print(
                f"FleetWebServer started at http://{self.config.host}:{self.config.port}/"
            )

    def stop(self) -> None:
        """Stop the web server."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def url(self) -> str:
        """Server URL."""
        return f"http://{self.config.host}:{self.config.port}/"


def main() -> None:
    """CLI entry point for web server."""
    import argparse

    parser = argparse.ArgumentParser(description="Fleet Web Server")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--workspace", "-w", default=".", help="Workspace path")
    parser.add_argument("--no-cors", action="store_true", help="Disable CORS")
    parser.add_argument(
        "--refresh",
        "-r",
        type=int,
        default=30,
        help="Dashboard refresh interval (seconds)",
    )
    args = parser.parse_args()

    config = ServerConfig(
        port=args.port,
        host=args.host,
        enable_cors=not args.no_cors,
        auto_refresh=args.refresh,
    )
    server = FleetWebServer(workspace=args.workspace, config=config)
    server.start(blocking=True)


if __name__ == "__main__":
    main()
