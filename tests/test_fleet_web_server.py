"""Tests for FleetWebServer — HTTP server for fleet dashboard.

Reference: fleet/fleet_web_server.py
"""

from __future__ import annotations

import json
import threading
import time

import pytest
import requests

from fleet.fleet_web_server import FleetRequestHandler, FleetWebServer, ServerConfig


class TestServerConfig:
    def test_defaults(self) -> None:
        config = ServerConfig()
        assert config.port == 8080
        assert config.host == "0.0.0.0"
        assert config.enable_cors is True
        assert config.auto_refresh == 30

    def test_custom(self) -> None:
        config = ServerConfig(port=9000, host="127.0.0.1", enable_cors=False, auto_refresh=60)
        assert config.port == 9000
        assert config.host == "127.0.0.1"
        assert config.enable_cors is False
        assert config.auto_refresh == 60


class TestFleetWebServer:
    def test_init(self) -> None:
        server = FleetWebServer()
        assert server.workspace.exists()
        assert server.config.port == 8080

    def test_init_custom_config(self) -> None:
        config = ServerConfig(port=9000, host="127.0.0.1")
        server = FleetWebServer(config=config)
        assert server.config.port == 9000
        assert server.config.host == "127.0.0.1"

    def test_url(self) -> None:
        server = FleetWebServer()
        assert server.url == "http://0.0.0.0:8080/"

    def test_url_custom(self) -> None:
        config = ServerConfig(port=9000, host="localhost")
        server = FleetWebServer(config=config)
        assert server.url == "http://localhost:9000/"

    def test_start_stop(self) -> None:
        config = ServerConfig(port=0)  # Let OS assign port
        server = FleetWebServer(config=config)
        server.start(blocking=False)
        time.sleep(0.5)  # Let server start
        assert server.is_running()
        server.stop()
        time.sleep(0.2)
        assert not server.is_running()

    def test_is_running_not_started(self) -> None:
        server = FleetWebServer()
        assert not server.is_running()

    def test_request_handler_set_cli(self) -> None:
        from fleet.fleet_cli import FleetCLI
        cli = FleetCLI()
        FleetRequestHandler.set_cli(cli)
        assert FleetRequestHandler._cli is cli

    def test_request_handler_set_config(self) -> None:
        config = ServerConfig()
        FleetRequestHandler.set_config(config)
        assert FleetRequestHandler._config is config


class TestFleetWebServerEndpoints:
    """Integration tests with actual HTTP requests."""

    @pytest.fixture(scope="class")
    def server(self):
        """Start a test server on a random port."""
        config = ServerConfig(port=0, host="127.0.0.1")
        server = FleetWebServer(config=config)
        server.start(blocking=False)
        time.sleep(0.5)

        # Get actual port
        port = server._server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"

        yield base_url

        server.stop()

    def test_dashboard(self, server) -> None:
        response = requests.get(f"{server}/", timeout=5)
        assert response.status_code == 200
        assert "Fleet Dashboard" in response.text
        assert "text/html" in response.headers.get("Content-Type", "")

    def test_api_health(self, server) -> None:
        response = requests.get(f"{server}/api/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "healthy" in data
        assert "critical" in data

    def test_api_status(self, server) -> None:
        response = requests.get(f"{server}/api/status", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "modules" in data
        assert "tests" in data

    def test_api_modules(self, server) -> None:
        response = requests.get(f"{server}/api/modules", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "modules" in data
        assert len(data["modules"]) == 20

    def test_api_metrics(self, server) -> None:
        response = requests.get(f"{server}/api/metrics", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "cycle" in data

    def test_api_benchmark(self, server) -> None:
        response = requests.get(f"{server}/api/benchmark", timeout=30)
        assert response.status_code == 200
        data = response.json()
        assert "benchmark_count" in data
        assert "fastest" in data
        assert "slowest" in data

    def test_post_beat(self, server) -> None:
        response = requests.post(f"{server}/api/beat", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "cycle" in data

    def test_post_report(self, server) -> None:
        response = requests.post(f"{server}/api/report", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_cors_headers(self, server) -> None:
        response = requests.get(f"{server}/api/health", timeout=5)
        assert "Access-Control-Allow-Origin" in response.headers
        assert response.headers["Access-Control-Allow-Origin"] == "*"

    def test_options_cors(self, server) -> None:
        response = requests.options(f"{server}/api/health", timeout=5)
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

    def test_404(self, server) -> None:
        response = requests.get(f"{server}/nonexistent", timeout=5)
        assert response.status_code == 404

    def test_api_health_json_content_type(self, server) -> None:
        response = requests.get(f"{server}/api/health", timeout=5)
        assert "application/json" in response.headers.get("Content-Type", "")

    def test_dashboard_html_content_type(self, server) -> None:
        response = requests.get(f"{server}/", timeout=5)
        assert "text/html" in response.headers.get("Content-Type", "")

    def test_dashboard_auto_refresh(self, server) -> None:
        response = requests.get(f"{server}/", timeout=5)
        assert "http-equiv=\"refresh\"" in response.text or "refresh" in response.text

    def test_dashboard_has_api_links(self, server) -> None:
        response = requests.get(f"{server}/", timeout=5)
        assert "api/health" in response.text
        assert "api/status" in response.text
        assert "api/modules" in response.text

    def test_status_data_types(self, server) -> None:
        response = requests.get(f"{server}/api/status", timeout=5)
        data = response.json()
        assert isinstance(data["modules"], int)
        assert isinstance(data["tests"], int)
        assert isinstance(data["healthy"], int)
        assert isinstance(data["warning"], int)
        assert isinstance(data["critical"], int)

    def test_modules_list_structure(self, server) -> None:
        response = requests.get(f"{server}/api/modules", timeout=5)
        data = response.json()
        modules = data["modules"]
        assert len(modules) == 20
        for mod in modules:
            assert "name" in mod
            assert "status" in mod

    def test_benchmark_structure(self, server) -> None:
        response = requests.get(f"{server}/api/benchmark", timeout=30)
        data = response.json()
        assert "benchmark_count" in data
        assert isinstance(data["benchmark_count"], int)
        assert data["benchmark_count"] == 5
