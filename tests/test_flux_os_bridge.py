"""Tests for FLUX OS Bridge — agent deployment and compilation.

Covers FLUX.MD generation, compilation, deployment, and status queries.
"""

import pytest

from fleet.flux_os_bridge import FluxOSBridge, FLUX_OS_AVAILABLE


class TestFluxOSBridge:
    def test_init(self):
        bridge = FluxOSBridge(node_id="alpha", fleet_id="cocapn")
        assert bridge.node_id == "alpha"
        assert bridge.fleet_id == "cocapn"

    def test_generate_flux_md(self):
        bridge = FluxOSBridge(node_id="alpha", fleet_id="cocapn")
        config = {"population_size": 100, "mutation_rate": 0.05}
        md = bridge.generate_flux_md("breeder_v1", config)
        assert "FLUX Agent: breeder_v1" in md
        assert "population_size: 100" in md
        assert "cocapn" in md
        assert "alpha" in md

    def test_compile_breeding_agent(self):
        bridge = FluxOSBridge(node_id="alpha")
        md = bridge.generate_flux_md("test_agent", {"population_size": 50})
        result = bridge.compile_breeding_agent("test_agent", md)
        assert result is True
        assert "test_agent" in bridge._compiled_agents

    def test_compile_then_deploy(self):
        bridge = FluxOSBridge(node_id="alpha")
        md = bridge.generate_flux_md("deploy_agent", {"population_size": 50})
        bridge.compile_breeding_agent("deploy_agent", md)
        result = bridge.deploy(
            "deploy_agent", target="arm64", board="rpi4", strategy="canary"
        )
        assert result is True
        assert len(bridge._deployments) == 1

    def test_deploy_without_compile(self):
        bridge = FluxOSBridge(node_id="alpha")
        result = bridge.deploy("missing", target="native")
        assert result is False

    def test_start_stop_mock(self):
        bridge = FluxOSBridge(node_id="alpha")
        md = bridge.generate_flux_md("start_agent", {})
        bridge.compile_breeding_agent("start_agent", md)
        assert bridge.start_breeding_loop("start_agent") is True
        assert bridge.stop_breeding_loop("start_agent") is True

    def test_get_deployment_status(self):
        bridge = FluxOSBridge(node_id="alpha")
        md = bridge.generate_flux_md("status_agent", {})
        bridge.compile_breeding_agent("status_agent", md)
        bridge.deploy("status_agent", target="native")
        status = bridge.get_deployment_status()
        assert status["node_id"] == "alpha"
        assert status["compiled_agents"] == 1
        assert status["active_deployments"] == 1
        assert status["flux_os_available"] == FLUX_OS_AVAILABLE

    def test_get_agent_logs(self):
        bridge = FluxOSBridge(node_id="alpha")
        logs = bridge.get_agent_logs("any_agent")
        assert isinstance(logs, list)
        assert len(logs) >= 1

    def test_multiple_deployments(self):
        bridge = FluxOSBridge(node_id="alpha")
        for i in range(3):
            md = bridge.generate_flux_md(f"agent_{i}", {})
            bridge.compile_breeding_agent(f"agent_{i}", md)
            bridge.deploy(f"agent_{i}", target="native")

        status = bridge.get_deployment_status()
        assert status["compiled_agents"] == 3
        assert status["active_deployments"] == 3

    def test_deployment_fields(self):
        bridge = FluxOSBridge(node_id="alpha")
        md = bridge.generate_flux_md("field_agent", {})
        bridge.compile_breeding_agent("field_agent", md)
        bridge.deploy("field_agent", target="arm64", board="rpi4", strategy="rolling")

        dep = bridge._deployments[0]
        assert dep["agent"] == "field_agent"
        assert dep["target"] == "arm64"
        assert dep["board"] == "rpi4"
        assert dep["strategy"] == "rolling"
        assert dep["node_id"] == "alpha"
        assert "timestamp" in dep

    def test_flux_md_has_opcodes(self):
        bridge = FluxOSBridge(node_id="alpha")
        md = bridge.generate_flux_md("opcode_agent", {})
        assert "LOAD population" in md
        assert "EVAL fitness" in md
        assert "SELECT parents" in md
        assert "CROSSOVER" in md
        assert "MUTATE" in md

    def test_flux_md_has_entry_point(self):
        bridge = FluxOSBridge(node_id="alpha")
        md = bridge.generate_flux_md("entry_agent", {})
        assert "main:" in md
        assert "INIT breeding_loop" in md
        assert "TICK metronome" in md
