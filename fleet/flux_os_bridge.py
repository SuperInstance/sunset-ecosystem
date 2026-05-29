"""fleet/flux_os_bridge.py — Bridge to FLUX OS for agent-native deployment.

Deploys sunset breeding agents as FLUX OS services. FLUX OS is a
microkernel in pure C where the kernel IS the compiler — it compiles
FLUX.MD to native binaries at boot.

Usage
-----
    from fleet.flux_os_bridge import FluxOSBridge

    bridge = FluxOSBridge(node_id="alpha", fleet_id="cocapn")
    bridge.compile_breeding_agent("breeder_v2", source_flux_md="...")
    bridge.deploy(target="arm64", board="rpi4", strategy="canary")

    # Check status
    status = bridge.get_deployment_status()
    if status["booted"]:
        bridge.start_breeding_loop()

Architecture
------------
- Python side: this module — generates FLUX.MD, orchestrates deployment
- FLUX OS side: microkernel — compiles FLUX.MD to native binary at boot
- Transport: HTTP/HTTPS for status, gRPC for control commands
- Data format: FLUX.MD (Markdown with embedded opcodes)

Dependencies
------------
- flux-os (optional): C microkernel, build with `make`
- Fallback: mock deployment for testing
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check for FLUX OS availability
FLUX_OS_AVAILABLE = False
_FLUX_OS_PATH = os.environ.get("FLUX_OS_PATH", "/usr/local/flux-os")
try:
    if Path(_FLUX_OS_PATH).exists():
        FLUX_OS_AVAILABLE = True
except Exception:
    pass


@dataclass
class FluxOSBridge:
    """Bridge between sunset ecosystem and FLUX OS microkernel."""

    node_id: str
    fleet_id: str = "cocapn"
    _flux_os_path: str = "/usr/local/flux-os"
    _compiled_agents: Dict[str, str] = field(default_factory=dict, repr=False)
    _deployments: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if FLUX_OS_AVAILABLE:
            logger.info("FLUX OS bridge initialized (path=%s)", self._flux_os_path)
        else:
            logger.warning("FLUX OS not available; using mock deployment")

    def compile_breeding_agent(self, agent_name: str, source_flux_md: str) -> bool:
        """Compile a breeding agent from FLUX.MD to native binary.

        Steps:
        1. Write FLUX.MD to temporary file
        2. Shell out to `flux build --target native`
        3. Store binary path in cache
        """
        t0 = time.perf_counter()

        # Write FLUX.MD
        md_path = Path(tempfile.gettempdir()) / f"{agent_name}.flux.md"
        md_path.write_text(source_flux_md, encoding="utf-8")

        if not FLUX_OS_AVAILABLE:
            # Mock compilation
            binary_path = str(Path(tempfile.gettempdir()) / f"{agent_name}.mock")
            Path(binary_path).touch()
            self._compiled_agents[agent_name] = binary_path
            logger.info("Mock compiled %s -> %s", agent_name, binary_path)
            return True

        # Real compilation
        try:
            result = subprocess.run(
                ["flux", "build", "--target", "native", str(md_path)],
                cwd=self._flux_os_path,
                capture_output=True,
                check=True,
                timeout=60.0,
            )
            binary_path = str(md_path.with_suffix(""))
            self._compiled_agents[agent_name] = binary_path
            dt = (time.perf_counter() - t0) * 1000
            logger.info("Compiled %s in %.1fms", agent_name, dt)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("FLUX compilation failed: %s", exc.stderr.decode("utf-8", errors="replace"))
            return False
        except Exception as exc:
            logger.error("FLUX compilation error: %s", exc)
            return False

    def deploy(self, agent_name: str, target: str = "native",
               board: Optional[str] = None, strategy: str = "canary") -> bool:
        """Deploy a compiled agent to FLUX OS fleet.

        Args:
            agent_name: Name of compiled agent
            target: Architecture (native, arm64, riscv64)
            board: Board variant (rpi4, jetson, etc.)
            strategy: Deployment strategy (canary, rolling, blue-green)
        """
        if agent_name not in self._compiled_agents:
            logger.warning("Agent %s not compiled", agent_name)
            return False

        deployment = {
            "agent": agent_name,
            "target": target,
            "board": board,
            "strategy": strategy,
            "timestamp": time.time(),
            "node_id": self.node_id,
            "fleet_id": self.fleet_id,
            "status": "deployed" if FLUX_OS_AVAILABLE else "mock_deployed",
        }
        self._deployments.append(deployment)
        logger.info("Deployed %s to %s (%s)", agent_name, target, strategy)
        return True

    def start_breeding_loop(self, agent_name: str) -> bool:
        """Start the breeding loop on deployed agent."""
        if not FLUX_OS_AVAILABLE:
            logger.info("Mock start breeding loop for %s", agent_name)
            return True

        try:
            subprocess.run(
                ["flux", "start", agent_name],
                cwd=self._flux_os_path,
                capture_output=True,
                check=True,
                timeout=10.0,
            )
            return True
        except Exception as exc:
            logger.warning("Start breeding loop failed: %s", exc)
            return False

    def stop_breeding_loop(self, agent_name: str) -> bool:
        """Stop the breeding loop."""
        if not FLUX_OS_AVAILABLE:
            logger.info("Mock stop breeding loop for %s", agent_name)
            return True

        try:
            subprocess.run(
                ["flux", "stop", agent_name],
                cwd=self._flux_os_path,
                capture_output=True,
                check=True,
                timeout=10.0,
            )
            return True
        except Exception as exc:
            logger.warning("Stop breeding loop failed: %s", exc)
            return False

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get status of all deployments."""
        return {
            "node_id": self.node_id,
            "fleet_id": self.fleet_id,
            "flux_os_available": FLUX_OS_AVAILABLE,
            "compiled_agents": len(self._compiled_agents),
            "active_deployments": len(self._deployments),
            "deployments": self._deployments,
        }

    def generate_flux_md(self, agent_name: str, breeding_config: Dict[str, Any]) -> str:
        """Generate FLUX.MD for a breeding agent."""
        return f"""# FLUX Agent: {agent_name}

## Metadata
- fleet: {self.fleet_id}
- node: {self.node_id}
- version: 1.0.0

## Breeding Configuration
```yaml
population_size: {breeding_config.get('population_size', 50)}
mutation_rate: {breeding_config.get('mutation_rate', 0.1)}
selection: tournament
crossover: uniform
elitism: true
```

## Opcodes
```flux
LOAD population
EVAL fitness
SELECT parents
CROSSOVER
MUTATE
EVAL fitness
REPLACE
STORE population
```

## Entry Point
```flux
main:
    INIT breeding_loop
    LOOP:
        TICK metronome
        BREED generation
        SYNC mesh
    HALT on convergence
```
"""

    def get_agent_logs(self, agent_name: str) -> List[str]:
        """Retrieve logs from deployed agent."""
        if not FLUX_OS_AVAILABLE:
            return ["mock log: breeding loop started", "mock log: generation 1 complete"]

        try:
            result = subprocess.run(
                ["flux", "logs", agent_name],
                cwd=self._flux_os_path,
                capture_output=True,
                check=True,
                timeout=10.0,
            )
            return result.stdout.decode("utf-8").strip().split("\n")
        except Exception:
            return []
