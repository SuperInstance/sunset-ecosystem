"""examples/flux_os_deploy.py — FLUX OS deployment demo.

Demonstrates compiling and deploying a breeding agent to FLUX OS.

Usage:
    python examples/flux_os_deploy.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fleet.flux_os_bridge import FluxOSBridge

def main():
    print("🐧 FLUX OS Deployment Demo")
    print("=" * 40)

    bridge = FluxOSBridge(node_id="alpha", fleet_id="cocapn")

    # Generate FLUX.MD for a breeding agent
    print("\n📝 Generating FLUX.MD...")
    config = {
        "population_size": 100,
        "mutation_rate": 0.05,
        "selection": "tournament",
        "crossover": "uniform",
    }
    md = bridge.generate_flux_md("breeder_alpha", config)
    print(f"  Generated {len(md)} bytes of FLUX.MD")
    print(f"  First line: {md.split(chr(10))[0]}")

    # Compile the agent
    print("\n⚙️  Compiling breeding agent...")
    result = bridge.compile_breeding_agent("breeder_alpha", md)
    print(f"  Compilation: {'✅ success' if result else '❌ failed'}")

    # Deploy to ARM64 (e.g., Raspberry Pi 4)
    print("\n🚀 Deploying to ARM64...")
    deploy_result = bridge.deploy(
        "breeder_alpha",
        target="arm64",
        board="rpi4",
        strategy="canary"
    )
    print(f"  Deployment: {'✅ success' if deploy_result else '❌ failed'}")

    # Start breeding loop
    print("\n▶️  Starting breeding loop...")
    start_result = bridge.start_breeding_loop("breeder_alpha")
    print(f"  Start: {'✅ success' if start_result else '❌ failed'}")

    # Check status
    print("\n📊 Deployment status:")
    status = bridge.get_deployment_status()
    print(f"  Node: {status['node_id']}")
    print(f"  Fleet: {status['fleet_id']}")
    print(f"  FLUX OS available: {status['flux_os_available']}")
    print(f"  Compiled agents: {status['compiled_agents']}")
    print(f"  Active deployments: {status['active_deployments']}")

    # Simulate logs
    print("\n📜 Agent logs:")
    logs = bridge.get_agent_logs("breeder_alpha")
    for log in logs:
        print(f"  {log}")

    # Stop breeding loop
    print("\n⏹️  Stopping breeding loop...")
    stop_result = bridge.stop_breeding_loop("breeder_alpha")
    print(f"  Stop: {'✅ success' if stop_result else '❌ failed'}")

    print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()
