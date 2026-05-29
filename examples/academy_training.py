"""examples/academy_training.py — Plato Academy training pipeline demo.

Demonstrates enrolling agents, running training modules, and promoting
to fleet service.

Usage:
    python examples/academy_training.py
"""

from fleet.plato_academy_bridge import PlatoAcademyBridge

def main():
    print("🎓 Plato Academy Training Pipeline Demo")
    print("=" * 40)

    bridge = PlatoAcademyBridge(node_id="alpha")

    # Show cohort findings
    print("\n📋 Known friction points from cohort testing:")
    friction_points = bridge.get_friction_points()
    for point in friction_points:
        print(f"  [{point['severity']:8s}] {point['agent']:12s}: {point['finding']}")

    # Apply critical fixes
    print("\n🔧 Applying critical fixes...")
    bridge.fix_friction_point("zero_authentication", "add_auth")
    bridge.fix_friction_point("no_web_ui", "add_web_ui")
    bridge.fix_friction_point("no_broadcast_endpoints", "add_broadcast")
    bridge.fix_friction_point("no_global_fleet_map", "add_fleet_map")
    print(f"  Applied {len(bridge.get_fixes())} fixes")

    # Enroll a new agent
    print("\n📝 Enrolling new agent...")
    agent = bridge.enroll_agent("greenhorn_001", level="greenhorn")
    print(f"  Agent ID: {agent.agent_id}")
    print(f"  Starting level: {agent.level}")

    # Run training modules
    print("\n📚 Running training modules:")
    modules = [
        ("boot_camp", "Basic orientation"),
        ("room_exploration", "Room navigation"),
        ("tile_creation", "Creating tiles"),
        ("spell_casting", "Using spells"),
        ("api_integration", "API development"),
        ("orchestration", "Fleet orchestration"),
        ("captain_chair", "Captain certification"),
    ]

    for module, description in modules:
        result = bridge.run_module("greenhorn_001", module)
        print(f"  ✅ {module:20s} ({description}) — Score: {result['score']:.0f}, Level: {result['level']}")

    # Check final progression
    print("\n🎖️ Final progression:")
    progress = bridge.get_progression("greenhorn_001")
    print(f"  Agent: {progress['agent_id']}")
    print(f"  Level: {progress['level']}")
    print(f"  Score: {progress['score']:.0f}")
    print(f"  Modules completed: {len(progress['modules_completed'])}")

    # Promote to fleet
    print("\n🚀 Promoting to fleet service...")
    if bridge.promote_to_fleet("greenhorn_001"):
        print("  ✅ Agent promoted to active fleet service!")
    else:
        print("  ❌ Agent not ready for fleet promotion")

    # Academy stats
    print("\n📊 Academy statistics:")
    stats = bridge.get_stats()
    print(f"  Total agents: {stats['total_agents']}")
    print(f"  Level distribution: {stats['level_distribution']}")
    print(f"  Cohort findings: {stats['cohort_findings']}")
    print(f"  Fixes applied: {stats['fixes_applied']}")

    print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()
