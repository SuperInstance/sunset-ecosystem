"""Holodeck Demo — 10 rooms, 50 agents, 20 ticks, interactive HTML export.

Run:
    python sunset-ecosystem/examples/holodeck_demo.py

Open the generated holodeck_demo.html in any browser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root so fleet.holodeck resolves
sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet.holodeck import Holodeck, MockPlatoSource


def run_demo(output_dir: str = ".") -> None:
    """Run the canonical holodeck demo."""
    out = Path(output_dir) / "holodeck_demo.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    print("🦀 Cocapn Fleet Holodeck — initialising …")

    # 10 rooms, 50 agents
    src = MockPlatoSource(room_count=10, agent_count=50, movement_prob=0.25, seed=42)
    hd = Holodeck()

    # bootstrap into holodeck
    hd.ingest_mock_source(src)
    print(f"  Rooms: {hd.room_count()} | Agents: {hd.agent_count()}")

    # 20 ticks
    print("  Running 20 simulation ticks …")
    for i in range(1, 21):
        src.tick()
        hd.ingest_mock_source(src)
        if i % 5 == 0:
            scene = hd.get_scene()
            conn_count = len(scene["connections"])
            print(f"    Tick {i:02d}: {conn_count} room-to-room connections formed")

    # export
    hd.export_html(str(out))
    print(f"\n✅ Exported to {out.resolve()}")

    # stats
    scene = hd.snapshot()
    print("\n── Scene Stats ──")
    print(f"  Rooms      : {len(scene['rooms'])}")
    print(f"  Agents     : {len(scene['agents'])}")
    print(f"  Connections: {len(scene['connections'])}")

    overcap = [
        rid for rid, r in scene["rooms"].items()
        if r["occupancy"] > r["capacity"]
    ]
    if overcap:
        print(f"  ⚠️  Overcapacity rooms: {', '.join(overcap)}")
    else:
        print("  No rooms over capacity ✓")

    phase_counts: dict[str, int] = {}
    for a in scene["agents"].values():
        phase_counts[a["phase"]] = phase_counts.get(a["phase"], 0) + 1
    print(f"  Agent phases: {phase_counts}")

    print(f"\n🌐 Open {out.name} in your browser to explore the fleet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holodeck 3D Fleet Visualizer Demo")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write holodeck_demo.html (default: current dir)",
    )
    args = parser.parse_args()
    run_demo(output_dir=args.output_dir)
