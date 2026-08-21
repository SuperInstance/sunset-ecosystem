#!/usr/bin/env python3
"""
End-to-End Sunset Ecosystem Demo
================================

Exercises the full stack in one script:
  1. NerveTopology — signal → perception → routing → grid
  2. AutoBreeder — tournament + rebirth (thermal-aware)
  3. Agentic Compiler — auto-detects hardware, compiles hot paths

Usage:
    python3 scripts/demo_full_stack.py

Output:
    JSON report with performance metrics and breeding log.
"""

from __future__ import annotations

import json, time, sys, random
import numpy as np

from nerve.topology import NerveTopology
from swarm.breeder_daemon import AutoBreeder
from swarm.thermal import ThermalBudget, DeviceType


def run_demo(n_rooms: int = 500, n_ticks: int = 200):
    print(f"=== Sunset Ecosystem Full-Stack Demo ===")
    print(f"Rooms: {n_rooms} | Ticks: {n_ticks}")
    print()

    # ── Setup ──────────────────────────────────────────────
    t0 = time.perf_counter()
    topo = NerveTopology(n_rooms=n_rooms)
    thermal = ThermalBudget()
    breeder = AutoBreeder(topo.grid, thermal, interval=30, cold_threshold=5)
    setup_ms = (time.perf_counter() - t0) * 1000

    # Enable agentic compiler (auto-optimizes after warmup)
    topo.enable_compiler(auto_compile_interval=50)

    print(f"[setup] {setup_ms:.1f}ms")
    print(f"  Topology: {topo}")
    print(f"  Backend: {topo.grid.__repr__().split('backend=')[1].rstrip(')')}")
    print(f"  Compiler: {topo._compiler is not None}")
    print()

    # ── Run ticks ─────────────────────────────────────────
    latencies = []
    compile_events = []
    breeding_events = []

    for tick in range(n_ticks):
        # Generate varied signals (some structured, some random)
        if tick % 20 < 10:
            # Structured: sine waves with phase shift per fiber
            signals = {
                fid: np.sin(np.linspace(0, 4 * np.pi, 64) + i * 0.5).astype(np.float32)
                for i, fid in enumerate(topo.fibers)
            }
        else:
            # Random noise
            signals = {
                fid: np.random.randn(64).astype(np.float32) * 0.5 for fid in topo.fibers
            }

        r = topo.tick(signals)
        latencies.append(r.latency_ms)

        if r.compiled_funcs:
            compile_events.append(
                {
                    "tick": r.tick,
                    "functions": r.compiled_funcs,
                }
            )

        # Run breeder every 30 ticks
        if tick > 50 and tick % 30 == 0:
            reborn = breeder.auto_breed()
            if reborn:
                breeding_events.append(
                    {
                        "tick": tick,
                        "reborn": reborn,
                    }
                )

    total_ms = sum(latencies)
    avg_ms = np.mean(latencies)
    p50_ms = float(np.percentile(latencies, 50))
    p99_ms = float(np.percentile(latencies, 99))
    max_ms = max(latencies)

    # ── Report ─────────────────────────────────────────────
    print(f"[results] {n_ticks} ticks in {total_ms:.0f}ms")
    print(
        f"  Per tick: avg={avg_ms:.2f}ms p50={p50_ms:.2f}ms p99={p99_ms:.2f}ms max={max_ms:.2f}ms"
    )
    print(f"  Throughput: {n_ticks / (total_ms / 1000):.0f} ticks/s")
    print()

    print(f"[grid] {topo.grid}")
    print(f"  Stats: {topo.grid.stats}")
    print()

    print(f"[breeding] {len(breeding_events)} cycles")
    total_reborn = sum(len(e["reborn"]) for e in breeding_events)
    print(f"  Total reborn: {total_reborn}")
    for e in breeding_events[:3]:
        print(f"  tick {e['tick']}: {len(e['reborn'])} rooms reborn")
    if len(breeding_events) > 3:
        print(f"  ... and {len(breeding_events) - 3} more cycles")
    print()

    print(f"[compiler] {len(compile_events)} auto-compile events")
    for e in compile_events[:3]:
        print(f"  tick {e['tick']}: {e['functions']}")
    print()

    # JSON report for FM
    report = {
        "config": {
            "n_rooms": n_rooms,
            "n_ticks": n_ticks,
            "n_fibers": topo.n_fibers,
        },
        "timing": {
            "setup_ms": setup_ms,
            "total_ms": total_ms,
            "avg_ms": avg_ms,
            "p50_ms": p50_ms,
            "p99_ms": p99_ms,
            "max_ms": max_ms,
            "ticks_per_sec": n_ticks / (total_ms / 1000),
        },
        "grid": topo.grid.stats,
        "breeding": {
            "cycles": len(breeding_events),
            "total_reborn": total_reborn,
            "events": breeding_events,
        },
        "compiler": {
            "enabled": topo._compiler is not None,
            "auto_compile_interval": topo._compiler_auto_compile_interval,
            "events": compile_events,
        },
        "thermal": {
            "total_max": thermal.total_max,
            "total_current": thermal.total_current,
        },
    }

    with open("/tmp/sunset_demo_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] written to /tmp/sunset_demo_report.json")
    print()
    print("=== Demo Complete ===")
    return report


if __name__ == "__main__":
    n_rooms = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    run_demo(n_rooms, n_ticks)
