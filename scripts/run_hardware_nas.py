#!/usr/bin/env python3
"""CLI: Run hardware-conditional NAS for a specified hardware profile.

Usage:
    python scripts/run_hardware_nas.py --profile jetson
    python scripts/run_hardware_nas.py --profile oracle1 --generations 20 --population 30
    python scripts/run_hardware_nas.py --profile laptop --max-evals 200 --output configs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure imports work (repo has no root __init__.py)
_NERVE = Path(__file__).parent.parent / "nerve"
sys.path.insert(0, str(_NERVE.parent))

from experiments.hardware_nas import (
    HardwareConditionalNAS,
    run_nas_for_profile,
    oracle1_profile,
    jetson_profile,
    laptop_profile,
)


PROFILE_MAP = {
    "oracle1": oracle1_profile,
    "jetson": jetson_profile,
    "laptop": laptop_profile,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hardware-Conditional NAS for RoomGrid configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --profile jetson
  %(prog)s --profile oracle1 --generations 20 --population 30 --max-evals 200
  %(prog)s --profile laptop --output top5.json
""",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILE_MAP.keys()),
        required=True,
        help="Hardware profile to optimize for",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=20,
        help="Population size for aging evolution (default: 20)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations (default: 10)",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=100,
        help="Maximum config evaluations (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON file to write top-5 configs (default: print to stdout)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top configs to output (default: 5)",
    )

    args = parser.parse_args()

    profile = PROFILE_MAP[args.profile]
    profile_name = args.profile

    nas = HardwareConditionalNAS(profile, max_evals=args.max_evals, seed=args.seed)

    print(f"🔧 Hardware-Conditional NAS")
    print(f"   Profile     : {profile_name} ({profile['device']})")
    print(f"   RAM         : {profile['ram_gb']} GB")
    print(f"   CPU cores   : {profile['cpu_cores']}")
    print(f"   GPU         : {profile['gpu']}")
    print(f"   Search space: {nas.SEARCH_SPACE_SIZE} configs")
    print(f"   Population  : {args.population}")
    print(f"   Generations : {args.generations}")
    print(f"   Max evals   : {args.max_evals}")
    print(f"   Seed        : {args.seed}")
    print()

    def progress_cb(stage, a, b, result):
        if stage == "init":
            print(f"   [init] {a:2d}/{b}  {result.config}  tps={result.ticks_per_second:8.1f}  mem={result.memory_mb:7.2f}MB")
        elif stage == "gen":
            if a % 2 == 0 or a == b - 1:
                print(f"   [gen {a:2d}] {result.config}  tps={result.ticks_per_second:8.1f}  div={result.diversity:.3f}  stab={result.stability:.3f}")
        elif stage == "gen_infeasible":
            print(f"   [gen {a:2d}] infeasible config skipped")

    frontier = nas.aging_evolution(
        population_size=args.population,
        generations=args.generations,
        progress_cb=progress_cb,
    )

    print()
    print(f"✅ Evolution complete: {nas.eval_count} evaluations")
    print(f"   Pareto frontier size: {len(frontier)}")

    # Rank by composite score
    scored = []
    for p in frontier:
        score = (
            p.get("ticks_per_second", 0)
            * p.get("diversity", 0)
            * p.get("stability", 0)
        ) / (1.0 + p.get("memory_mb", 1))
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = scored[: args.top_k]

    output = {
        "profile": profile_name,
        "device": profile["device"],
        "search_space_size": nas.SEARCH_SPACE_SIZE,
        "eval_count": nas.eval_count,
        "pareto_frontier_size": len(frontier),
        "top_configs": [
            {
                "rank": i + 1,
                "composite_score": round(score, 6),
                "config": {
                    "n_rooms": p["n_rooms"],
                    "d_latent": p["d_latent"],
                    "h_history": p["h_history"],
                    "l_signal": p["l_signal"],
                    "chaos_decay": p["chaos_decay"],
                    "route_density": p["route_density"],
                },
                "metrics": {
                    "ticks_per_second": round(p["ticks_per_second"], 2),
                    "memory_mb": round(p["memory_mb"], 2),
                    "diversity": round(p["diversity"], 4),
                    "stability": round(p["stability"], 4),
                },
            }
            for i, (score, p) in enumerate(top_k)
        ],
    }

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\n📁 Written to: {out_path.absolute()}")
    else:
        print("\n📊 Top configs:")
        print(json.dumps(output, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
