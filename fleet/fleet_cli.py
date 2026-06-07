"""Fleet CLI — Command-line interface for sunset-ecosystem operations.

Commands:
  status      Show fleet status (nodes, shards, tests, modules)
  query       Query agent by ID or similarity
  insert      Insert a new agent entry
  memory      Interact with FleetMemory (write, query, shards)
  cache       Run CognitiveCache maintenance and show stats
  swarm       Distributed queries across nodes
  test        Run test suite
  bridge      List xlang-foundation bridges and their status

Usage:
  python -m fleet.fleet_cli status
  python -m fleet.fleet_cli query --agent-id agent_42
  python -m fleet.fleet_cli memory --write --agent-id agent_1 --vector "1.0,0.0,0.5"
  python -m fleet.fleet_cli test --module test_fleet_memory
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet.cognitive_cache import CognitiveCache, PredictionEngine
from fleet.fleet_memory import FleetMemory
from swarm.mesh_vector_tables import MeshVectorTable, VectorTableEntry
from swarm.scene_tracker import SceneTracker, CacheStrategy
from swarm.tiered_mesh_storage import TieredMeshStorage, TierConfig
from swarm.vector_swarm import SwarmRouter, VectorSwarm


def cmd_status(args: argparse.Namespace) -> int:
    """Show fleet status."""
    # Count modules
    swarm_dir = Path(__file__).parent.parent / "swarm"
    fleet_dir = Path(__file__).parent.parent / "fleet"
    tests_dir = Path(__file__).parent.parent / "tests"
    docs_dir = Path(__file__).parent.parent / "docs"

    swarm_modules = len([f for f in swarm_dir.glob("*.py") if f.is_file() and not f.name.startswith("_")])
    fleet_modules = len([f for f in fleet_dir.glob("*.py") if f.is_file() and not f.name.startswith("_")])
    test_files = len([f for f in tests_dir.glob("test_*.py") if f.is_file()])
    docs = len([f for f in docs_dir.glob("*.md") if f.is_file()])

    print("=" * 60)
    print("SUNSET ECOSYSTEM — Fleet Status")
    print("=" * 60)
    print(f"  Swarm modules:       {swarm_modules}")
    print(f"  Fleet modules:       {fleet_modules}")
    print(f"  Test files:          {test_files}")
    print(f"  Documentation:       {docs}")
    print(f"  Total modules:       {swarm_modules + fleet_modules}")
    print()
    print("  Core Components:")
    print("    - MeshVectorTable (CRDT vector store)")
    print("    - TieredMeshStorage (hot/warm/cold)")
    print("    - HNSW Mesh Table (approximate NN)")
    print("    - FleetMemory (time-sharded long-term)")
    print("    - VectorSwarm (distributed search)")
    print("    - CognitiveCache (predictive caching)")
    print()
    print("  Bridges:")
    print("    - Quanta VDB (C++ HNSW)")
    print("    - xlang Agent (event-driven runtime)")
    print("    - caslang (sandboxed execution)")
    print("    - xMind (AgentFlow orchestration)")
    print("=" * 60)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Query the mesh table."""
    table = MeshVectorTable(table_id="cli")

    if args.agent_id:
        entry = table.query(args.agent_id)
        if entry:
            print(json.dumps({
                "agent_id": entry.agent_id,
                "timestamp": entry.timestamp,
                "node_id": entry.node_id,
                "generation": entry.generation,
                "fitness": entry.fitness,
                "capability_mask": entry.capability_mask,
                "thermal_pressure": entry.thermal_pressure,
                "vector_shape": entry.vector.shape,
                "vector_sample": entry.vector[:3].tolist(),
            }, indent=2))
        else:
            print(f"Agent '{args.agent_id}' not found.")
            return 1

    elif args.similar:
        vector = np.array([float(x) for x in args.similar.split(",")], dtype=np.float32)
        entries = table.query_similarity_sorted(vector, k=args.k)
        print(f"Top-{args.k} similar entries:")
        for entry in entries:
            print(f"  {entry.agent_id}: fitness={entry.fitness:.3f}")

    return 0


def cmd_insert(args: argparse.Namespace) -> int:
    """Insert an entry."""
    table = MeshVectorTable(table_id="cli")
    vector = np.array([float(x) for x in args.vector.split(",")], dtype=np.float32)
    entry = VectorTableEntry(
        agent_id=args.agent_id,
        vector=vector,
        timestamp=time.time(),
        node_id=args.node_id,
        generation=args.generation,
        fitness=args.fitness,
        signature=f"cli_{args.agent_id}",
    )
    table.insert(entry, skip_verify=True)
    print(f"Inserted {args.agent_id} with fitness {args.fitness}")
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    """FleetMemory operations."""
    memory = FleetMemory()

    if args.write:
        vector = np.array([float(x) for x in args.vector.split(",")], dtype=np.float32) if args.vector else np.zeros(3, dtype=np.float32)
        memory.write(args.agent_id, vector, timestamp=time.time())
        print(f"Wrote {args.agent_id} to FleetMemory")

    elif args.query:
        now = time.time()
        results = memory.query(
            start_time=now - 3600,
            end_time=now,
        )
        print(f"Found {len(results)} entries in last hour:")
        for entry in results[:10]:
            print(f"  {entry.agent_id}: fitness={entry.fitness:.3f}")

    elif args.shards:
        shards = memory.get_shard_ids()
        print(f"Active shards: {shards}")
        print(f"Shard count: {len(shards)}")

    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    """CognitiveCache operations."""
    base = MeshVectorTable(table_id="cli_cache")
    storage = TieredMeshStorage(base_table=base)
    tracker = SceneTracker(base, strategy=CacheStrategy())
    cache = CognitiveCache(storage, tracker)

    if args.maintenance:
        cache.run_maintenance()
        print("Maintenance completed.")

    stats = cache.stats
    print("Cache Stats:")
    print(json.dumps(stats, indent=2, default=str))
    return 0


def cmd_swarm(args: argparse.Namespace) -> int:
    """VectorSwarm operations."""
    router = SwarmRouter()
    node1 = MeshVectorTable(table_id="node1")
    node2 = MeshVectorTable(table_id="node2")
    router.register_node("node1", ["shard1"], node1)
    router.register_node("node2", ["shard2"], node2)
    swarm = VectorSwarm(router)

    if args.query_id:
        results = swarm.query_by_id(args.query_id)
        print(f"Query results from {len(results)} nodes:")
        for result in results:
            print(f"  {result.source}: {len(result.entries)} entries")

    elif args.knn:
        vector = np.array([float(x) for x in args.knn.split(",")], dtype=np.float32)
        results = swarm.query_knn(vector, k=args.k)
        print(f"KNN results:")
        for entry, dist in results:
            print(f"  {entry.agent_id}: distance={dist:.3f}")

    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Run tests."""
    import subprocess
    cmd = ["python3", "-m", "pytest", "-v"]
    if args.module:
        cmd.append(f"tests/{args.module}.py")
    else:
        cmd.append("tests/")
    if args.failfast:
        cmd.append("-x")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def cmd_bridge(args: argparse.Namespace) -> int:
    """List bridge status."""
    bridges = {
        "Quanta VDB": "swarm/hnsw_mesh_table.py — HNSW hybrid index",
        "xlang Agent": "fleet/xlang_agent_bridge.py — AgentFlow execution",
        "caslang": "fleet/caslang_executor.py — Sandbox execution",
        "xMind": "fleet/xlang_agent_bridge.py — YAML blueprints",
    }
    print("xlang-foundation Bridges:")
    for name, desc in bridges.items():
        print(f"  {name}: {desc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fleet",
        description="Sunset Ecosystem CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # status
    subparsers.add_parser("status", help="Show fleet status")

    # query
    query_parser = subparsers.add_parser("query", help="Query agents")
    query_parser.add_argument("--agent-id", help="Query by agent ID")
    query_parser.add_argument("--similar", help="Query vector (comma-separated)")
    query_parser.add_argument("-k", type=int, default=5, help="Top-k results")

    # insert
    insert_parser = subparsers.add_parser("insert", help="Insert an agent")
    insert_parser.add_argument("--agent-id", required=True)
    insert_parser.add_argument("--vector", required=True, help="Comma-separated floats")
    insert_parser.add_argument("--node-id", default="cli")
    insert_parser.add_argument("--generation", type=int, default=0)
    insert_parser.add_argument("--fitness", type=float, default=0.5)

    # memory
    memory_parser = subparsers.add_parser("memory", help="FleetMemory operations")
    memory_parser.add_argument("--write", action="store_true", help="Write entry")
    memory_parser.add_argument("--query", action="store_true", help="Query entries")
    memory_parser.add_argument("--shards", action="store_true", help="List shards")
    memory_parser.add_argument("--agent-id", help="Agent ID")
    memory_parser.add_argument("--vector", help="Comma-separated vector")

    # cache
    cache_parser = subparsers.add_parser("cache", help="CognitiveCache operations")
    cache_parser.add_argument("--maintenance", action="store_true", help="Run maintenance")

    # swarm
    swarm_parser = subparsers.add_parser("swarm", help="VectorSwarm operations")
    swarm_parser.add_argument("--query-id", help="Query by agent ID")
    swarm_parser.add_argument("--knn", help="KNN query vector")
    swarm_parser.add_argument("-k", type=int, default=5)

    # test
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--module", help="Specific test module")
    test_parser.add_argument("-x", "--failfast", action="store_true", help="Stop on first failure")

    # bridge
    subparsers.add_parser("bridge", help="List bridge status")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "status": cmd_status,
        "query": cmd_query,
        "insert": cmd_insert,
        "memory": cmd_memory,
        "cache": cmd_cache,
        "swarm": cmd_swarm,
        "test": cmd_test,
        "bridge": cmd_bridge,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
