"""Fleet API — FastAPI server for sunset-ecosystem operations.

Endpoints:
  GET  /health          Health check
  GET  /status           Fleet status
  POST /agents          Insert agent
  GET  /agents/{id}     Query agent by ID
  POST /agents/similar  Similarity search
  POST /memory/write    Write to FleetMemory
  POST /memory/query    Query FleetMemory
  POST /swarm/knn       Distributed KNN search
  GET  /cache/stats     CognitiveCache stats
  POST /cache/maintenance Run maintenance
  GET  /tests           Test inventory

Usage:
  uvicorn fleet.fleet_api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fleet.cognitive_cache import CognitiveCache, PredictionEngine
from fleet.fleet_memory import FleetMemory
from swarm.mesh_vector_tables import MeshVectorTable, VectorTableEntry
from swarm.scene_tracker import SceneTracker, CacheStrategy
from swarm.tiered_mesh_storage import TieredMeshStorage, TierConfig
from swarm.vector_swarm import SwarmRouter, VectorSwarm

app = FastAPI(
    title="Sunset Ecosystem API",
    description="Fleet-scale mesh vector database with emergent applications",
    version="0.1.0",
)

# In-memory stores (singleton for demo)
_table = MeshVectorTable(table_id="api")
_storage = TieredMeshStorage(base_table=_table)
_tracker = SceneTracker(_table, strategy=CacheStrategy())
_cache = CognitiveCache(_storage, _tracker)
_memory = FleetMemory(node_id="api")


# ── Models ──────────────────────────────────────────────────

class AgentEntry(BaseModel):
    agent_id: str
    vector: list[float]
    timestamp: float | None = None
    node_id: str = "api"
    generation: int = 0
    fitness: float = 0.5
    capability_mask: int = 0
    thermal_pressure: float = 0.0


class SimilarityQuery(BaseModel):
    vector: list[float]
    k: int = 5


class MemoryWrite(BaseModel):
    agent_id: str
    vector: list[float]
    timestamp: float | None = None


class MemoryQuery(BaseModel):
    start_time: float
    end_time: float
    filter_fitness: float | None = None


class SwarmKnnQuery(BaseModel):
    vector: list[float]
    k: int = 5
    consistency: str = "quorum"


# ── Health ──────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "timestamp": time.time()}


# ── Status ──────────────────────────────────────────────────

@app.get("/status")
def status() -> dict[str, Any]:
    swarm_dir = Path(__file__).parent.parent / "swarm"
    fleet_dir = Path(__file__).parent.parent / "fleet"
    tests_dir = Path(__file__).parent.parent / "tests"
    docs_dir = Path(__file__).parent.parent / "docs"

    return {
        "swarm_modules": len([f for f in swarm_dir.glob("*.py") if f.is_file()]),
        "fleet_modules": len([f for f in fleet_dir.glob("*.py") if f.is_file()]),
        "test_files": len([f for f in tests_dir.glob("test_*.py") if f.is_file()]),
        "docs": len([f for f in docs_dir.glob("*.md") if f.is_file()]),
        "api_version": "0.1.0",
    }


# ── Agents ──────────────────────────────────────────────────

@app.post("/agents")
def insert_agent(entry: AgentEntry) -> dict[str, Any]:
    vector = np.array(entry.vector, dtype=np.float32)
    vte = VectorTableEntry(
        agent_id=entry.agent_id,
        vector=vector,
        timestamp=entry.timestamp or time.time(),
        node_id=entry.node_id,
        generation=entry.generation,
        fitness=entry.fitness,
        capability_mask=entry.capability_mask,
        thermal_pressure=entry.thermal_pressure,
        signature=f"api_{entry.agent_id}",
    )
    _table.insert(vte, skip_verify=True)
    return {"status": "inserted", "agent_id": entry.agent_id}


@app.get("/agents/{agent_id}")
def get_agent(agent_id: str) -> dict[str, Any]:
    entry = _table.query(agent_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return {
        "agent_id": entry.agent_id,
        "vector": entry.vector.tolist(),
        "timestamp": entry.timestamp,
        "node_id": entry.node_id,
        "generation": entry.generation,
        "fitness": entry.fitness,
    }


@app.post("/agents/similar")
def similar_agents(query: SimilarityQuery) -> dict[str, Any]:
    vector = np.array(query.vector, dtype=np.float32)
    entries = _table.query_similarity_sorted(vector, k=query.k)
    return {
        "query": query.vector,
        "k": query.k,
        "results": [
            {
                "agent_id": e.agent_id,
                "fitness": e.fitness,
                "vector_sample": e.vector[:3].tolist(),
            }
            for e in entries
        ],
    }


# ── Memory ──────────────────────────────────────────────────

@app.post("/memory/write")
def memory_write(req: MemoryWrite) -> dict[str, Any]:
    vector = np.array(req.vector, dtype=np.float32)
    ts = req.timestamp or time.time()
    _memory.remember(req.agent_id, vector, timestamp=ts)
    return {"status": "written", "agent_id": req.agent_id, "timestamp": ts}


@app.post("/memory/query")
def memory_query(req: MemoryQuery) -> dict[str, Any]:
    filter_fn = None
    if req.filter_fitness:
        filter_fn = lambda e: e.fitness >= req.filter_fitness
    # Query via FleetMemory.recall
    results = []
    for shard in _memory._shards.values():
        for entry in shard.all_entries():
            if req.start_time <= entry.timestamp <= req.end_time:
                if filter_fn is None or filter_fn(entry):
                    results.append(entry)

    return {
        "count": len(results),
        "results": [
            {
                "agent_id": e.agent_id,
                "timestamp": e.timestamp,
                "fitness": e.fitness,
            }
            for e in results[:50]
        ],
    }


@app.get("/memory/shards")
def memory_shards() -> dict[str, Any]:
    report = _memory.get_shard_report()
    return {"shards": report.get("shard_ids", []), "count": len(report.get("shard_ids", []))}


# ── Cache ───────────────────────────────────────────────────

@app.get("/cache/stats")
def cache_stats() -> dict[str, Any]:
    return _cache.stats


@app.post("/cache/maintenance")
def cache_maintenance() -> dict[str, Any]:
    _cache.run_maintenance()
    return {"status": "maintenance_completed"}


# ── Swarm ───────────────────────────────────────────────────

@app.post("/swarm/knn")
def swarm_knn(query: SwarmKnnQuery) -> dict[str, Any]:
    # Simple demo: single node
    router = SwarmRouter()
    router.register_node("local", ["default"], _table)
    swarm = VectorSwarm(router)
    vector = np.array(query.vector, dtype=np.float32)
    results = swarm.query_knn(vector, k=query.k, consistency=query.consistency)
    return {
        "query": query.vector,
        "k": query.k,
        "consistency": query.consistency,
        "results": [
            {
                "agent_id": e.agent_id,
                "distance": dist,
                "fitness": e.fitness,
            }
            for e, dist in results
        ],
    }


# ── Tests ───────────────────────────────────────────────────

@app.get("/tests")
def test_inventory() -> dict[str, Any]:
    tests_dir = Path(__file__).parent.parent / "tests"
    test_files = [f.name for f in tests_dir.glob("test_*.py") if f.is_file()]
    return {
        "test_files": test_files,
        "count": len(test_files),
    }


# ── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
