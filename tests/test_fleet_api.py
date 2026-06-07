"""Tests for Fleet API.

Covers:
- Health endpoint
- Status endpoint
- Agent insert and query
- Similarity search
- Memory write and query
- Cache stats and maintenance
- Swarm KNN
- Test inventory
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fleet.fleet_api import app

client = TestClient(app)


class TestHealth:
    def test_health(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestStatus:
    def test_status(self) -> None:
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "swarm_modules" in data
        assert "fleet_modules" in data
        assert data["api_version"] == "0.1.0"


class TestAgents:
    def test_insert_and_query(self) -> None:
        # Insert
        response = client.post("/agents", json={
            "agent_id": "test_agent_1",
            "vector": [1.0, 0.0, 0.5],
            "fitness": 0.8,
        })
        assert response.status_code == 200
        assert response.json()["status"] == "inserted"

        # Query
        response = client.get("/agents/test_agent_1")
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "test_agent_1"
        assert data["fitness"] == 0.8

    def test_query_not_found(self) -> None:
        response = client.get("/agents/nonexistent")
        assert response.status_code == 404

    def test_similar(self) -> None:
        # Insert multiple
        for i in range(3):
            client.post("/agents", json={
                "agent_id": f"sim_{i}",
                "vector": [float(i), 0.0, 0.0],
                "fitness": 0.5 + i * 0.1,
            })

        response = client.post("/agents/similar", json={
            "vector": [1.0, 0.0, 0.0],
            "k": 2,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 2


class TestMemory:
    def test_write_and_query(self) -> None:
        now = 1717800000.0
        response = client.post("/memory/write", json={
            "agent_id": "mem_1",
            "vector": [1.0, 2.0, 3.0],
            "timestamp": now,
        })
        assert response.status_code == 200

        response = client.post("/memory/query", json={
            "start_time": now - 1,
            "end_time": now + 1,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1

    def test_shards(self) -> None:
        response = client.get("/memory/shards")
        assert response.status_code == 200
        assert "shards" in response.json()


class TestCache:
    def test_stats(self) -> None:
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "prediction_hits" in data

    def test_maintenance(self) -> None:
        response = client.post("/cache/maintenance")
        assert response.status_code == 200
        assert response.json()["status"] == "maintenance_completed"


class TestSwarm:
    def test_knn(self) -> None:
        # Insert entries
        for i in range(3):
            client.post("/agents", json={
                "agent_id": f"swarm_{i}",
                "vector": [float(i), 0.0, 0.0],
            })

        response = client.post("/swarm/knn", json={
            "vector": [1.0, 0.0, 0.0],
            "k": 2,
            "consistency": "all",
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 2


class TestTests:
    def test_inventory(self) -> None:
        response = client.get("/tests")
        assert response.status_code == 200
        data = response.json()
        assert "test_files" in data
        assert data["count"] > 0
