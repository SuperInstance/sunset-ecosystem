"""Root conftest — sets up turbovec mock before any test imports it."""

from __future__ import annotations

import sys
import types

import numpy as np

# ── Mock turbovec before any test file imports swarm.vector_table ──
if "turbovec" not in sys.modules:
    _mock_turbovec = types.ModuleType("turbovec")

    class _MockIdMapIndex:
        """Minimal stand-in for turbovec.IdMapIndex."""

        def __init__(self, dim: int, bit_width: int = 4) -> None:
            self.dim = dim
            self.bit_width = bit_width
            self._vectors: dict[int, np.ndarray] = {}

        def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
            for vec, aid in zip(vectors, ids):
                self._vectors[int(aid)] = vec.copy()

        def search(
            self,
            query: np.ndarray,
            k: int = 10,
            allowlist: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            if not self._vectors:
                return (
                    np.zeros((1, k), dtype=np.float32),
                    np.zeros((1, k), dtype=np.uint64),
                )
            q = query[0]
            candidates = list(self._vectors.items())
            if allowlist is not None:
                allowed = set(int(a) for a in allowlist)
                candidates = [(aid, v) for aid, v in candidates if aid in allowed]

            qn = q / (np.linalg.norm(q) + 1e-8)
            sims: list[tuple[int, float]] = []
            for aid, vec in candidates:
                vn = vec / (np.linalg.norm(vec) + 1e-8)
                sims.append((aid, float(np.dot(qn, vn))))
            sims.sort(key=lambda x: x[1], reverse=True)
            top = sims[:k]
            while len(top) < k:
                top.append((0, 0.0))
            scores = np.array([[s for _, s in top]], dtype=np.float32)
            ids_arr = np.array([[aid for aid, _ in top]], dtype=np.uint64)
            return scores, ids_arr

        def remove(self, agent_id: int) -> bool:
            return self._vectors.pop(agent_id, None) is not None

        def contains(self, agent_id: int) -> bool:
            return agent_id in self._vectors

        def prepare(self) -> None:
            pass

        def write(self, path: str) -> None:
            pass

        @classmethod
        def load(cls, path: str) -> "_MockIdMapIndex":
            return cls(dim=256)

    _mock_turbovec.IdMapIndex = _MockIdMapIndex  # type: ignore[attr-defined]
    sys.modules["turbovec"] = _mock_turbovec

    # ── Clear cached downstream modules so re-imports use the mock ──
    # Do NOT delete "turbovec" itself — that would let the real module load.
    for _mod_name in list(sys.modules):
        if _mod_name in (
            "swarm.flux_vector_table",
            "swarm.vector_table",
            "sunset.turbovec",
        ):
            del sys.modules[_mod_name]


# ── Mock plato_core so pytest.importorskip("plato_core") passes ──
# Some test files conditionally skip when plato_core is unavailable.
# Installing a full mock here ensures those tests always run with a
# consistent interface, and prevents partial mocks in individual test
# files from leaking incomplete state.
if "plato_core" not in sys.modules:
    _mock_plato = types.ModuleType("plato_core")
    _mock_plato_types = types.ModuleType("plato_core.types")

    class _MockLamportClock:
        def __init__(self, node_id: int = 0) -> None:
            self._tick = 0
            self.node_id = node_id

        def tick(self) -> int:
            self._tick += 1
            return self._tick

        def update(self, other: int) -> int:
            self._tick = max(self._tick, other) + 1
            return self._tick

    class _MockLifecycleEvent:
        def __init__(
            self,
            from_state=None,
            to_state=None,
            reason="",
            lamport=0,
        ) -> None:
            self.from_state = from_state
            self.to_state = to_state
            self.reason = reason
            self.lamport = lamport

    class _MockTileLifecycle:
        ACTIVE = "active"
        SUPERSEDED = "superseded"
        ARCHIVED = "archived"

    class _MockTileType:
        CHECKPOINT = "checkpoint"
        PREDICTION = "prediction"
        EVALUATION = "evaluation"
        METRICS = "metrics"
        DECISION = "decision"
        EPISTEME = "episteme"
        HYBRID = "hybrid"
        BIRTH = "birth"
        SEED = "seed"
        REFINEMENT = "refinement"
        INTEGRATION = "integration"

    class _MockTrainingTile:
        def __init__(self, **kwargs) -> None:
            self.tile_id = kwargs.get("tile_id", "")
            self.tile_type = kwargs.get("tile_type", _MockTileType.METRICS)
            self.room = kwargs.get("room", "")
            self.description = kwargs.get("description", "")
            self.state = kwargs.get("state", _MockTileLifecycle.ACTIVE)
            self.lamport = kwargs.get("lamport", 0)
            self.lifecycle_events = list(kwargs.get("lifecycle_events", []))
            self.content_hash = kwargs.get("content_hash", "")
            self.signature = kwargs.get("signature", "")
            self.name = kwargs.get("name", "")
            self._payload = kwargs.get("_payload", {})
            # Store any extra kwargs for round-trip fidelity
            self._extra = {
                k: v
                for k, v in kwargs.items()
                if k
                not in {
                    "tile_id",
                    "tile_type",
                    "room",
                    "description",
                    "state",
                    "lamport",
                    "lifecycle_events",
                    "content_hash",
                    "signature",
                    "name",
                    "_payload",
                }
            }

        def is_active(self) -> bool:
            return self.state == _MockTileLifecycle.ACTIVE

        def transition(self, new_state, reason="", lamport=0) -> None:
            self.state = new_state
            self.lifecycle_events.append(
                _MockLifecycleEvent(
                    from_state=self.state,
                    to_state=new_state,
                    reason=reason,
                    lamport=lamport,
                )
            )

        def to_dict(self) -> dict:
            return {
                "tile_id": self.tile_id,
                "tile_type": self.tile_type,
                "room": self.room,
                "description": self.description,
                "state": self.state,
                "lamport": self.lamport,
                "lifecycle_events": [
                    {
                        "from_state": e.from_state,
                        "to_state": e.to_state,
                        "reason": e.reason,
                        "lamport": e.lamport,
                    }
                    for e in self.lifecycle_events
                ],
                "content_hash": self.content_hash,
                "signature": self.signature,
                "name": self.name,
                "_payload": self._payload,
                **self._extra,
            }

        @classmethod
        def from_dict(cls, d: dict) -> "_MockTrainingTile":
            events = [
                _MockLifecycleEvent(
                    from_state=e.get("from_state"),
                    to_state=e.get("to_state"),
                    reason=e.get("reason", ""),
                    lamport=e.get("lamport", 0),
                )
                for e in d.get("lifecycle_events", [])
            ]
            kwargs = {k: v for k, v in d.items() if k != "lifecycle_events"}
            kwargs["lifecycle_events"] = events
            return cls(**kwargs)

    def _mock_content_hash(data: str) -> str:
        import hashlib

        if isinstance(data, bytes):
            return hashlib.sha256(data).hexdigest()[:16]
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    _mock_plato_types.LamportClock = _MockLamportClock
    _mock_plato_types.LifecycleEvent = _MockLifecycleEvent
    _mock_plato_types.TileLifecycle = _MockTileLifecycle
    _mock_plato_types.TileType = _MockTileType
    _mock_plato_types.TrainingTile = _MockTrainingTile
    _mock_plato_types.content_hash = _mock_content_hash
    _mock_plato.types = _mock_plato_types
    sys.modules["plato_core"] = _mock_plato
    sys.modules["plato_core.types"] = _mock_plato_types
