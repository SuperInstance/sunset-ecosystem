"""swarm/arrow_mesh.py — Arrow serialization for mesh gossip payloads.

Wraps MeshVectorGossip data structures (GossipDigest, DeltaBatch) as
Apache Arrow RecordBatches.  Benefits:
- Zero-copy deserialization (mmap-compatible)
- Schema evolution (new columns don't break old readers)
- SIMD-friendly columnar layout for vector deltas
- Language-agnostic interop (Rust, C++, Java, JS all read Arrow)
- Optional: Arrow Flight gRPC transport (future path)

Usage
-----
    from swarm.arrow_mesh import ArrowMeshCodec
    from swarm.mesh_vector_gossip import GossipDigest, DeltaBatch

    codec = ArrowMeshCodec()

    # Encode digest for wire
    batch = codec.digest_to_arrow(GossipDigest(...))
    ipc_bytes = codec.serialize_batch(batch)

    # Decode on peer
    batch2 = codec.deserialize_batch(ipc_bytes)
    digest = codec.digest_from_arrow(batch2)

Graceful degradation: if pyarrow is not installed, falls back to
pickle/JSON encoding with a warning.
"""

from __future__ import annotations

import json
import logging
import struct
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional pyarrow — fleet nodes without it still function (JSON fallback)
try:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    HAS_PYARROW = True
except Exception as exc:  # noqa: BLE001
    pa = None  # type: ignore[assignment]
    ipc = None  # type: ignore[assignment]
    HAS_PYARROW = False
    warnings.warn(f"pyarrow not available; arrow_mesh using JSON fallback ({exc})")


# ── Arrow schemas (columnar) ──────────────────────────────────────────────

_DIGEST_SCHEMA_FIELDS: List[pa.Field] = []
_DELTA_SCHEMA_FIELDS: List[pa.Field] = []

if HAS_PYARROW:
    _DIGEST_SCHEMA_FIELDS = [
        pa.field("node_id", pa.string(), nullable=False),
        pa.field("agent_count", pa.int64(), nullable=False),
        pa.field("version_vector_keys", pa.list_(pa.int64()), nullable=False),
        pa.field("version_vector_vals", pa.list_(pa.int64()), nullable=False),
        pa.field("bitfield", pa.string(), nullable=False),
        pa.field("max_wall_time", pa.float64(), nullable=False),
    ]
    _DIGEST_SCHEMA = pa.schema(_DIGEST_SCHEMA_FIELDS)

    _DELTA_SCHEMA_FIELDS = [
        pa.field("node_id", pa.string(), nullable=False),
        pa.field("sequence", pa.int64(), nullable=False),
        pa.field("agent_id", pa.list_(pa.int64()), nullable=False),
        pa.field("vector", pa.list_(pa.float64()), nullable=False),
        pa.field("score", pa.list_(pa.float64()), nullable=True),
        pa.field("timestamp", pa.list_(pa.float64()), nullable=True),
        pa.field("metadata_json", pa.list_(pa.string()), nullable=True),
    ]
    _DELTA_SCHEMA = pa.schema(_DELTA_SCHEMA_FIELDS)


# ── Codec ───────────────────────────────────────────────────────────────


class ArrowMeshCodec:
    """Serialize / deserialize mesh gossip structures via Arrow."""

    def __init__(self, *, use_arrow: bool = True):
        self.use_arrow = use_arrow and HAS_PYARROW
        if use_arrow and not HAS_PYARROW:
            logger.warning("Arrow requested but pyarrow missing; using JSON fallback")

    # ── Digest ─────────────────────────────────────────────────────────

    def digest_to_arrow(self, digest: Any) -> Any:  # pa.RecordBatch
        """Convert GossipDigest to Arrow RecordBatch."""
        if not self.use_arrow:
            raise RuntimeError("pyarrow not available")

        keys = list(digest.version_vector.keys())
        vals = list(digest.version_vector.values())
        arrays = [
            pa.array([digest.node_id]),
            pa.array([digest.agent_count]),
            pa.array([keys]),
            pa.array([vals]),
            pa.array([digest.bitfield]),
            pa.array([digest.max_wall_time]),
        ]
        return pa.RecordBatch.from_arrays(
            arrays, [f.name for f in _DIGEST_SCHEMA_FIELDS]
        )

    def digest_from_arrow(self, batch: Any) -> Dict[str, Any]:
        """Convert Arrow RecordBatch back to dict compatible with GossipDigest."""
        if not self.use_arrow:
            raise RuntimeError("pyarrow not available")

        # Single-row batch
        d = {
            "node_id": batch.column("node_id")[0].as_py(),
            "agent_count": batch.column("agent_count")[0].as_py(),
            "version_vector": dict(
                zip(
                    batch.column("version_vector_keys")[0].as_py(),
                    batch.column("version_vector_vals")[0].as_py(),
                )
            ),
            "bitfield": batch.column("bitfield")[0].as_py(),
            "max_wall_time": batch.column("max_wall_time")[0].as_py(),
        }
        return d

    # ── DeltaBatch ─────────────────────────────────────────────────────

    def deltas_to_arrow(self, batch: Any) -> Any:  # pa.RecordBatch
        """Convert DeltaBatch to Arrow RecordBatch."""
        if not self.use_arrow:
            raise RuntimeError("pyarrow not available")

        node_ids = []
        sequences = []
        agent_ids = []
        vectors = []
        scores = []
        timestamps = []
        metadata_json = []

        for delta in batch.deltas:
            node_ids.append(delta.get("node_id", batch.node_id))
            sequences.append(batch.sequence)
            agent_ids.append(delta.get("agent_id", -1))
            vec = delta.get("vector", [])
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()
            vectors.append(vec)
            scores.append(delta.get("score", None))
            timestamps.append(delta.get("timestamp", time.time()))
            metadata_json.append(json.dumps(delta.get("metadata", {})))

        arrays = [
            pa.array(node_ids),
            pa.array(sequences),
            pa.array(agent_ids),
            pa.array(vectors),
            pa.array(scores),
            pa.array(timestamps),
            pa.array(metadata_json),
        ]
        return pa.RecordBatch.from_arrays(
            arrays, [f.name for f in _DELTA_SCHEMA_FIELDS]
        )

    def deltas_from_arrow(self, record_batch: Any) -> Dict[str, Any]:
        """Convert Arrow RecordBatch back to dict compatible with DeltaBatch."""
        if not self.use_arrow:
            raise RuntimeError("pyarrow not available")

        deltas = []
        n = record_batch.num_rows
        cols = {name: record_batch.column(name) for name in record_batch.schema.names}
        for i in range(n):
            meta = {}
            if cols["metadata_json"][i].as_py():
                try:
                    meta = json.loads(cols["metadata_json"][i].as_py())
                except json.JSONDecodeError:
                    meta = {}
            delta = {
                "node_id": cols["node_id"][i].as_py(),
                "agent_id": cols["agent_id"][i].as_py(),
                "vector": np.array(cols["vector"][i].as_py(), dtype=np.float32),
                "score": cols["score"][i].as_py(),
                "timestamp": cols["timestamp"][i].as_py(),
                "metadata": meta,
            }
            deltas.append(delta)

        return {
            "node_id": cols["node_id"][0].as_py() if n > 0 else "",
            "deltas": deltas,
            "sequence": cols["sequence"][0].as_py() if n > 0 else 0,
        }

    # ── IPC serialization ────────────────────────────────────────────────

    def serialize_batch(self, record_batch: Any) -> bytes:
        """Serialize RecordBatch to IPC bytes (sink / wire format)."""
        if not self.use_arrow:
            raise RuntimeError("pyarrow not available")

        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, record_batch.schema) as writer:
            writer.write_batch(record_batch)
        return sink.getvalue().to_pybytes()

    def deserialize_batch(self, buf: bytes) -> Any:
        """Deserialize IPC bytes back to RecordBatch."""
        if not self.use_arrow:
            raise RuntimeError("pyarrow not available")

        reader = ipc.RecordBatchStreamReader(pa.py_buffer(buf))
        return reader.read_next_batch()

    # ── Fallback JSON ───────────────────────────────────────────────────

    def digest_to_json(self, digest: Any) -> bytes:
        """JSON encoding for nodes without pyarrow."""
        return json.dumps(
            {
                "node_id": digest.node_id,
                "agent_count": digest.agent_count,
                "version_vector": digest.version_vector,
                "bitfield": digest.bitfield,
                "max_wall_time": digest.max_wall_time,
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def digest_from_json(self, buf: bytes) -> Dict[str, Any]:
        """Decode JSON back to digest dict."""
        return json.loads(buf.decode("utf-8"))

    def deltas_to_json(self, batch: Any) -> bytes:
        """JSON encoding for DeltaBatch fallback."""
        payload = {
            "node_id": batch.node_id,
            "sequence": batch.sequence,
            "deltas": [
                {
                    **d,
                    "vector": (
                        d["vector"].tolist()
                        if isinstance(d["vector"], np.ndarray)
                        else d["vector"]
                    ),
                }
                for d in batch.deltas
            ],
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def deltas_from_json(self, buf: bytes) -> Dict[str, Any]:
        """Decode JSON back to DeltaBatch-compatible dict."""
        raw = json.loads(buf.decode("utf-8"))
        for d in raw.get("deltas", []):
            if "vector" in d and not isinstance(d["vector"], np.ndarray):
                d["vector"] = np.array(d["vector"], dtype=np.float32)
        return raw

    # ── Auto-detect ──────────────────────────────────────────────────────

    def serialize_digest(self, digest: Any) -> bytes:
        if self.use_arrow:
            return self.serialize_batch(self.digest_to_arrow(digest))
        return self.digest_to_json(digest)

    def deserialize_digest(self, buf: bytes) -> Dict[str, Any]:
        if self.use_arrow:
            return self.digest_from_arrow(self.deserialize_batch(buf))
        return self.digest_from_json(buf)

    def serialize_deltas(self, batch: Any) -> bytes:
        if self.use_arrow:
            return self.serialize_batch(self.deltas_to_arrow(batch))
        return self.deltas_to_json(batch)

    def deserialize_deltas(self, buf: bytes) -> Dict[str, Any]:
        if self.use_arrow:
            return self.deltas_from_arrow(self.deserialize_batch(buf))
        return self.deltas_from_json(buf)


# ── Wire format helper (magic + version + payload) ──────────────────────

_ARROW_MESH_MAGIC = b"AM01"  # Arrow Mesh v1


def encode_wire(
    codec: ArrowMeshCodec, payload: bytes, msg_type: str = "digest"
) -> bytes:
    """Prefix payload with magic header for stream framing.

    Format: [magic 4 bytes][type 1 byte][len 4 bytes BE][payload ...]
    """
    type_byte = {"digest": 0x01, "deltas": 0x02}.get(msg_type, 0x00)
    length = len(payload)
    return _ARROW_MESH_MAGIC + struct.pack(">B I", type_byte, length) + payload


def decode_wire(buf: bytes) -> Tuple[str, bytes]:
    """Decode wire-framed payload. Returns (msg_type, payload_bytes)."""
    if not buf.startswith(_ARROW_MESH_MAGIC):
        # Assume raw JSON/Arrow IPC (backwards compatible)
        return ("raw", buf)
    type_byte = buf[4]
    length = struct.unpack(">I", buf[5:9])[0]
    payload = buf[9 : 9 + length]
    type_map = {0x01: "digest", 0x02: "deltas", 0x00: "unknown"}
    return (type_map.get(type_byte, "unknown"), payload)


# ── Integration shim for MeshVectorGossip ───────────────────────────────


class ArrowMeshGossip:
    """Drop-in wrapper that adds Arrow serialization to MeshVectorGossip.

    Usage
    -----
        gossip = MeshVectorGossip(node_id="Oracle1", local_table=table)
        arrow_gossip = ArrowMeshGossip(gossip)

        # serialize a digest for the wire
        ipc_bytes = arrow_gossip.encode_digest_for_peer()

        # on receiving peer
        arrow_gossip.apply_digest_from_peer(ipc_bytes)
    """

    def __init__(self, mesh_gossip: Any):
        self.mesh = mesh_gossip
        self.codec = ArrowMeshCodec()

    def encode_digest_for_peer(self) -> bytes:
        """Return wire-ready digest bytes."""
        digest = self.mesh.local_digest()
        payload = self.codec.serialize_digest(digest)
        return encode_wire(self.codec, payload, "digest")

    def apply_digest_from_peer(self, wire_bytes: bytes) -> Dict[str, Any]:
        """Decode and return digest dict."""
        _msg_type, payload = decode_wire(wire_bytes)
        return self.codec.deserialize_digest(payload)

    def encode_deltas_for_peer(self, batch: Any) -> bytes:
        """Return wire-ready DeltaBatch bytes."""
        payload = self.codec.serialize_deltas(batch)
        return encode_wire(self.codec, payload, "deltas")

    def apply_deltas_from_peer(self, wire_bytes: bytes) -> Dict[str, Any]:
        """Decode and return DeltaBatch-compatible dict."""
        _msg_type, payload = decode_wire(wire_bytes)
        return self.codec.deserialize_deltas(payload)
