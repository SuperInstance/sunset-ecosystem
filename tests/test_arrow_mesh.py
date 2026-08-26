"""Tests for Arrow mesh serialization.

Covers codec round-trips, wire framing, JSON fallback, and integration
shim. Arrow-specific tests are skipped if pyarrow is not installed.
"""

import json
import struct

import numpy as np
import pytest

from swarm.arrow_mesh import (
    ArrowMeshCodec,
    ArrowMeshGossip,
    encode_wire,
    decode_wire,
    _ARROW_MESH_MAGIC,
)

# Conditionally import pyarrow for type hints in tests
try:
    import pyarrow as pa

    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False


# ---------------------------------------------------------------------------
# Helpers — mock digest / delta
# ---------------------------------------------------------------------------


class MockDigest:
    def __init__(self):
        self.node_id = "Oracle1"
        self.agent_count = 42
        self.version_vector = {1: 3, 7: 2, 9: 5}
        self.bitfield = "deadbeef"
        self.max_wall_time = 1234.5


class MockDeltaBatch:
    def __init__(self, deltas=None, sequence=0):
        self.node_id = "Oracle1"
        self.sequence = sequence
        self.deltas = deltas or []


# ---------------------------------------------------------------------------
# JSON fallback (always available)
# ---------------------------------------------------------------------------


class TestJsonFallback:
    def test_digest_roundtrip(self):
        codec = ArrowMeshCodec(use_arrow=False)
        digest = MockDigest()
        buf = codec.digest_to_json(digest)
        d = codec.digest_from_json(buf)
        assert d["node_id"] == "Oracle1"
        assert d["agent_count"] == 42
        assert d["version_vector"]["7"] == 2
        assert d["bitfield"] == "deadbeef"
        assert d["max_wall_time"] == 1234.5

    def test_deltas_roundtrip(self):
        codec = ArrowMeshCodec(use_arrow=False)
        batch = MockDeltaBatch(
            deltas=[
                {
                    "agent_id": 7,
                    "vector": np.array([0.1, -0.2, 0.3], dtype=np.float32),
                    "score": 0.85,
                    "timestamp": 1000.0,
                    "metadata": {"breed": "scout"},
                }
            ],
            sequence=3,
        )
        buf = codec.deltas_to_json(batch)
        d = codec.deltas_from_json(buf)
        assert d["node_id"] == "Oracle1"
        assert d["sequence"] == 3
        assert len(d["deltas"]) == 1
        delta = d["deltas"][0]
        assert delta["agent_id"] == 7
        assert np.allclose(delta["vector"], [0.1, -0.2, 0.3])
        assert delta["score"] == 0.85
        assert delta["metadata"]["breed"] == "scout"

    def test_auto_detect_digest(self):
        codec = ArrowMeshCodec(use_arrow=False)
        digest = MockDigest()
        buf = codec.serialize_digest(digest)
        d = codec.deserialize_digest(buf)
        assert d["node_id"] == "Oracle1"

    def test_auto_detect_deltas(self):
        codec = ArrowMeshCodec(use_arrow=False)
        batch = MockDeltaBatch(deltas=[{"agent_id": 1, "vector": [0.5], "score": 0.9}])
        buf = codec.serialize_deltas(batch)
        d = codec.deserialize_deltas(buf)
        assert d["deltas"][0]["agent_id"] == 1


# ---------------------------------------------------------------------------
# Wire framing
# ---------------------------------------------------------------------------


class TestWireFraming:
    def test_encode_decode_digest(self):
        codec = ArrowMeshCodec(use_arrow=False)
        payload = b"hello digest"
        wire = encode_wire(codec, payload, "digest")
        assert wire[:4] == _ARROW_MESH_MAGIC
        msg_type, decoded = decode_wire(wire)
        assert msg_type == "digest"
        assert decoded == payload

    def test_encode_decode_deltas(self):
        codec = ArrowMeshCodec(use_arrow=False)
        payload = b"hello deltas"
        wire = encode_wire(codec, payload, "deltas")
        msg_type, decoded = decode_wire(wire)
        assert msg_type == "deltas"
        assert decoded == payload

    def test_raw_fallback(self):
        raw = b"some random payload"
        msg_type, decoded = decode_wire(raw)
        assert msg_type == "raw"
        assert decoded == raw

    def test_wire_header_format(self):
        codec = ArrowMeshCodec(use_arrow=False)
        payload = b"x" * 100
        wire = encode_wire(codec, payload, "digest")
        assert len(wire) == 4 + 1 + 4 + 100  # magic + type + len + payload
        assert struct.unpack(">I", wire[5:9])[0] == 100


# ---------------------------------------------------------------------------
# Arrow round-trips (skipped if pyarrow missing)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestArrowCodec:
    def test_digest_arrow_roundtrip(self):
        codec = ArrowMeshCodec(use_arrow=True)
        digest = MockDigest()
        batch = codec.digest_to_arrow(digest)
        d = codec.digest_from_arrow(batch)
        assert d["node_id"] == "Oracle1"
        assert d["agent_count"] == 42
        assert d["version_vector"][1] == 3
        assert d["bitfield"] == "deadbeef"

    def test_deltas_arrow_roundtrip(self):
        codec = ArrowMeshCodec(use_arrow=True)
        batch = MockDeltaBatch(
            deltas=[
                {
                    "agent_id": 7,
                    "vector": np.array([0.1, -0.2, 0.3], dtype=np.float32),
                    "score": 0.85,
                    "timestamp": 1000.0,
                    "metadata": {"key": "val"},
                }
            ],
            sequence=5,
        )
        rb = codec.deltas_to_arrow(batch)
        d = codec.deltas_from_arrow(rb)
        assert d["node_id"] == "Oracle1"
        assert d["sequence"] == 5
        assert len(d["deltas"]) == 1
        delta = d["deltas"][0]
        assert delta["agent_id"] == 7
        assert np.allclose(delta["vector"], [0.1, -0.2, 0.3])
        assert delta["score"] == 0.85

    def test_ipc_roundtrip(self):
        codec = ArrowMeshCodec(use_arrow=True)
        digest = MockDigest()
        batch = codec.digest_to_arrow(digest)
        ipc_bytes = codec.serialize_batch(batch)
        batch2 = codec.deserialize_batch(ipc_bytes)
        d = codec.digest_from_arrow(batch2)
        assert d["node_id"] == "Oracle1"

    def test_schema_evolution_ignores_unknown(self):
        codec = ArrowMeshCodec(use_arrow=True)
        digest = MockDigest()
        batch = codec.digest_to_arrow(digest)
        # Schema should have expected fields
        names = {f.name for f in batch.schema}
        assert "node_id" in names
        assert "bitfield" in names
        assert "max_wall_time" in names


# ---------------------------------------------------------------------------
# ArrowMeshGossip shim
# ---------------------------------------------------------------------------


class MockMeshGossip:
    def __init__(self):
        self.node_id = "Oracle1"
        self.last_digest = MockDigest()

    def local_digest(self):
        return self.last_digest


class TestArrowMeshGossip:
    def test_encode_digest_wire(self):
        mesh = MockMeshGossip()
        arrow_mesh = ArrowMeshGossip(mesh)
        wire = arrow_mesh.encode_digest_for_peer()
        msg_type, payload = decode_wire(wire)
        assert msg_type == "digest"
        # Payload should be decodable
        d = arrow_mesh.codec.deserialize_digest(payload)
        assert d["node_id"] == "Oracle1"

    def test_apply_digest_from_peer(self):
        mesh = MockMeshGossip()
        arrow_mesh = ArrowMeshGossip(mesh)
        wire = arrow_mesh.encode_digest_for_peer()
        d = arrow_mesh.apply_digest_from_peer(wire)
        assert d["node_id"] == "Oracle1"
        assert d["agent_count"] == 42

    def test_encode_deltas_wire(self):
        mesh = MockMeshGossip()
        arrow_mesh = ArrowMeshGossip(mesh)
        batch = MockDeltaBatch(
            deltas=[{"agent_id": 3, "vector": [0.1, 0.2], "score": 0.9}],
            sequence=1,
        )
        wire = arrow_mesh.encode_deltas_for_peer(batch)
        msg_type, payload = decode_wire(wire)
        assert msg_type == "deltas"
        d = arrow_mesh.codec.deserialize_deltas(payload)
        assert d["deltas"][0]["agent_id"] == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_delta_batch_json(self):
        codec = ArrowMeshCodec(use_arrow=False)
        batch = MockDeltaBatch(deltas=[], sequence=0)
        buf = codec.deltas_to_json(batch)
        d = codec.deltas_from_json(buf)
        assert d["deltas"] == []

    def test_missing_metadata_json(self):
        codec = ArrowMeshCodec(use_arrow=False)
        batch = MockDeltaBatch(
            deltas=[{"agent_id": 1, "vector": [0.5], "score": None}],
        )
        buf = codec.deltas_to_json(batch)
        d = codec.deltas_from_json(buf)
        assert d["deltas"][0]["score"] is None

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
    def test_empty_delta_batch_arrow(self):
        codec = ArrowMeshCodec(use_arrow=True)
        batch = MockDeltaBatch(deltas=[], sequence=0)
        rb = codec.deltas_to_arrow(batch)
        assert rb.num_rows == 0

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
    def test_multi_row_deltas_arrow(self):
        codec = ArrowMeshCodec(use_arrow=True)
        batch = MockDeltaBatch(
            deltas=[
                {"agent_id": 1, "vector": [0.1, 0.2], "score": 0.9},
                {"agent_id": 2, "vector": [0.3, 0.4], "score": 0.8},
            ],
            sequence=7,
        )
        rb = codec.deltas_to_arrow(batch)
        assert rb.num_rows == 2
        d = codec.deltas_from_arrow(rb)
        assert len(d["deltas"]) == 2
        assert d["sequence"] == 7

    def test_json_encoding_is_compact(self):
        codec = ArrowMeshCodec(use_arrow=False)
        digest = MockDigest()
        buf = codec.digest_to_json(digest)
        # Should be valid JSON, no extra whitespace
        text = buf.decode("utf-8")
        assert " " not in text or text.count(" ") < 5  # minimal whitespace

    def test_numpy_vector_serialization(self):
        codec = ArrowMeshCodec(use_arrow=False)
        vec = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        batch = MockDeltaBatch(
            deltas=[{"agent_id": 1, "vector": vec, "score": 0.9}],
        )
        buf = codec.deltas_to_json(batch)
        d = codec.deltas_from_json(buf)
        assert isinstance(d["deltas"][0]["vector"], np.ndarray)
        assert d["deltas"][0]["vector"].dtype == np.float32
