"""Tests for serialization.py — Schema-based serialization.

Run: python3 -m pytest tests/test_serialization.py -v --tb=short
"""

from __future__ import annotations

import json
import zlib

import pytest

from fleet.serialization import (
    SerializationRegistry,
    Schema,
    SchemaValidationError,
    SerializationError,
)


class TestSerializationRegistry:
    def test_create(self):
        reg = SerializationRegistry()
        assert reg.stats()["serialized"] == 0

    def test_register_and_has(self):
        reg = SerializationRegistry()
        reg.register("breed", Schema("breed", required=["score"]))
        assert reg.has_schema("breed") is True
        assert reg.has_schema("missing") is False

    def test_json_roundtrip(self):
        reg = SerializationRegistry()
        obj = {"score": 0.95, "room_id": 42}
        blob = reg.serialize("any", obj, validate=False)
        result = reg.deserialize("any", blob, validate=False)
        assert result == obj

    def test_schema_validation_pass(self):
        reg = SerializationRegistry()
        reg.register(
            "breed", Schema("breed", required=["score"], types={"score": float})
        )
        obj = {"score": 0.95}
        blob = reg.serialize("breed", obj)
        result = reg.deserialize("breed", blob)
        assert result["score"] == 0.95

    def test_schema_validation_missing_field(self):
        reg = SerializationRegistry()
        reg.register("breed", Schema("breed", required=["score"]))
        with pytest.raises(SchemaValidationError):
            reg.serialize("breed", {"room_id": 42})

    def test_schema_validation_wrong_type(self):
        reg = SerializationRegistry()
        reg.register("breed", Schema("breed", types={"score": float}))
        with pytest.raises(SchemaValidationError):
            reg.serialize("breed", {"score": "not-a-float"})

    def test_zlib_compression(self):
        reg = SerializationRegistry(default_compression="zlib")
        obj = {"data": "x" * 1000}
        blob = reg.serialize("any", obj, validate=False)
        assert len(blob) < len(json.dumps(obj).encode())  # Compressed
        result = reg.deserialize("any", blob, validate=False)
        assert result == obj

    def test_explicit_compression(self):
        reg = SerializationRegistry()
        obj = {"a": 1}
        blob = reg.serialize("any", obj, compression="zlib", validate=False)
        assert zlib.decompress(blob) == b'{"a":1}'

    def test_decompress_failure(self):
        reg = SerializationRegistry()
        with pytest.raises(SerializationError):
            reg.deserialize("any", b"not-zlib", compression="zlib", validate=False)

    def test_bytes_format(self):
        reg = SerializationRegistry()
        blob = b"raw-payload"
        result = reg.serialize("any", blob, format="bytes", validate=False)
        assert result == blob
        assert reg.deserialize("any", result, format="bytes", validate=False) == blob

    def test_bytes_format_error(self):
        reg = SerializationRegistry()
        with pytest.raises(SerializationError):
            reg.serialize("any", {"a": 1}, format="bytes", validate=False)

    def test_unknown_format(self):
        reg = SerializationRegistry()
        with pytest.raises(SerializationError):
            reg.serialize("any", {"a": 1}, format="xml", validate=False)

    def test_stats(self):
        reg = SerializationRegistry()
        reg.serialize("any", {"a": 1}, validate=False)
        reg.deserialize("any", b'{"a":1}', validate=False)
        stats = reg.stats()
        assert stats["serialized"] == 1
        assert stats["deserialized"] == 1

    def test_repr(self):
        reg = SerializationRegistry()
        assert "SerializationRegistry" in repr(reg)
