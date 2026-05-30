"""Tests for serialization_helper.py — Binary/JSON serialization.

Run: python3 -m pytest tests/test_serialization_helper.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.serialization_helper import SerializationHelper


class TestSerializationHelper:
    def test_create(self):
        ser = SerializationHelper()
        assert "json" in repr(ser)

    def test_json_roundtrip(self):
        ser = SerializationHelper("json")
        data = {"x": 1, "y": [1, 2, 3]}
        packed = ser.pack(data)
        unpacked = ser.unpack(packed)
        assert unpacked == data

    def test_pickle_roundtrip(self):
        with pytest.raises(ValueError):
            SerializationHelper("pickle")

    def test_binary_roundtrip(self):
        ser = SerializationHelper("binary")
        data = {"x": 1, "y": "hello", "z": [1, 2, 3]}
        packed = ser.pack(data)
        unpacked = ser.unpack(packed)
        assert unpacked == data

    def test_binary_types(self):
        ser = SerializationHelper("binary")
        for value in [42, 3.14, "hello", True, None, [1, 2], {"a": "b"}]:
            packed = ser.pack(value)
            unpacked = ser.unpack(packed)
            assert unpacked == value

    def test_str_methods(self):
        ser = SerializationHelper()
        data = {"x": 1}
        text = ser.pack_to_str(data)
        assert isinstance(text, str)
        assert ser.unpack_from_str(text) == data

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            SerializationHelper("invalid")

    def test_repr(self):
        ser = SerializationHelper("binary")
        assert "SerializationHelper" in repr(ser)
