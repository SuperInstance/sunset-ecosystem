"""Tests for encoding_helper.py — Base64, hex, URL encoding.

Run: python3 -m pytest tests/test_encoding_helper.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.encoding_helper import EncodingHelper


class TestEncodingHelper:
    def test_create(self):
        enc = EncodingHelper()
        assert enc.stats()["formats"] == 3

    def test_base64_roundtrip(self):
        enc = EncodingHelper()
        data = "hello world"
        encoded = enc.base64_encode(data)
        decoded = enc.base64_decode(encoded)
        assert decoded == data

    def test_base64_url_roundtrip(self):
        enc = EncodingHelper()
        data = "hello world"
        encoded = enc.base64_url_encode(data)
        decoded = enc.base64_url_decode(encoded)
        assert decoded == data

    def test_hex_roundtrip(self):
        enc = EncodingHelper()
        data = "hello world"
        encoded = enc.hex_encode(data)
        decoded = enc.hex_decode(encoded)
        assert decoded == data

    def test_url_roundtrip(self):
        enc = EncodingHelper()
        data = "hello world!"
        encoded = enc.url_encode(data)
        decoded = enc.url_decode(encoded)
        assert decoded == data

    def test_url_encode_params(self):
        enc = EncodingHelper()
        params = {"a": "1", "b": "hello world"}
        encoded = enc.url_encode_params(params)
        assert "a=1" in encoded
        assert "b=hello+world" in encoded

    def test_repr(self):
        enc = EncodingHelper()
        assert "EncodingHelper" in repr(enc)
