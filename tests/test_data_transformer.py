"""Tests for data_transformer.py — Format conversion.

Run: python3 -m pytest tests/test_data_transformer.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.data_transformer import DataTransformer


class TestDataTransformer:
    def test_create(self):
        tx = DataTransformer()
        assert "DataTransformer" in repr(tx)

    def test_json_roundtrip(self):
        tx = DataTransformer()
        data = {"x": 1, "y": [1, 2, 3]}
        json_str = tx.to_json(data)
        parsed = tx.from_json(json_str)
        assert parsed == data

    def test_csv_roundtrip(self):
        tx = DataTransformer()
        rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
        csv_str = tx.to_csv(rows)
        parsed = tx.from_csv(csv_str)
        assert len(parsed) == 2
        assert parsed[0]["a"] == "1"

    def test_empty_csv(self):
        tx = DataTransformer()
        assert tx.to_csv([]) == ""

    def test_xml_roundtrip(self):
        tx = DataTransformer()
        data = {"person": {"name": "Alice", "age": "30"}}
        xml_str = tx.to_xml(data, root_tag="root")
        parsed = tx.from_xml(xml_str)
        assert parsed["root"]["person"]["name"] == "Alice"

    def test_xml_list(self):
        tx = DataTransformer()
        data = {"items": [{"x": "1"}, {"x": "2"}]}
        xml_str = tx.to_xml(data, root_tag="root", item_tag="item")
        parsed = tx.from_xml(xml_str)
        assert isinstance(parsed["root"]["items"]["item"], list)

    def test_validate(self):
        tx = DataTransformer()
        assert tx.validate({"x": 1}, lambda d: "x" in d) is True
        assert tx.validate({"x": 1}, lambda d: "y" in d) is False
