"""Data transformer with JSON, CSV, and XML format support.

Converts between common data formats with schema validation hooks.
Used for fleet data ingestion, export, and interoperability.

Usage:
    tx = DataTransformer()
    csv = tx.to_csv([{"a": 1, "b": 2}])
    json = tx.to_json({"x": 1})
    parsed = tx.from_json('{"x": 1}')
"""

from __future__ import annotations

import csv
import io
import json
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional


class DataTransformer:
    """
    Lightweight data format transformer.

    No external dependencies — uses stdlib only.
    """

    def __init__(self, json_encoder: Optional[type] = None):
        self._json_encoder = json_encoder

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def to_json(self, data: Any, indent: Optional[int] = None) -> str:
        """Serialize to JSON string."""
        kwargs: Dict[str, Any] = {}
        if self._json_encoder:
            kwargs["cls"] = self._json_encoder
        if indent:
            kwargs["indent"] = indent
        return json.dumps(data, **kwargs)

    def from_json(self, text: str) -> Any:
        """Parse JSON string."""
        return json.loads(text)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def to_csv(self, rows: List[Dict[str, Any]]) -> str:
        """Serialize list of dicts to CSV string."""
        if not rows:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()

    def from_csv(self, text: str) -> List[Dict[str, str]]:
        """Parse CSV string to list of dicts."""
        input_file = io.StringIO(text)
        reader = csv.DictReader(input_file)
        return list(reader)

    # ------------------------------------------------------------------
    # XML
    # ------------------------------------------------------------------

    def to_xml(
        self,
        data: Dict[str, Any],
        root_tag: str = "root",
        item_tag: str = "item",
    ) -> str:
        """Serialize dict to simple XML string."""
        root = ET.Element(root_tag)
        self._dict_to_xml(root, data, item_tag)
        return ET.tostring(root, encoding="unicode")

    def _dict_to_xml(self, parent: ET.Element, data: Any, item_tag: str) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                child = ET.SubElement(parent, str(key))
                self._dict_to_xml(child, value, item_tag)
        elif isinstance(data, list):
            for item in data:
                child = ET.SubElement(parent, item_tag)
                self._dict_to_xml(child, item, item_tag)
        else:
            parent.text = str(data)

    def from_xml(self, text: str) -> Dict[str, Any]:
        """Parse simple XML string to dict."""
        root = ET.fromstring(text)
        return {root.tag: self._xml_to_dict(root)}

    def _xml_to_dict(self, element: ET.Element) -> Any:
        children = list(element)
        if not children:
            return element.text
        result: Dict[str, Any] = {}
        for child in children:
            tag = child.tag
            child_data = self._xml_to_dict(child)
            if tag in result:
                if not isinstance(result[tag], list):
                    result[tag] = [result[tag]]
                result[tag].append(child_data)
            else:
                result[tag] = child_data
        return result

    # ------------------------------------------------------------------
    # Validation hook
    # ------------------------------------------------------------------

    def validate(self, data: Any, schema: Callable[[Any], bool]) -> bool:
        """Run a validation function against data."""
        try:
            return schema(data)
        except Exception:
            return False

    def __repr__(self) -> str:
        return "<DataTransformer>"
