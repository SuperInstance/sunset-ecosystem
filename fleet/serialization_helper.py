"""Binary and JSON serialization helpers.

Provides efficient serialization for common fleet data types.
Supports JSON, msgpack-style binary, and custom encoders. Used for
inter-node communication and state persistence.

Usage:
    ser = SerializationHelper()
    packed = ser.pack({"x": 1})
    data = ser.unpack(packed)
"""
from __future__ import annotations

import json
import struct
from typing import Any, Dict, List, Optional, Union


class SerializationHelper:
    """
    Serialization helper with safe backends.

    Supports JSON and a custom msgpack-style binary format.
    Pickle is intentionally NOT supported for security reasons.
    """

    def __init__(self, format: str = "json"):
        if format not in ("json", "binary"):
            raise ValueError("format must be 'json' or 'binary'. Pickle is not supported for security.")
        self._format = format

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def pack(self, data: Any, format: Optional[str] = None) -> bytes:
        """Serialize data to bytes."""
        fmt = format or self._format
        if fmt == "json":
            return json.dumps(data).encode("utf-8")
        if fmt == "binary":
            return self._binary_pack(data)
        raise ValueError(f"Unknown format: {fmt}")

    def unpack(self, data: bytes, format: Optional[str] = None) -> Any:
        """Deserialize bytes to data."""
        fmt = format or self._format
        if fmt == "json":
            return json.loads(data.decode("utf-8"))
        if fmt == "binary":
            value, _offset = self._binary_unpack(data)
            return value
        raise ValueError(f"Unknown format: {fmt}")

    # ------------------------------------------------------------------
    # Binary format (msgpack-like simple)
    # ------------------------------------------------------------------

    def _binary_pack(self, data: Any) -> bytes:
        """Simple binary pack for common types."""
        if isinstance(data, dict):
            items = b"".join(
                self._binary_pack(k) + self._binary_pack(v)
                for k, v in data.items()
            )
            return b"\x01" + struct.pack(">I", len(data)) + items
        if isinstance(data, list):
            items = b"".join(self._binary_pack(v) for v in data)
            return b"\x02" + struct.pack(">I", len(data)) + items
        if isinstance(data, str):
            encoded = data.encode("utf-8")
            return b"\x03" + struct.pack(">I", len(encoded)) + encoded
        if isinstance(data, int):
            return b"\x04" + struct.pack(">q", data)
        if isinstance(data, float):
            return b"\x05" + struct.pack(">d", data)
        if isinstance(data, bool):
            return b"\x06" + (b"\x01" if data else b"\x00")
        if data is None:
            return b"\x07"
        raise ValueError(f"Unsupported type: {type(data)}")

    def _binary_unpack(self, data: bytes, offset: int = 0) -> tuple:
        """Unpack binary data. Returns (value, new_offset)."""
        if offset >= len(data):
            raise ValueError("Unexpected end of data")
        tag = data[offset]
        offset += 1
        if tag == 0x01:  # dict
            count = struct.unpack(">I", data[offset:offset + 4])[0]
            offset += 4
            result = {}
            for _ in range(count):
                k, offset = self._binary_unpack(data, offset)
                v, offset = self._binary_unpack(data, offset)
                result[k] = v
            return result, offset
        if tag == 0x02:  # list
            count = struct.unpack(">I", data[offset:offset + 4])[0]
            offset += 4
            result = []
            for _ in range(count):
                v, offset = self._binary_unpack(data, offset)
                result.append(v)
            return result, offset
        if tag == 0x03:  # str
            length = struct.unpack(">I", data[offset:offset + 4])[0]
            offset += 4
            return data[offset:offset + length].decode("utf-8"), offset + length
        if tag == 0x04:  # int
            return struct.unpack(">q", data[offset:offset + 8])[0], offset + 8
        if tag == 0x05:  # float
            return struct.unpack(">d", data[offset:offset + 8])[0], offset + 8
        if tag == 0x06:  # bool
            return data[offset] == 1, offset + 1
        if tag == 0x07:  # None
            return None, offset
        raise ValueError(f"Unknown tag: {tag}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def pack_to_str(self, data: Any) -> str:
        """Pack to JSON string."""
        return json.dumps(data)

    def unpack_from_str(self, text: str) -> Any:
        """Unpack from JSON string."""
        return json.loads(text)

    def __repr__(self) -> str:
        return f"<SerializationHelper format={self._format}>"
