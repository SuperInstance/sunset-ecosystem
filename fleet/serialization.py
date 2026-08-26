"""Schema-based serialization with validation and compression options.

Provides typed serialization (JSON, msgpack-style) with optional compression
and schema validation. Used for fleet node communication and state persistence.

Usage:
    ser = SerializationRegistry()
    ser.register("breed_result", BreedResultSchema)
    blob = ser.serialize("breed_result", {"score": 0.95})
    obj = ser.deserialize("breed_result", blob)
"""

from __future__ import annotations

import json
import logging
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SerializationError(Exception):
    pass


class SchemaValidationError(SerializationError):
    pass


@dataclass
class Schema:
    """Lightweight schema descriptor."""

    name: str
    required: list = None
    types: dict = None
    validator: Optional[Callable[[Any], None]] = None

    def validate(self, obj: Any) -> None:
        if self.validator is not None:
            self.validator(obj)
        if self.required is not None:
            for key in self.required:
                if key not in obj:
                    raise SchemaValidationError(f"Missing required field: {key}")
        if self.types is not None:
            for key, expected in self.types.items():
                if key in obj and not isinstance(obj[key], expected):
                    raise SchemaValidationError(
                        f"Field {key} must be {expected.__name__}, got {type(obj[key]).__name__}"
                    )


class SerializationRegistry:
    """
    Typed serialization with optional compression.

    :param default_format: "json" or "bytes" (raw pickle-like not supported for safety).
    :param default_compression: "zlib" or None.
    """

    def __init__(
        self,
        default_format: str = "json",
        default_compression: Optional[str] = None,
    ):
        self._default_format = default_format
        self._default_compression = default_compression
        self._schemas: Dict[str, Schema] = {}
        self._stats: Dict[str, int] = {
            "serialized": 0,
            "deserialized": 0,
            "bytes_in": 0,
            "bytes_out": 0,
        }

    # ------------------------------------------------------------------
    # Schema registry
    # ------------------------------------------------------------------

    def register(self, name: str, schema: Schema) -> None:
        self._schemas[name] = schema

    def has_schema(self, name: str) -> bool:
        return name in self._schemas

    def get_schema(self, name: str) -> Optional[Schema]:
        return self._schemas.get(name)

    # ------------------------------------------------------------------
    # Serialize / deserialize
    # ------------------------------------------------------------------

    def serialize(
        self,
        schema_name: str,
        obj: Any,
        format: Optional[str] = None,
        compression: Optional[str] = None,
        validate: bool = True,
    ) -> bytes:
        fmt = format or self._default_format
        comp = compression or self._default_compression
        if validate:
            schema = self._schemas.get(schema_name)
            if schema is not None:
                schema.validate(obj)

        if fmt == "json":
            payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        elif fmt == "bytes":
            if not isinstance(obj, bytes):
                raise SerializationError("bytes format requires a bytes object")
            payload = obj
        else:
            raise SerializationError(f"Unknown format: {fmt}")

        if comp == "zlib":
            payload = zlib.compress(payload)

        self._stats["serialized"] += 1
        self._stats["bytes_in"] += len(payload)
        return payload

    def deserialize(
        self,
        schema_name: str,
        blob: bytes,
        format: Optional[str] = None,
        compression: Optional[str] = None,
        validate: bool = True,
    ) -> Any:
        fmt = format or self._default_format
        comp = compression or self._default_compression

        if comp == "zlib":
            try:
                blob = zlib.decompress(blob)
            except zlib.error as e:
                raise SerializationError(f"Decompression failed: {e}")

        if fmt == "json":
            obj = json.loads(blob.decode("utf-8"))
        elif fmt == "bytes":
            obj = blob
        else:
            raise SerializationError(f"Unknown format: {fmt}")

        if validate:
            schema = self._schemas.get(schema_name)
            if schema is not None:
                schema.validate(obj)

        self._stats["deserialized"] += 1
        self._stats["bytes_out"] += len(blob)
        return obj

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def __repr__(self) -> str:
        return f"<SerializationRegistry schemas={len(self._schemas)}>"
