"""ring_buffer.py — Python bindings for lock-free SPSC ring buffer.

Wraps the C implementation in nerve/ring_buffer.so.
"""

from __future__ import annotations

__all__ = ["RingBuffer"]

import ctypes
import os
import struct
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")

# Load the shared library
_so_path = Path(__file__).parent / "ring_buffer.so"
if not _so_path.exists():
    # Fallback: look relative to package root
    _so_path = Path(__file__).parent / "ring_buffer.so"
_rb = ctypes.CDLL(str(_so_path))

# C API signatures
_rb.rb_create.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
_rb.rb_create.restype = ctypes.c_void_p

_rb.rb_destroy.argtypes = [ctypes.c_void_p]
_rb.rb_destroy.restype = None

_rb.rb_capacity.argtypes = [ctypes.c_void_p]
_rb.rb_capacity.restype = ctypes.c_uint64

_rb.rb_size.argtypes = [ctypes.c_void_p]
_rb.rb_size.restype = ctypes.c_uint64

_rb.rb_free.argtypes = [ctypes.c_void_p]
_rb.rb_free.restype = ctypes.c_uint64

_rb.rb_push.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_rb.rb_push.restype = ctypes.c_int

_rb.rb_pop.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_rb.rb_pop.restype = ctypes.c_int

_rb.rb_peek.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_rb.rb_peek.restype = ctypes.c_int

_rb.rb_push_bulk.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
_rb.rb_push_bulk.restype = ctypes.c_uint64

_rb.rb_pop_bulk.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
_rb.rb_pop_bulk.restype = ctypes.c_uint64


class RingBuffer:
    """Lock-free single-producer single-consumer ring buffer.

    Backed by C ring_buffer.so with atomic head/tail and memory fences.
    Safe for one writer thread and one reader thread without locks.
    """

    def __init__(self, capacity: int, elem_size: int = 8) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if elem_size <= 0:
            raise ValueError("elem_size must be > 0")
        self._elem_size = elem_size
        self._handle = _rb.rb_create(capacity, elem_size)
        if not self._handle:
            raise MemoryError("Failed to create ring buffer")

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            _rb.rb_destroy(self._handle)
            self._handle = None

    # ── properties ──────────────────────────────────────

    @property
    def capacity(self) -> int:
        return int(_rb.rb_capacity(self._handle))

    @property
    def size(self) -> int:
        return int(_rb.rb_size(self._handle))

    @property
    def free(self) -> int:
        return int(_rb.rb_free(self._handle))

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    @property
    def is_full(self) -> bool:
        return self.free == 0

    # ── single element ──────────────────────────────────

    def push_bytes(self, data: bytes) -> bool:
        if len(data) != self._elem_size:
            raise ValueError(f"Expected {self._elem_size} bytes, got {len(data)}")
        buf = ctypes.create_string_buffer(data)
        return bool(_rb.rb_push(self._handle, buf))

    def pop_bytes(self) -> bytes | None:
        buf = ctypes.create_string_buffer(self._elem_size)
        if _rb.rb_pop(self._handle, buf):
            return bytes(buf)
        return None

    def peek_bytes(self) -> bytes | None:
        buf = ctypes.create_string_buffer(self._elem_size)
        if _rb.rb_peek(self._handle, buf):
            return bytes(buf)
        return None

    # ── typed helpers (uint64) ──────────────────────────

    def push_u64(self, value: int) -> bool:
        return self.push_bytes(struct.pack("<Q", value))

    def pop_u64(self) -> int | None:
        data = self.pop_bytes()
        if data is None:
            return None
        return struct.unpack("<Q", data)[0]

    def push_i64(self, value: int) -> bool:
        return self.push_bytes(struct.pack("<q", value))

    def pop_i64(self) -> int | None:
        data = self.pop_bytes()
        if data is None:
            return None
        return struct.unpack("<q", data)[0]

    def push_f64(self, value: float) -> bool:
        return self.push_bytes(struct.pack("<d", value))

    def pop_f64(self) -> float | None:
        data = self.pop_bytes()
        if data is None:
            return None
        return struct.unpack("<d", data)[0]

    # ── bulk operations ───────────────────────────────────

    def push_bulk_u64(self, values: list[int]) -> int:
        """Push list of uint64. Returns number actually pushed."""
        n = len(values)
        packed = struct.pack(f"<{n}Q", *values)
        buf = ctypes.create_string_buffer(packed)
        return int(_rb.rb_push_bulk(self._handle, buf, n))

    def pop_bulk_u64(self, max_n: int) -> list[int]:
        """Pop up to max_n uint64 values."""
        buf = ctypes.create_string_buffer(max_n * 8)
        got = int(_rb.rb_pop_bulk(self._handle, buf, max_n))
        if got == 0:
            return []
        return list(struct.unpack(f"<{got}Q", bytes(buf)[: got * 8]))

    # ── diagnostics ───────────────────────────────────────

    def __repr__(self) -> str:
        return f"RingBuffer(cap={self.capacity}, size={self.size}, elem={self._elem_size}b)"
