"""Unified memory pool for agent vectors across CPU/GPU/NPU.

Eliminates cudaMemcpy overhead by mapping CPU memory directly into GPU
address space via zero-copy (cudaHostAllocMapped) or unified memory
(cudaMallocManaged).  On Jetson the CPU/GPU share physical memory so
no migrate is needed.

Fallback: plain numpy arrays with mmap for cross-process sharing.
"""

from __future__ import annotations

__all__ = ["MemoryHandle", "UnifiedMemoryPool"]

import logging
import mmap
import os
import tempfile
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

log = logging.getLogger(__name__)

# ── CUDA / CuPy availability ──────────────────────────────
_CUDA_AVAILABLE = False
_CUPY_MODULE = None

try:
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() > 0:
        _CUDA_AVAILABLE = True
        _CUPY_MODULE = cp
        log.info("CuPy CUDA available — %d device(s)", cp.cuda.runtime.getDeviceCount())
    else:
        log.debug("CuPy installed but no CUDA devices found")
except Exception as exc:
    log.debug("CuPy not available (%s) — will use NumPy fallback", type(exc).__name__)

# ── MemoryHandle ──────────────────────────────────────────


@dataclass(slots=True)
class MemoryHandle:
    """Opaque handle returned by UnifiedMemoryPool.allocate().

    Attributes:
        handle_id:   UUID string uniquely identifying this allocation.
        size_bytes:  Number of bytes requested.
        device:      Device where the memory currently lives
                     ("cpu", "gpu", "npu", "jetson", "managed").
        ptr:         Integer address accessible on *current* device.
        unified:     True if this handle uses unified (same pointer everywhere).
        _valid:      Internal flag — becomes False after free().
        _arrays:     Mapping device -> underlying array object (for cleanup).
    """

    handle_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    size_bytes: int = 0
    device: str = "cpu"
    ptr: int = 0
    unified: bool = False
    _valid: bool = True
    _arrays: Dict[str, Any] = field(default_factory=dict, repr=False)

    def invalidate(self) -> None:
        self._valid = False
        self._arrays.clear()
        self.ptr = 0

    def __repr__(self) -> str:
        status = "valid" if self._valid else "freed"
        return (
            f"MemoryHandle({self.handle_id}, {self.size_bytes}B, "
            f"device={self.device}, ptr=0x{self.ptr:x}, unified={self.unified}, {status})"
        )


# ── UnifiedMemoryPool ─────────────────────────────────────


class UnifiedMemoryPool:
    """Zero-copy memory pool for agent vectors across CPU/GPU/NPU.

    Three operational modes are detected automatically:

    1. **CUDA Unified Memory** (desktop/datacenter GPU with Pascal+).
       Uses ``cudaMallocManaged`` so the same pointer is valid on CPU and GPU.
       ``migrate()`` is a lightweight prefetch hint, not a copy.

    2. **CUDA Zero-Copy / Mapped** (older GPUs or explicit host pinning).
       Uses ``cudaHostAllocMapped`` so CPU-allocated pages are mapped into the
       GPU address space.  The pointer is identical on both sides.

    3. **Jetson Shared Memory** (TEGRA / Jetson Nano/Orin).
       CPU and GPU share the same physical DRAM.  A plain ``np.zeros`` array
       is already unified — no migrate needed.

    4. **NumPy Fallback** (no CUDA).
       Uses plain host numpy arrays.  ``migrate()`` copies data to a new
       array and updates the handle pointer.

    All modes present the same API so higher-level code never branches.
    """

    def __init__(self, capacity_bytes: int = 512 * 1024 * 1024):
        self.capacity = capacity_bytes
        self._used: int = 0
        self._handles: Dict[str, MemoryHandle] = {}
        self._mode: str = self._detect_mode()
        log.info(
            "UnifiedMemoryPool(capacity=%d MB, mode=%s)",
            capacity_bytes // (1024 * 1024),
            self._mode,
        )

    # ── Mode detection ───────────────────────────────────────

    def _detect_mode(self) -> str:
        if _CUDA_AVAILABLE:
            # Check if we're on Jetson (shared memory)
            try:
                # Jetson devices report integrated GPU and unified memory
                props = _CUPY_MODULE.cuda.Device(0).attributes
                # integrated == 1 means shared CPU/GPU memory (Jetson)
                if props.get("Integrated", 0) == 1:
                    return "jetson"
            except Exception:
                pass
            # Prefer managed memory (Pascal+); fallback to mapped
            try:
                # Quick probe: can we allocate 1 byte of managed memory?
                probe = _CUPY_MODULE.cuda.malloc_managed(1)
                _CUPY_MODULE.cuda.free(probe)
                return "managed"
            except Exception:
                try:
                    probe = _CUPY_MODULE.cuda.host_alloc(
                        1, _CUPY_MODULE.cuda.host_alloc_mapped
                    )
                    _CUPY_MODULE.cuda.free(probe)
                    return "mapped"
                except Exception:
                    return "cuda-fallback"
        return "numpy"

    # ── Allocation ───────────────────────────────────────────

    def allocate(self, size_bytes: int, device: str = "cpu") -> MemoryHandle:
        """Allocate memory on the specified device.

        Args:
            size_bytes: Number of bytes to allocate (must be > 0).
            device:     Target device — "cpu", "gpu", "npu", or "jetson".

        Returns:
            MemoryHandle with a device-accessible pointer.

        Raises:
            ValueError:  If size_bytes <= 0 or device is unsupported.
            MemoryError: If allocation exceeds pool capacity.
        """
        if size_bytes <= 0:
            raise ValueError(f"size_bytes must be positive, got {size_bytes}")
        if device not in {"cpu", "gpu", "npu", "jetson", "managed"}:
            raise ValueError(f"Unsupported device '{device}'")

        if self._used + size_bytes > self.capacity:
            raise MemoryError(
                f"Pool capacity exceeded: used={self._used}B + "
                f"request={size_bytes}B > capacity={self.capacity}B"
            )

        handle = MemoryHandle(size_bytes=size_bytes, device=device)

        if self._mode == "jetson":
            # Jetson: CPU/GPU share DRAM — plain numpy is already unified
            arr = np.zeros(size_bytes, dtype=np.uint8, order="C")
            handle.ptr = arr.ctypes.data
            handle.device = "jetson"
            handle.unified = True
            handle._arrays["jetson"] = arr

        elif self._mode == "managed":
            # cudaMallocManaged — same pointer on CPU and GPU
            arr = _CUPY_MODULE.cuda.malloc_managed(size_bytes)
            handle.ptr = int(arr)
            handle.device = "managed"
            handle.unified = True
            handle._arrays["managed"] = arr

        elif self._mode == "mapped":
            # cudaHostAllocMapped — host memory mapped into GPU address space
            arr = _CUPY_MODULE.cuda.host_alloc(
                size_bytes, _CUPY_MODULE.cuda.host_alloc_mapped
            )
            handle.ptr = int(arr)
            handle.device = "cpu"  # lives on host, visible to GPU
            handle.unified = True
            handle._arrays["cpu"] = arr

        elif self._mode == "cuda-fallback":
            # CUDA available but unified allocators failed — use device mem
            arr = _CUPY_MODULE.cuda.alloc(size_bytes)
            handle.ptr = int(arr)
            handle.device = device
            handle.unified = False
            handle._arrays[device] = arr

        else:
            # NumPy fallback — host array, optionally mmap-backed
            arr = self._allocate_numpy(size_bytes)
            handle.ptr = arr.ctypes.data
            handle.device = "cpu"
            handle.unified = False
            handle._arrays["cpu"] = arr

        self._used += size_bytes
        self._handles[handle.handle_id] = handle
        log.debug("Allocated %s", handle)
        return handle

    def _allocate_numpy(self, size_bytes: int) -> np.ndarray:
        """Allocate a host numpy array, optionally backed by mmap."""
        # For large allocations use mmap so other processes can share
        if size_bytes >= 64 * 1024 * 1024:  # 64 MB threshold
            try:
                fd, path = tempfile.mkstemp(suffix=".sunset_umem")
                os.truncate(fd, size_bytes)
                os.close(fd)
                mm = mmap.mmap(path, size_bytes, access=mmap.ACCESS_WRITE)
                arr = np.frombuffer(mm, dtype=np.uint8)
                # Keep mm alive via the array's base so it doesn't get collected
                arr._sunset_mmap = mm  # type: ignore[attr-defined]
                arr._sunset_mmap_path = path  # type: ignore[attr-defined]
                return arr
            except Exception as exc:
                log.warning("mmap allocation failed (%s), using plain numpy", exc)
        return np.zeros(size_bytes, dtype=np.uint8, order="C")

    # ── Migration ──────────────────────────────────────────────

    def migrate(self, handle: MemoryHandle, target_device: str) -> MemoryHandle:
        """Migrate memory to another device.

        For unified memory (managed / mapped / jetson) this is a no-op
        because the same pointer is valid everywhere.

        For cuda-fallback or numpy fallback, this copies data to a new
        array on the target device and updates the handle.

        Args:
            handle:        Existing MemoryHandle.
            target_device: Destination device — "cpu", "gpu", "npu".

        Returns:
            The same handle (mutated in place) with updated ``device``
            and ``ptr`` fields.

        Raises:
            ValueError: If handle is invalid or target_device unsupported.
        """
        if not handle._valid:
            raise ValueError("Cannot migrate freed handle")
        if handle.handle_id not in self._handles:
            raise ValueError("Handle not owned by this pool")
        if target_device not in {"cpu", "gpu", "npu", "jetson", "managed"}:
            raise ValueError(f"Unsupported target device '{target_device}'")

        if handle.unified:
            # Zero-copy: pointer is valid everywhere.  Just update metadata.
            handle.device = target_device
            log.debug(
                "Migrate (no-op, unified) %s -> %s", handle.handle_id, target_device
            )
            return handle

        # ── Fallback copy path ────────────────────────────────
        old_device = handle.device
        old_arr = handle._arrays.get(old_device)

        if old_arr is None:
            raise RuntimeError(f"No array found for device '{old_device}'")

        if self._mode == "cuda-fallback" and _CUDA_AVAILABLE:
            if target_device == "cpu":
                new_arr = _CUPY_MODULE.asnumpy(old_arr)
            else:
                new_arr = _CUPY_MODULE.asarray(old_arr)
            handle._arrays[target_device] = new_arr
            handle.ptr = (
                int(new_arr.data) if hasattr(new_arr, "data") else new_arr.ctypes.data
            )
        else:
            # NumPy fallback: make a copy
            new_arr = np.array(old_arr, copy=True, order="C")
            handle._arrays[target_device] = new_arr
            handle.ptr = new_arr.ctypes.data

        handle.device = target_device
        log.debug(
            "Migrate (copy) %s %s -> %s", handle.handle_id, old_device, target_device
        )
        return handle

    # ── Pointer retrieval ────────────────────────────────────

    def get_pointer(self, handle: MemoryHandle, device: str = "cpu") -> int:
        """Get a device-accessible pointer for the handle.

        For unified memory this returns the same pointer regardless of device.
        For fallback memory it may trigger an implicit migrate if the handle
        is not currently on the requested device.
        """
        if not handle._valid:
            raise ValueError("Cannot get pointer for freed handle")
        if handle.unified:
            return handle.ptr
        if handle.device != device:
            self.migrate(handle, device)
        return handle.ptr

    # ── Free ─────────────────────────────────────────────────

    def free(self, handle: MemoryHandle) -> None:
        """Release memory back to the pool.

        After freeing, the handle is invalidated and must not be used.
        """
        if not handle._valid:
            log.debug("Double-free of %s — ignored", handle.handle_id)
            return
        if handle.handle_id not in self._handles:
            log.warning(
                "Free called on handle not owned by this pool: %s", handle.handle_id
            )
            return

        # Release underlying arrays
        for dev, arr in handle._arrays.items():
            if self._mode in ("managed", "mapped", "cuda-fallback") and _CUDA_AVAILABLE:
                try:
                    _CUPY_MODULE.cuda.free(arr)
                except Exception as exc:
                    log.debug("cuda.free failed for %s: %s", dev, exc)
            # numpy / mmap arrays are garbage-collected

        self._used -= handle.size_bytes
        del self._handles[handle.handle_id]
        handle.invalidate()
        log.debug("Freed handle %s", handle.handle_id)

    # ── Pool inspection ──────────────────────────────────────

    @property
    def used_bytes(self) -> int:
        return self._used

    @property
    def free_bytes(self) -> int:
        return self.capacity - self._used

    @property
    def mode(self) -> str:
        return self._mode

    def __repr__(self) -> str:
        return (
            f"UnifiedMemoryPool(capacity={self.capacity}B, "
            f"used={self._used}B, mode={self._mode}, handles={len(self._handles)})"
        )
