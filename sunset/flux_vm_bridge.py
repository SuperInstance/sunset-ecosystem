"""FluxVMBridge — Python wrapper for the FLUX VM FFI.

Wraps the Rust `flux-vm-v3` shared library via ctypes, exposing:
  - VM lifecycle (new, free, reset)
  - Bytecode loading
  - Constraint pre-loading
  - Stack pre-loading (room latent values)
  - Execution (run)
  - Result extraction (pass/fail, cycles)
  - Proof certificate retrieval (SHA-256 root hash)
  - Provenance log access

Usage
-----
    from sunset.flux_vm_bridge import FluxVMBridge

    bridge = FluxVMBridge()
    bridge.load_bytecode(bytecode_bytes)
    bridge.load_constraint(lo=-10, hi=10)
    for value in room_latents:
        bridge.push_value(int(value * 1000))  # fixed-point
    passed = bridge.run()
    cycles = bridge.get_cycles()
    proof_hash = bridge.get_proof_hash()
"""

from __future__ import annotations

__all__ = ["FluxVMBridge", "FluxVMError", "FluxVMProof"]

import ctypes
import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


class FluxVMError(Exception):
    """Raised when the FLUX VM returns an error code or encounters a fault."""

    def __init__(self, code: int, message: str = "") -> None:
        self.code = code
        super().__init__(f"FluxVM error {code}{': ' + message if message else ''}")


@dataclass(frozen=True)
class FluxVMProof:
    """A proof certificate from a VM execution."""

    root_hash: bytes  # 32 bytes SHA-256
    cycle_count: int
    hex: str

    def verify(self, expected: bytes) -> bool:
        """Verify this proof against an expected root hash."""
        return self.root_hash == expected

    def to_dict(self) -> dict:
        return {
            "root_hash": self.hex,
            "cycle_count": self.cycle_count,
        }


class FluxVMBridge:
    """Python bridge to the FLUX-C v3 VM via ctypes.

    Automatically discovers the shared library in common locations:
      1. ``FLUX_VM_SO`` environment variable (absolute path)
      2. ``sunset-ecosystem/flux_vm/libflux_vm_v3.so`` (copied build)
      3. ``../flux-vm-v3-temp/target/release/libflux_vm_v3.so`` (dev tree)
      4. System LD_LIBRARY_PATH
    """

    _SO_NAME = "libflux_vm_v3.so"

    def __init__(self, so_path: Optional[str] = None) -> None:
        self._lib = self._load_library(so_path)
        self._vm: Optional[int] = None  # opaque pointer as int
        self._last_cycles: int = 0
        self._last_pass: bool = False

    # ── library discovery ───────────────────────────────────

    @classmethod
    def _find_so(cls) -> str:
        """Discover the shared library path."""
        # 1. Environment override
        env_path = os.environ.get("FLUX_VM_SO")
        if env_path and Path(env_path).exists():
            return env_path

        # 2. Copied build inside sunset-ecosystem
        here = Path(__file__).resolve().parent
        candidate = here.parent / "flux_vm" / cls._SO_NAME
        if candidate.exists():
            return str(candidate)

        # 3. Dev tree adjacent to sunset-ecosystem
        dev = here.parent.parent / "flux-vm-v3-temp" / "target" / "release" / cls._SO_NAME
        if dev.exists():
            return str(dev)

        # 4. Debug build fallback
        dev_debug = here.parent.parent / "flux-vm-v3-temp" / "target" / "debug" / cls._SO_NAME
        if dev_debug.exists():
            return str(dev_debug)

        raise FluxVMError(-1, f"Cannot find {cls._SO_NAME}. Set FLUX_VM_SO or build flux-vm-v3.")

    def _load_library(self, so_path: Optional[str]) -> ctypes.CDLL:
        path = so_path or self._find_so()
        lib = ctypes.CDLL(path)

        # Bind function signatures
        lib.flux_vm_new.restype = ctypes.c_void_p
        lib.flux_vm_new.argtypes = []

        lib.flux_vm_free.restype = None
        lib.flux_vm_free.argtypes = [ctypes.c_void_p]

        lib.flux_vm_load_bytecode.restype = ctypes.c_int
        lib.flux_vm_load_bytecode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint]

        lib.flux_vm_load_constraint.restype = ctypes.c_int
        lib.flux_vm_load_constraint.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

        lib.flux_vm_push_value.restype = ctypes.c_int
        lib.flux_vm_push_value.argtypes = [ctypes.c_void_p, ctypes.c_int]

        lib.flux_vm_run.restype = ctypes.c_int
        lib.flux_vm_run.argtypes = [ctypes.c_void_p]

        lib.flux_vm_get_result.restype = ctypes.c_int
        lib.flux_vm_get_result.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_int)]

        lib.flux_vm_get_proof.restype = ctypes.c_int
        lib.flux_vm_get_proof.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint]

        lib.flux_vm_get_provenance_len.restype = ctypes.c_int
        lib.flux_vm_get_provenance_len.argtypes = [ctypes.c_void_p]

        lib.flux_vm_reset.restype = ctypes.c_int
        lib.flux_vm_reset.argtypes = [ctypes.c_void_p]

        return lib

    # ── VM lifecycle ────────────────────────────────────────

    def new(self) -> "FluxVMBridge":
        """Create a new VM instance. Idempotent if already created."""
        if self._vm is None:
            self._vm = self._lib.flux_vm_new()
            if not self._vm:
                raise FluxVMError(-1, "flux_vm_new returned null")
        return self

    def free(self) -> None:
        """Destroy the VM instance."""
        if self._vm is not None:
            self._lib.flux_vm_free(self._vm)
            self._vm = None

    def reset(self) -> "FluxVMBridge":
        """Reset VM state (stack, registers, bytecode, etc.)."""
        self._ensure_vm()
        ret = self._lib.flux_vm_reset(self._vm)
        if ret != 0:
            raise FluxVMError(ret, "flux_vm_reset failed")
        return self

    def __enter__(self) -> "FluxVMBridge":
        self.new()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.free()

    # ── loading ──────────────────────────────────────────────

    def load_bytecode(self, bytecode: bytes) -> "FluxVMBridge":
        """Load a bytecode program into the VM."""
        self._ensure_vm()
        arr = (ctypes.c_ubyte * len(bytecode)).from_buffer_copy(bytecode)
        ret = self._lib.flux_vm_load_bytecode(self._vm, arr, len(bytecode))
        if ret != 0:
            raise FluxVMError(ret, "flux_vm_load_bytecode failed")
        return self

    def load_constraint(self, lo: int, hi: int) -> "FluxVMBridge":
        """Load a scalar range constraint [lo, hi] for RangeCheck / BatchCheck."""
        self._ensure_vm()
        ret = self._lib.flux_vm_load_constraint(self._vm, lo, hi)
        if ret != 0:
            raise FluxVMError(ret, "flux_vm_load_constraint failed")
        return self

    def push_value(self, value: int) -> "FluxVMBridge":
        """Push a single i32 onto the VM stack (used for room latent values)."""
        self._ensure_vm()
        ret = self._lib.flux_vm_push_value(self._vm, value)
        if ret != 0:
            raise FluxVMError(ret, "flux_vm_push_value failed")
        return self

    def push_values(self, values: List[int]) -> "FluxVMBridge":
        """Push multiple i32 values onto the VM stack."""
        for v in values:
            self.push_value(v)
        return self

    # ── execution ────────────────────────────────────────────

    def run(self) -> bool:
        """Execute the loaded bytecode.

        Returns True if the program halted with pass=True.
        Returns False if the program halted with pass=False (constraint violation).

        Raises FluxVMError on VM fault (invalid opcode, cycle limit, etc.).
        """
        self._ensure_vm()
        ret = self._lib.flux_vm_run(self._vm)
        if ret == 1:
            self._last_pass = True
        elif ret == 0:
            self._last_pass = False
        else:
            raise FluxVMError(ret, f"flux_vm_run failed (return={ret})")

        # Pull detailed result
        cycles = ctypes.c_ulonglong()
        pass_flag = ctypes.c_int()
        res = self._lib.flux_vm_get_result(self._vm, ctypes.byref(cycles), ctypes.byref(pass_flag))
        if res == 0:
            self._last_cycles = cycles.value
            self._last_pass = bool(pass_flag.value)
        return self._last_pass

    # ── result accessors ─────────────────────────────────────

    def passed(self) -> bool:
        """Did the last run pass?"""
        return self._last_pass

    def get_cycles(self) -> int:
        """Cycle count from the last run."""
        return self._last_cycles

    def get_proof_hash(self) -> Optional[bytes]:
        """Retrieve the 32-byte SHA-256 proof certificate root hash."""
        self._ensure_vm()
        buf = (ctypes.c_ubyte * 32)()
        ret = self._lib.flux_vm_get_proof(self._vm, buf, 32)
        if ret == 32:
            return bytes(buf)
        return None

    def get_proof(self) -> Optional[FluxVMProof]:
        """Retrieve a full FluxVMProof object."""
        h = self.get_proof_hash()
        if h is None:
            return None
        return FluxVMProof(
            root_hash=h,
            cycle_count=self._last_cycles,
            hex=h.hex(),
        )

    def get_provenance_len(self) -> int:
        """Number of provenance log entries."""
        self._ensure_vm()
        return self._lib.flux_vm_get_provenance_len(self._vm)

    # ── helpers ──────────────────────────────────────────────

    def _ensure_vm(self) -> None:
        if self._vm is None:
            self.new()

    # ── high-level: constraint check a room batch ────────────

    def check_rooms(
        self,
        bytecode: bytes,
        room_values: List[int],
        constraint_lo: int,
        constraint_hi: int,
    ) -> Tuple[bool, Optional[FluxVMProof]]:
        """One-shot: reset, load bytecode + constraint + values, run, return proof.

        This is the primary API for the breeding pipeline.
        """
        self.reset()
        self.load_bytecode(bytecode)
        self.load_constraint(constraint_lo, constraint_hi)
        self.push_values(room_values)
        passed = self.run()
        proof = self.get_proof()
        return passed, proof
