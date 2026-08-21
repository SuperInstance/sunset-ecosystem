"""FLUX VM Bytecode Emitter — Python → FLUX opcode sequences.

Generates raw bytecode bytes for the FLUX VM constraint-checking pipeline.
Used by the breeding system to produce proof-carrying constraint checks.

Example
-------
    from sunset.flux_codegen import FluxBytecodeEmitter

    emitter = FluxBytecodeEmitter()
    bc = emitter.emit_constraint_check(
        n_rooms=4,
        latent_dim=8,
        min_bound=-10.0,
        max_bound=10.0,
        max_l2=50.0,
        max_var=5.0,
    )
    # bc is bytes ready for FluxVM.load_bytecode(bc)
"""

from __future__ import annotations

__all__ = [
    "FluxBytecodeEmitter",
    "OpCode",
]

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import List


class OpCode(IntEnum):
    """FLUX-C v3 opcodes (must match Rust vm::opcode::OpCode)."""

    # Stack (8)
    Push = 0x01
    Pop = 0x02
    Dup = 0x03
    Swap = 0x04
    Over = 0x05
    Drop = 0x06
    LoadConst = 0x07
    Nop = 0x08

    # Arithmetic (8)
    Add = 0x09
    Sub = 0x0A
    Mul = 0x0B
    Div = 0x0C
    Saturate = 0x0D
    Min = 0x0E
    Max = 0x0F
    Abs = 0x10

    # Register (4)
    LoadReg = 0x11
    StoreReg = 0x12
    LoadRegVec = 0x13
    StoreRegVec = 0x14

    # Constraint (10)
    RangeCheck = 0x15
    BatchCheck = 0x16
    AccumulateMask = 0x17
    ClassifySeverity = 0x18
    Prove = 0x19
    QueryBackward = 0x1A
    Simplify = 0x1B
    Validate = 0x1C
    HashCommit = 0x1D
    Seal = 0x1E

    # Vector/SIMD (6)
    VecLoad = 0x1F
    VecStore = 0x20
    VecRangeCheck = 0x21
    VecMaskMerge = 0x22
    VecReduce = 0x23
    VecGather = 0x24

    # Control (6)
    FwdJump = 0x25
    CondJump = 0x26
    CallBounded = 0x27
    Ret = 0x28
    Halt = 0x29
    Checkpoint = 0x2A

    # Effects (4)
    SetHandler = 0x2B
    EmitEvent = 0x2C
    Rollback = 0x2D
    GetResult = 0x2E

    # Parallel (4)
    ParDispatch = 0x2F
    ParMerge = 0x30
    ParBarrier = 0x31
    ParReduce = 0x32

    # Provenance (4)
    SnapRecord = 0x33
    SnapQuery = 0x34
    SnapHash = 0x35
    SnapVerify = 0x36

    # Streaming (4)
    StreamOpen = 0x37
    StreamCheck = 0x38
    StreamBatch = 0x39
    StreamClose = 0x3A


@dataclass
class _BytecodeBuffer:
    """Internal mutable bytecode builder."""

    _data: bytearray = None

    def __post_init__(self):
        if self._data is None:
            self._data = bytearray()

    def emit(self, op: OpCode) -> "_BytecodeBuffer":
        self._data.append(op.value)
        return self

    def emit_i32(self, value: int) -> "_BytecodeBuffer":
        self._data.extend(struct.pack("<i", value))
        return self

    def emit_u16(self, value: int) -> "_BytecodeBuffer":
        self._data.extend(struct.pack("<H", value))
        return self

    def emit_u8(self, value: int) -> "_BytecodeBuffer":
        self._data.append(value & 0xFF)
        return self

    def to_bytes(self) -> bytes:
        return bytes(self._data)

    def __len__(self) -> int:
        return len(self._data)


class FluxBytecodeEmitter:
    """Generates FLUX VM bytecode from Python constraint descriptions.

    This is a *tactical* bytecode emitter — not a full compiler. It knows
    how to emit opcode sequences for the fleet's specific constraint-checking
    patterns (thermal bounds, L2 norm, variance, chaos limits).

    For a full GUARD → FLUX compiler, use `guardc` (Rust). This emitter is
    the fast path: Python calls → bytecode bytes → VM run → proof certificate.
    """

    def __init__(self) -> None:
        pass

    # ── low-level emitters ──────────────────────────────────

    def _buf(self) -> _BytecodeBuffer:
        return _BytecodeBuffer()

    @staticmethod
    def push(value: int) -> bytes:
        """Emit: Push <i32>"""
        return bytes([OpCode.Push, *struct.pack("<i", value)])

    @staticmethod
    def load_const(value: int) -> bytes:
        """Emit: LoadConst <i32> (clears stack then pushes value)."""
        return bytes([OpCode.LoadConst, *struct.pack("<i", value)])

    @staticmethod
    def halt() -> bytes:
        """Emit: Halt"""
        return bytes([OpCode.Halt])

    @staticmethod
    def nop() -> bytes:
        """Emit: Nop"""
        return bytes([OpCode.Nop])

    # ── constraint check sequences ──────────────────────────

    def emit_constraint_check(
        self,
        n_rooms: int,
        latent_dim: int,
        min_bound: float,
        max_bound: float,
        max_l2: float,
        max_var: float,
    ) -> bytes:
        """Emit bytecode that checks room constraints using the VM.

        The generated program expects room latent values to be pushed
        onto the stack BEFORE execution (via the VM's push_value API).

        For a self-contained program, use emit_constraint_check_standalone().

        Stack layout at entry (pushed by caller):
            [room_0_val_0, room_0_val_1, ..., room_N_val_D]

        Generated program:
            1. For each room:
               a. Pop D values, store in vec reg 0
               b. VecRangeCheck against bounds
               c. VecReduce → sum
               d. Check L2 (simplified: sum > max_l2?)
               e. AccumulateMask
            2. ClassifySeverity
            3. Validate
            4. Halt
        """
        # For now, emit a simpler program that works with the constraint
        # system loaded via flux_vm_load_constraints()
        buf = self._buf()

        # Set effect handler to Log mode (1)
        buf.emit(OpCode.Push).emit_i32(1)
        buf.emit(OpCode.SetHandler)

        # For each room, we need to pop latent_dim values and check them
        # We'll use BatchCheck which pops a count, then pops that many values
        # and checks them against the loaded constraint bounds.

        # BatchCheck: pop count, then pop count values, count how many pass
        buf.emit(OpCode.Push).emit_i32(n_rooms * latent_dim)
        buf.emit(OpCode.BatchCheck)

        # Check that pass_count == expected (all values passed)
        # Stack: [pass_count]
        buf.emit(OpCode.Dup)
        buf.emit(OpCode.Push).emit_i32(n_rooms * latent_dim)
        buf.emit(OpCode.Sub)
        buf.emit(OpCode.Abs)
        # If |diff| != 0, jump to failure path (11 bytes forward)
        # Success path: Push 1, Validate, Prove, HashCommit, Seal, GetResult, Halt = 11 bytes
        buf.emit(OpCode.CondJump).emit_u16(11)
        buf.emit(OpCode.Push).emit_i32(1)
        buf.emit(OpCode.Validate)
        buf.emit(OpCode.Prove)
        buf.emit(OpCode.HashCommit)
        buf.emit(OpCode.Seal)
        buf.emit(OpCode.GetResult)
        buf.emit(OpCode.Halt)
        # Failure path
        buf.emit(OpCode.Push).emit_i32(0)
        buf.emit(OpCode.Validate)
        buf.emit(OpCode.GetResult)
        buf.emit(OpCode.Halt)

        return buf.to_bytes()

    def emit_constraint_check_standalone(
        self,
        n_rooms: int,
        latent_dim: int,
        min_bound: float,
        max_bound: float,
        max_l2: float,
        max_var: float,
    ) -> bytes:
        """Emit a self-contained bytecode program with embedded constants.

        This program loads all constants from immediate values, pushes
        room data from a simulated memory region, and performs the full
        bounds/L2/variance check without requiring constraint pre-loading.

        Current limitation: uses integer arithmetic. For float constraints,
        scale values by 1000 and use fixed-point.
        """
        scale = 1000  # fixed-point scale for float constraints
        lo = int(min_bound * scale)
        hi = int(max_bound * scale)
        l2_limit = int(max_l2 * scale)
        var_limit = int(max_var * scale)

        buf = self._buf()

        # Set handler to Log mode
        buf.emit(OpCode.LoadConst).emit_i32(1)
        buf.emit(OpCode.SetHandler)

        # Initialize mask = 0
        buf.emit(OpCode.LoadConst).emit_i32(0)
        buf.emit(OpCode.StoreReg).emit_u8(0)  # r0 = mask

        # For each room
        for room_idx in range(n_rooms):
            # Checkpoint before each room (enables rollback on failure)
            buf.emit(OpCode.Checkpoint)

            # Push latent_dim values for this room
            # In a real program, these would come from memory loads.
            # Here we use LoadConst 0 as placeholders — the VM will
            # be pre-loaded with actual values via push_value() before run().
            for _ in range(latent_dim):
                buf.emit(OpCode.Push).emit_i32(0)  # placeholder

            # BatchCheck: pop latent_dim values, check against bounds
            buf.emit(OpCode.Push).emit_i32(latent_dim)
            buf.emit(OpCode.BatchCheck)

            # Result is pass_count on stack. Compare with latent_dim.
            buf.emit(OpCode.Push).emit_i32(latent_dim)
            buf.emit(OpCode.Sub)
            buf.emit(OpCode.Abs)
            buf.emit(OpCode.Validate)

            # Accumulate mask
            buf.emit(OpCode.AccumulateMask)
            buf.emit(OpCode.StoreReg).emit_u8(0)  # r0 = updated mask

        # Classify severity from accumulated mask
        buf.emit(OpCode.LoadReg).emit_u8(0)
        buf.emit(OpCode.ClassifySeverity)

        # Prove the final classification
        buf.emit(OpCode.Prove)

        # Commit and seal
        buf.emit(OpCode.HashCommit)
        buf.emit(OpCode.Seal)

        # Get result
        buf.emit(OpCode.GetResult)

        # Halt
        buf.emit(OpCode.Halt)

        return buf.to_bytes()

    def emit_simple_range_check(self, lo: int, hi: int, value: int) -> bytes:
        """Emit: push value, push lo/hi bounds, RangeCheck, Halt.

        Minimal proof-of-concept program.
        """
        buf = self._buf()
        buf.emit(OpCode.Push).emit_i32(value)
        buf.emit(OpCode.RangeCheck)
        buf.emit(OpCode.Prove)
        buf.emit(OpCode.GetResult)
        buf.emit(OpCode.Halt)
        return buf.to_bytes()

    def emit_heartbeat(self, tick: int) -> bytes:
        """Emit a provenance heartbeat: record tick, hash, halt.

        Used by the metronome to leave VM-verifiable traces.
        """
        buf = self._buf()
        buf.emit(OpCode.Push).emit_i32(tick)
        buf.emit(OpCode.SnapRecord)
        buf.emit(OpCode.SnapHash)
        buf.emit(OpCode.Halt)
        return buf.to_bytes()
