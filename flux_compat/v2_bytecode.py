"""Legacy FLUX v2 bytecode format definitions and parser.

v2 never shipped source code — only ISA definitions (flux-isa, flux-isa-mini,
flux-isa-edge, flux-isa-std, flux-isa-thor, flux-ast).  This module invents a
reasonable v2 binary encoding so the compat layer has something concrete to
parse and translate.

v2 bytecode layout (invented but grounded in typical stack-VM practice):
    Offset  Size  Meaning
    0       4     Magic: b'FLX2'
    4       1     Version minor (usually 0)
    5       1     Flags (bit 0 = has_constraints, bit 1 = has_debug)
    6       2     Constant pool count (N)
    8       N*4   Constant pool (i32 little-endian)
    8+N*4   2     Instruction count (M)
    10+N*4  M*?   Instructions (opcode u8 + optional imm)
    ???     2     Constraint count (C)  [if flags bit 0 set]
    ???     C*?   Constraint payloads     [if flags bit 0 set]
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, BinaryIO, Tuple


# v2 opcodes — a ~30-instruction set derived from the ISA definitions.
# Grouped by the same taxonomy v3 uses so mapping is readable.
V2_OPCODES: dict[int, str] = {
    # Stack (6) — v2 lacked Over and Nop
    0x01: "Push",
    0x02: "Pop",
    0x03: "Dup",
    0x04: "Swap",
    0x05: "Drop",
    0x06: "LoadConst",
    # Arithmetic (6) — v2 lacked Saturate and Abs
    0x10: "Add",
    0x11: "Sub",
    0x12: "Mul",
    0x13: "Div",
    0x14: "Min",
    0x15: "Max",
    # Register (2) — v2 had no vector registers
    0x20: "LoadReg",
    0x21: "StoreReg",
    # Legacy constraint (5) — simpler than v3
    0x30: "Check",
    0x31: "Assert",
    0x32: "Range",
    0x33: "Batch",
    0x34: "Accumulate",
    # Control (5) — v2 jump was unbounded, no checkpoint
    0x40: "Jump",
    0x41: "JumpZero",
    0x42: "Call",
    0x43: "Ret",
    0x44: "Halt",
    # Legacy I/O (2) — v2 treated I/O as read/write ports
    0x50: "Read",
    0x51: "Write",
    # Debug / meta (1)
    0x60: "Break",
}

V2_OPCODE_IMM_BYTES: dict[str, int] = {
    "Push": 4,
    "LoadConst": 4,
    "LoadReg": 1,
    "StoreReg": 1,
    "Jump": 2,
    "JumpZero": 2,
    "Call": 2,
    "Range": 2,        # lower, upper bounds packed as two i8s
    "Batch": 1,        # batch size
    "Read": 1,         # port id
    "Write": 1,        # port id
}


@dataclass
class V2Instruction:
    opcode: str
    raw: int
    operand: Optional[int] = None


@dataclass
class V2Constraint:
    kind: str
    payload: bytes


@dataclass
class V2Module:
    """Parsed v2 module — intermediate representation before translation."""
    magic: bytes
    version_minor: int
    flags: int
    constants: List[int]
    instructions: List[V2Instruction]
    constraints: List[V2Constraint] = field(default_factory=list)
    debug_symbols: Optional[dict] = None


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError(f"Unexpected EOF: wanted {n} bytes, got {len(data)}")
    return data


def parse_v2(path: str) -> V2Module:
    """Parse a v2 bytecode file into a V2Module."""
    with open(path, "rb") as f:
        magic = _read_exact(f, 4)
        if magic != b"FLX2":
            raise ValueError(f"Bad v2 magic: expected b'FLX2', got {magic!r}")

        version_minor = struct.unpack("<B", _read_exact(f, 1))[0]
        flags = struct.unpack("<B", _read_exact(f, 1))[0]

        const_count = struct.unpack("<H", _read_exact(f, 2))[0]
        constants = [
            struct.unpack("<i", _read_exact(f, 4))[0]
            for _ in range(const_count)
        ]

        inst_count = struct.unpack("<H", _read_exact(f, 2))[0]
        instructions: List[V2Instruction] = []
        for _ in range(inst_count):
            raw_op = struct.unpack("<B", _read_exact(f, 1))[0]
            op_name = V2_OPCODES.get(raw_op)
            if op_name is None:
                raise ValueError(f"Unknown v2 opcode: 0x{raw_op:02x}")
            imm_sz = V2_OPCODE_IMM_BYTES.get(op_name, 0)
            operand: Optional[int] = None
            if imm_sz == 4:
                operand = struct.unpack("<i", _read_exact(f, 4))[0]
            elif imm_sz == 2:
                operand = struct.unpack("<h", _read_exact(f, 2))[0]
            elif imm_sz == 1:
                operand = struct.unpack("<B", _read_exact(f, 1))[0]
            instructions.append(V2Instruction(opcode=op_name, raw=raw_op, operand=operand))

        constraints: List[V2Constraint] = []
        if flags & 0x01:
            c_count = struct.unpack("<H", _read_exact(f, 2))[0]
            for _ in range(c_count):
                # v2 constraints are opaque kind-byte + length-prefixed payload
                c_kind = struct.unpack("<B", _read_exact(f, 1))[0]
                c_len = struct.unpack("<H", _read_exact(f, 2))[0]
                c_payload = _read_exact(f, c_len)
                kind_str = {0x01: "aviation", 0x02: "temperature", 0x03: "custom"}.get(c_kind, "unknown")
                constraints.append(V2Constraint(kind=kind_str, payload=c_payload))

        debug_symbols = None
        if flags & 0x02:
            # v2 debug: just skip for now; not needed for translation
            d_len = struct.unpack("<I", _read_exact(f, 4))[0]
            f.read(d_len)  # skip

        return V2Module(
            magic=magic,
            version_minor=version_minor,
            flags=flags,
            constants=constants,
            instructions=instructions,
            constraints=constraints,
            debug_symbols=debug_symbols,
        )
