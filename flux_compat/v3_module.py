"""FLUX v3 Module representation in Python.

Mirrors the Rust v3 structure enough for the compat layer to produce
a loadable, inspectable module object from legacy v2 bytecode.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Instruction:
    """Single v3 instruction with optional immediate operand."""
    opcode: str          # e.g. "Push", "RangeCheck"
    operand: Optional[int] = None
    raw_bytes: bytes = field(default=b"", repr=False)

    def to_bytes(self) -> bytes:
        """Serialize to v3 raw bytecode (opcode u8 + optional imm)."""
        from .opcode_map import V3_OPCODE_BYTES
        op_byte = V3_OPCODE_BYTES.get(self.opcode, 0x00)
        out = bytes([op_byte])
        if self.operand is not None:
            if self.opcode in ("Push", "LoadConst"):
                out += struct.pack("<i", self.operand)
            elif self.opcode in ("LoadReg", "StoreReg", "LoadRegVec", "StoreRegVec"):
                out += struct.pack("<B", self.operand & 0xFF)
            elif self.opcode in ("FwdJump", "CondJump", "CallBounded"):
                out += struct.pack("<h", self.operand)
        return out


@dataclass
class ConstraintDef:
    """Constraint definition compatible with v3 check.rs presets."""
    kind: str            # "aviation", "temperature", "custom"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Module:
    """A v3 FLUX module — the canonical format produced by load_v2()."""
    version: int = 3
    constants: List[int] = field(default_factory=list)
    instructions: List[Instruction] = field(default_factory=list)
    constraints: List[ConstraintDef] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_bytecode(self) -> bytes:
        """Flatten module to a v3 bytecode blob (for VM ingestion)."""
        header = b"FLX3" + struct.pack("<HH", self.version, len(self.constants))
        const_blob = b"".join(struct.pack("<i", c) for c in self.constants)
        inst_blob = b"".join(i.to_bytes() for i in self.instructions)
        return header + const_blob + inst_blob

    def disasm(self) -> str:
        """Human-readable disassembly."""
        lines = [
            f"; FLUX v{self.version} module",
            f"; constants: {len(self.constants)}",
            f"; instructions: {len(self.instructions)}",
        ]
        for idx, inst in enumerate(self.instructions):
            op_str = inst.opcode
            if inst.operand is not None:
                op_str += f" {inst.operand}"
            lines.append(f"{idx:04x}:  {op_str}")
        if self.warnings:
            lines.append("; warnings during translation:")
            for w in self.warnings:
                lines.append(f";   ⚠ {w}")
        return "\n".join(lines)
