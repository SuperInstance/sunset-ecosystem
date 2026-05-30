"""Tests for flux_compat.v2_bytecode — legacy FLUX v2 parser."""

import struct
import tempfile

import pytest

from flux_compat.v2_bytecode import (
    V2Instruction,
    V2Constraint,
    V2Module,
    parse_v2,
    V2_OPCODES,
)


def _build_v2(constants=None, instructions=None, constraints=None, flags=0) -> bytes:
    """Build a valid v2 bytecode blob in memory."""
    constants = constants or []
    instructions = instructions or []
    constraints = constraints or []

    data = b"FLX2"
    data += struct.pack("<B", 0)  # version minor
    flags = flags | (0x01 if constraints else 0x00) | (0x02 if False else 0x00)
    data += struct.pack("<B", flags)
    data += struct.pack("<H", len(constants))
    for c in constants:
        data += struct.pack("<i", c)
    data += struct.pack("<H", len(instructions))
    for raw_op, operand in instructions:
        data += struct.pack("<B", raw_op)
        op_name = V2_OPCODES.get(raw_op)
        if op_name is None:
            continue  # unknown opcode, no operand
        from flux_compat.v2_bytecode import V2_OPCODE_IMM_BYTES
        imm_sz = V2_OPCODE_IMM_BYTES.get(op_name, 0)
        if imm_sz == 4:
            data += struct.pack("<i", operand or 0)
        elif imm_sz == 2:
            data += struct.pack("<h", operand or 0)
        elif imm_sz == 1:
            data += struct.pack("<B", operand or 0)
    if flags & 0x01:
        data += struct.pack("<H", len(constraints))
        for c_kind, c_payload in constraints:
            data += struct.pack("<B", c_kind)
            data += struct.pack("<H", len(c_payload))
            data += c_payload
    return data


class TestParseV2:
    def test_minimal(self):
        bc = _build_v2()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(bc)
            path = f.name
        mod = parse_v2(path)
        assert isinstance(mod, V2Module)
        assert mod.magic == b"FLX2"
        assert mod.constants == []
        assert mod.instructions == []

    def test_bad_magic(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"BAD!" + b"\x00" * 10)
            path = f.name
        with pytest.raises(ValueError, match="Bad v2 magic"):
            parse_v2(path)

    def test_constants(self):
        bc = _build_v2(constants=[1, 2, 3])
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(bc)
            path = f.name
        mod = parse_v2(path)
        assert mod.constants == [1, 2, 3]

    def test_instructions(self):
        # Push(0x01) with 4-byte immediate, then Halt(0x44)
        bc = _build_v2(instructions=[(0x01, 42), (0x44, None)])
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(bc)
            path = f.name
        mod = parse_v2(path)
        assert len(mod.instructions) == 2
        assert mod.instructions[0].opcode == "Push"
        assert mod.instructions[0].operand == 42
        assert mod.instructions[1].opcode == "Halt"

    def test_unknown_opcode(self):
        bc = _build_v2(instructions=[(0x99, None)])
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(bc)
            path = f.name
        with pytest.raises(ValueError, match="Unknown v2 opcode"):
            parse_v2(path)

    def test_constraints(self):
        bc = _build_v2(
            instructions=[(0x44, None)],
            constraints=[(0x01, b"aviation_data")],
        )
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(bc)
            path = f.name
        mod = parse_v2(path)
        assert len(mod.constraints) == 1
        assert mod.constraints[0].kind == "aviation"
        assert mod.constraints[0].payload == b"aviation_data"

    def test_empty_constraints(self):
        bc = _build_v2(instructions=[(0x44, None)], constraints=[])
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(bc)
            path = f.name
        mod = parse_v2(path)
        assert mod.constraints == []

    def test_repr(self):
        inst = V2Instruction(opcode="Push", raw=0x01, operand=42)
        assert "Push" in repr(inst)

    def test_constraint_repr(self):
        c = V2Constraint(kind="custom", payload=b"data")
        assert "custom" in repr(c)
