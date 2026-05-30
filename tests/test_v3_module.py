"""Tests for flux_compat.v3_module — FLUX v3 module representation."""

import pytest

from flux_compat.v3_module import Instruction, ConstraintDef, Module


class TestInstruction:
    def test_to_bytes_push(self):
        inst = Instruction(opcode="Push", operand=42)
        bc = inst.to_bytes()
        assert len(bc) == 5  # 1 opcode + 4 bytes i32

    def test_to_bytes_halt(self):
        inst = Instruction(opcode="Halt")
        bc = inst.to_bytes()
        assert len(bc) == 1

    def test_to_bytes_unknown(self):
        inst = Instruction(opcode="UnknownOp")
        bc = inst.to_bytes()
        assert bc[0] == 0x00  # falls back to 0x00

    def test_to_bytes_jump(self):
        inst = Instruction(opcode="FwdJump", operand=10)
        bc = inst.to_bytes()
        assert len(bc) == 3  # 1 opcode + 2 bytes i16

    def test_raw_bytes_default(self):
        inst = Instruction(opcode="Nop")
        assert inst.raw_bytes == b""


class TestConstraintDef:
    def test_create(self):
        c = ConstraintDef(kind="aviation", params={"altitude": 10000})
        assert c.kind == "aviation"
        assert c.params["altitude"] == 10000

    def test_default_params(self):
        c = ConstraintDef(kind="temperature")
        assert c.params == {}


class TestModule:
    def test_defaults(self):
        mod = Module()
        assert mod.version == 3
        assert mod.constants == []
        assert mod.instructions == []
        assert mod.constraints == []

    def test_to_bytecode(self):
        mod = Module(
            constants=[1, 2, 3],
            instructions=[
                Instruction(opcode="Push", operand=42),
                Instruction(opcode="Halt"),
            ],
        )
        bc = mod.to_bytecode()
        assert bc.startswith(b"FLX3")
        assert len(bc) > 8

    def test_disasm(self):
        mod = Module(
            instructions=[
                Instruction(opcode="Push", operand=42),
                Instruction(opcode="Halt"),
            ],
            warnings=["test warning"],
        )
        asm = mod.disasm()
        assert "FLUX v3" in asm
        assert "Push 42" in asm
        assert "Halt" in asm
        assert "test warning" in asm

    def test_disasm_empty(self):
        mod = Module()
        asm = mod.disasm()
        assert "constants: 0" in asm
        assert "instructions: 0" in asm

    def test_repr(self):
        mod = Module(version=3, constants=[1, 2])
        assert "Module" in repr(mod)
