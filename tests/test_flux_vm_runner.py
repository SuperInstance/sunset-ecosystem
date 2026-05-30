"""Tests for swarm.flux_vm_runner — FLUX VM bytecode interpreter."""

import struct

import pytest

from swarm.flux_vm_runner import FluxVMRunner, FluxTrap
from swarm.flux_compiler import FluxOpcode


class TestFluxVMRunner:
    def _push(self, val: float) -> bytes:
        return bytes([FluxOpcode.Push]) + struct.pack("<f", val)

    def _load_const(self, idx: int) -> bytes:
        return bytes([FluxOpcode.LoadConst, idx])

    def _fwd_jump(self, offset: int) -> bytes:
        return bytes([FluxOpcode.FwdJump]) + struct.pack("<H", offset)

    def _cond_jump(self, offset: int) -> bytes:
        return bytes([FluxOpcode.CondJump]) + struct.pack("<H", offset)

    def _range_check(self, lo: float, hi: float) -> bytes:
        return bytes([FluxOpcode.RangeCheck]) + struct.pack("<f", lo) + struct.pack("<f", hi)

    def test_push_and_halt(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(3.5) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(3.5)

    def test_add(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(2.0) + self._push(3.0) + bytes([FluxOpcode.Add]) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(5.0)

    def test_sub(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(5.0) + self._push(2.0) + bytes([FluxOpcode.Sub]) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(3.0)

    def test_mul(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(3.0) + self._push(4.0) + bytes([FluxOpcode.Mul]) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(12.0)

    def test_div(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(8.0) + self._push(2.0) + bytes([FluxOpcode.Div]) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(4.0)

    def test_div_by_zero(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(1.0) + self._push(0.0) + bytes([FluxOpcode.Div]) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == float("inf")

    def test_load_const(self):
        runner = FluxVMRunner([7.0, 8.0])
        bc = self._load_const(1) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(8.0)

    def test_dup(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(5.0) + bytes([FluxOpcode.Dup]) + bytes([FluxOpcode.Add]) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(10.0)

    def test_swap(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(2.0) + self._push(3.0) + bytes([FluxOpcode.Swap]) + bytes([FluxOpcode.Sub]) + bytes([FluxOpcode.Halt])
        # Stack: 2, 3 -> swap -> 3, 2 -> sub = 3 - 2 = 1
        assert runner.run(bc) == pytest.approx(1.0)

    def test_min(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(3.0) + self._push(1.0) + bytes([FluxOpcode.Min]) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(1.0)

    def test_max(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(3.0) + self._push(1.0) + bytes([FluxOpcode.Max]) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(3.0)

    def test_abs(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(-5.0) + bytes([FluxOpcode.Abs]) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(5.0)

    def test_range_check_pass(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(5.0) + self._range_check(0.0, 10.0) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(1.0)

    def test_range_check_fail(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(15.0) + self._range_check(0.0, 10.0) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(0.0)

    def test_validate_pass(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(1.0) + bytes([FluxOpcode.Validate]) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(1.0)

    def test_validate_fail(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(0.0) + bytes([FluxOpcode.Validate]) + bytes([FluxOpcode.Halt])
        with pytest.raises(FluxTrap):
            runner.run(bc)

    def test_fwd_jump(self):
        runner = FluxVMRunner([1.0])
        # push 1, push 2, jump over add+halt, push 3, halt
        # result should be 3 (add is skipped)
        bc = (
            self._push(1.0)
            + self._push(2.0)
            + self._fwd_jump(2)  # skip Add(1) + Halt(1)
            + bytes([FluxOpcode.Add])
            + bytes([FluxOpcode.Halt])
            + self._push(3.0)
            + bytes([FluxOpcode.Halt])
        )
        assert runner.run(bc) == pytest.approx(3.0)

    def test_cond_jump_taken(self):
        runner = FluxVMRunner([1.0])
        # push 0 (false), cond_jump, push 2, add, halt, push 3, halt
        # 0 is <= 0, so jump taken -> skip to push 3.0
        bc = (
            self._push(0.0)
            + self._cond_jump(12)  # skip Push 2.0(5) + Add(1) + Halt(1)
            + self._push(2.0)
            + self._push(3.0)
            + bytes([FluxOpcode.Add])
            + bytes([FluxOpcode.Halt])
            + self._push(4.0)
            + bytes([FluxOpcode.Halt])
        )
        assert runner.run(bc) == pytest.approx(4.0)

    def test_cond_jump_not_taken(self):
        runner = FluxVMRunner([1.0])
        # push 1 (true), cond_jump, push 2, push 3, add, halt, push 4, halt
        # 1 is > 0, so jump not taken -> execute push 2.0 + push 3.0 + add
        bc = (
            self._push(1.0)
            + self._cond_jump(12)  # skip Push 4.0(5) + Add(1) + Halt(1)
            + self._push(2.0)
            + self._push(3.0)
            + bytes([FluxOpcode.Add])
            + bytes([FluxOpcode.Halt])
            + self._push(4.0)
            + bytes([FluxOpcode.Halt])
        )
        assert runner.run(bc) == pytest.approx(5.0)

    def test_empty_stack(self):
        runner = FluxVMRunner([1.0])
        bc = bytes([FluxOpcode.Halt])
        assert runner.run(bc) == 0.0

    def test_pop(self):
        runner = FluxVMRunner([1.0])
        bc = self._push(1.0) + self._push(2.0) + bytes([FluxOpcode.Pop]) + bytes([FluxOpcode.Halt])
        assert runner.run(bc) == pytest.approx(1.0)
