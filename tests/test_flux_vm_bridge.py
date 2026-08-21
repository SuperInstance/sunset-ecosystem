"""Tests for the FLUX VM Python bridge and bytecode emitter.

These tests verify:
  1. FluxBytecodeEmitter produces valid bytecode sequences
  2. FluxVMBridge loads the shared library and manages VM lifecycle
  3. End-to-end: bytecode → VM → proof certificate
"""

from __future__ import annotations

import os
import struct
import sys

# Point to the dev tree .so if not copied yet
_dev_so = "/root/.openclaw/workspace/flux-vm-v3-temp/target/release/libflux_vm_v3.so"
if os.path.exists(_dev_so) and not os.environ.get("FLUX_VM_SO"):
    os.environ["FLUX_VM_SO"] = _dev_so

import pytest

from sunset.flux_codegen import FluxBytecodeEmitter, OpCode
from sunset.flux_vm_bridge import FluxVMBridge, FluxVMError


class TestBytecodeEmitter:
    """Verify bytecode sequences match expected opcode layouts."""

    def test_push_emits_5_bytes(self):
        bc = FluxBytecodeEmitter.push(42)
        assert len(bc) == 5
        assert bc[0] == OpCode.Push
        assert struct.unpack("<i", bc[1:5])[0] == 42

    def test_halt_emits_1_byte(self):
        bc = FluxBytecodeEmitter.halt()
        assert bc == bytes([OpCode.Halt])

    def test_simple_range_check_program(self):
        emitter = FluxBytecodeEmitter()
        bc = emitter.emit_simple_range_check(lo=-10, hi=10, value=5)
        assert bc[0] == OpCode.Push
        assert bc[5] == OpCode.RangeCheck
        assert bc[6] == OpCode.Prove
        assert bc[7] == OpCode.GetResult
        assert bc[8] == OpCode.Halt

    def test_constraint_check_program_structure(self):
        emitter = FluxBytecodeEmitter()
        bc = emitter.emit_constraint_check(
            n_rooms=2,
            latent_dim=4,
            min_bound=-10.0,
            max_bound=10.0,
            max_l2=50.0,
            max_var=5.0,
        )
        assert len(bc) > 10
        assert (
            bc[0] == OpCode.Push
        )  # handler mode (was LoadConst, now Push to preserve pre-loaded stack)
        assert bc[-1] == OpCode.Halt

    def test_heartbeat_program(self):
        emitter = FluxBytecodeEmitter()
        bc = emitter.emit_heartbeat(tick=1234)
        assert bc[0] == OpCode.Push
        assert struct.unpack("<i", bc[1:5])[0] == 1234
        assert bc[5] == OpCode.SnapRecord
        assert bc[6] == OpCode.SnapHash
        assert bc[7] == OpCode.Halt

    def test_standalone_program_has_checkpoints(self):
        emitter = FluxBytecodeEmitter()
        bc = emitter.emit_constraint_check_standalone(
            n_rooms=2,
            latent_dim=4,
            min_bound=-10.0,
            max_bound=10.0,
            max_l2=50.0,
            max_var=5.0,
        )
        assert OpCode.Checkpoint in bc
        assert OpCode.BatchCheck in bc
        assert OpCode.Halt in bc


class TestVMBridgeLifecycle:
    """Verify VM creation, reset, and destruction."""

    def test_bridge_loads_library(self):
        bridge = FluxVMBridge()
        assert bridge._lib is not None

    def test_new_creates_vm(self):
        bridge = FluxVMBridge().new()
        assert bridge._vm is not None
        bridge.free()

    def test_context_manager(self):
        with FluxVMBridge() as bridge:
            assert bridge._vm is not None
        # after exit, vm should be freed
        assert bridge._vm is None

    def test_reset_clears_state(self):
        with FluxVMBridge() as b:
            b.load_bytecode(FluxBytecodeEmitter.halt())
            b.reset()
            # after reset, bytecode is cleared; should still accept new bytecode
            b.load_bytecode(FluxBytecodeEmitter.halt())


class TestVMExecution:
    """Execute real bytecode on the VM and verify results."""

    def test_run_halt_passes(self):
        with FluxVMBridge() as b:
            b.load_bytecode(FluxBytecodeEmitter.halt())
            assert b.run() is True
            assert b.passed() is True
            assert b.get_cycles() > 0

    def test_run_simple_range_check_pass(self):
        with FluxVMBridge() as b:
            emitter = FluxBytecodeEmitter()
            bc = emitter.emit_simple_range_check(lo=-10, hi=10, value=5)
            b.load_bytecode(bc)
            b.load_constraint(lo=-10, hi=10)
            assert b.run() is True
            assert b.passed() is True

    def test_run_simple_range_check_fail(self):
        with FluxVMBridge() as b:
            emitter = FluxBytecodeEmitter()
            bc = emitter.emit_simple_range_check(lo=-10, hi=10, value=15)
            b.load_bytecode(bc)
            b.load_constraint(lo=-10, hi=10)
            assert b.run() is False
            assert b.passed() is False

    def test_proof_certificate_after_run(self):
        with FluxVMBridge() as b:
            emitter = FluxBytecodeEmitter()
            bc = emitter.emit_simple_range_check(lo=-10, hi=10, value=5)
            b.load_bytecode(bc)
            b.load_constraint(lo=-10, hi=10)
            b.run()
            proof = b.get_proof()
            assert proof is not None
            assert len(proof.root_hash) == 32
            assert proof.cycle_count > 0
            assert len(proof.hex) == 64  # 32 bytes * 2 hex chars

    def test_provenance_log_populated(self):
        with FluxVMBridge() as b:
            # Use a heartbeat program which includes SnapRecord
            emitter = FluxBytecodeEmitter()
            bc = emitter.emit_heartbeat(tick=1234)
            b.load_bytecode(bc)
            b.run()
            assert b.get_provenance_len() > 0

    def test_push_value_and_run(self):
        with FluxVMBridge() as b:
            # Program: Pop, Halt (pops the value we push)
            bc = bytes([OpCode.Pop, OpCode.Halt])
            b.load_bytecode(bc)
            b.push_value(42)
            assert b.run() is True

    def test_batch_check_with_pushed_values(self):
        with FluxVMBridge() as b:
            # Program: Push count, BatchCheck, Validate (pass_count > 0), Halt
            bc = (
                FluxBytecodeEmitter.push(3)
                + bytes([OpCode.BatchCheck])
                + bytes([OpCode.Validate])
                + bytes([OpCode.GetResult, OpCode.Halt])
            )
            b.load_bytecode(bc)
            b.load_constraint(lo=0, hi=100)
            # All 3 values pass [0, 100]
            b.push_values([10, 20, 30])
            assert b.run() is True

    def test_batch_check_all_must_pass(self):
        """Use conditional jump: fail if pass_count != expected."""
        with FluxVMBridge() as b:
            # Bytecode:
            #   Push 3, BatchCheck      → stack [pass_count]
            #   Dup                     → [pass_count, pass_count]
            #   Push 3, Sub             → [pass_count, diff]
            #   Abs                     → [pass_count, |diff|]
            #   Push 0, Swap            → [pass_count, 0, |diff|]
            #   Sub                     → [pass_count, -|diff|]
            #   Validate                → passes when -|diff| != 0 (i.e. diff>0)
            #   ... wait that's inverted too
            #
            # Simpler: just validate that pass_count == 3 by using:
            #   Dup, Push 3, Sub, Abs, Validate
            # When diff=0: stack [3, 0], Validate sees 0 → FAIL
            # When diff>0: stack [3, |diff|], Validate sees |diff| > 0 → PASS
            # That's backwards from what we want.
            #
            # Fix: Add 1 then subtract 1 is identity. Use Push 0, Sub to negate:
            #   ... after Abs we have |diff|
            #   Push 0, Sub → -|diff|
            #   Abs again → |diff|  (no change)
            #
            # Best fix: use a conditional jump to skip setting pass=False:
            #   After BatchCheck: [pass_count]
            #   Dup, Push 3, Sub, Abs  → [pass_count, |diff|]
            #   CondJump 4              → if |diff| != 0, jump to failure path
            #   Push 1                  # success path
            #   Validate
            #   GetResult, Halt
            #   Push 0                  # failure path (jump target)
            #   Validate
            #   GetResult, Halt
            # Success path after CondJump: Push 1, Validate, GetResult, Halt = 8 bytes
            # Failure path: Push 0, Validate, GetResult, Halt = 8 bytes
            bc = (
                FluxBytecodeEmitter.push(3)
                + bytes([OpCode.BatchCheck])
                + bytes([OpCode.Dup])
                + FluxBytecodeEmitter.push(3)
                + bytes([OpCode.Sub, OpCode.Abs])
                + bytes([OpCode.CondJump, 0x08, 0x00])  # jump 8 bytes if |diff| != 0
                + FluxBytecodeEmitter.push(1)
                + bytes([OpCode.Validate, OpCode.GetResult, OpCode.Halt])
                + FluxBytecodeEmitter.push(0)
                + bytes([OpCode.Validate, OpCode.GetResult, OpCode.Halt])
            )
            b.load_bytecode(bc)
            b.load_constraint(lo=0, hi=100)
            b.push_values([10, 20, 30])
            assert b.run() is True

    def test_batch_check_some_fail(self):
        """Same conditional-jump bytecode, but some values fail."""
        with FluxVMBridge() as b:
            bc = (
                FluxBytecodeEmitter.push(3)
                + bytes([OpCode.BatchCheck])
                + bytes([OpCode.Dup])
                + FluxBytecodeEmitter.push(3)
                + bytes([OpCode.Sub, OpCode.Abs])
                + bytes([OpCode.CondJump, 0x08, 0x00])  # jump 8 bytes if |diff| != 0
                + FluxBytecodeEmitter.push(1)
                + bytes([OpCode.Validate, OpCode.GetResult, OpCode.Halt])
                + FluxBytecodeEmitter.push(0)
                + bytes([OpCode.Validate, OpCode.GetResult, OpCode.Halt])
            )
            b.load_bytecode(bc)
            b.load_constraint(lo=0, hi=20)
            # 10, 20 pass; 30 fails → pass_count=2 != 3 → jump to failure path
            b.push_values([10, 20, 30])
            assert b.run() is False

    def test_check_rooms_high_level(self):
        with FluxVMBridge() as b:
            bc = FluxBytecodeEmitter().emit_constraint_check(
                n_rooms=2,
                latent_dim=4,
                min_bound=0.0,
                max_bound=100.0,
                max_l2=1000.0,
                max_var=100.0,
            )
            # 8 values, all within [0, 100]
            values = [10, 20, 30, 40, 50, 60, 70, 80]
            passed, proof = b.check_rooms(
                bytecode=bc,
                room_values=values,
                constraint_lo=0,
                constraint_hi=100,
            )
            assert passed is True
            assert proof is not None
            assert len(proof.root_hash) == 32


class TestVMErrorHandling:
    """Verify error paths raise FluxVMError."""

    def test_invalid_bytecode_raises(self):
        with FluxVMBridge() as b:
            # 0xFF is not a valid opcode
            b.load_bytecode(bytes([0xFF, 0x29]))
            with pytest.raises(FluxVMError):
                b.run()

    def test_reset_without_vm_raises(self):
        b = FluxVMBridge()
        b._vm = None
        # reset() calls _ensure_vm() which auto-creates, so test free() instead
        # free() with None should not raise
        b.free()
        # But load_bytecode without VM should auto-create
        # Test with a truly invalid scenario: _vm is a bad pointer
        # Actually, _ensure_vm auto-creates, so the only way to get an error
        # is if the VM creation itself fails (out of memory) or if we
        # pass a null pointer to a function that doesn't call _ensure_vm.
        # The FFI functions flux_vm_* all return -1 on null.
        # Let's test the error path via a manually set invalid vm pointer.
        b._vm = 0  # null pointer as integer
        with pytest.raises(FluxVMError):
            b.reset()
