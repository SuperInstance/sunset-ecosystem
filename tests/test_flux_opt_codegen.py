"""Tests for FLUX optimization bytecode generator.

Run: python3 -m pytest tests/test_flux_opt_codegen.py -v --tb=short
"""

from __future__ import annotations

import pytest

from flux_compat.flux_opt_codegen import (
    FLUXOptimizerCodegen,
    generate_direct_module,
    generate_esch_module,
    generate_crs2lm_module,
)
from flux_compat.v3_module import Instruction, Module


# ── helpers ────────────────────────────────────────────────


def _collect_opcodes(mod: Module) -> list[str]:
    """Return list of opcode mnemonic strings from a module."""
    return [inst.opcode for inst in mod.instructions]


def _has_opcode_sequence(mod: Module, *expected: str) -> bool:
    """Check if the module contains a contiguous subsequence of opcodes."""
    ops = _collect_opcodes(mod)
    if not expected:
        return True
    for i in range(len(ops) - len(expected) + 1):
        if ops[i : i + len(expected)] == list(expected):
            return True
    return False


# ── Module validity ────────────────────────────────────────


class TestModuleValidity:
    def test_direct_module_valid(self):
        mod = generate_direct_module(
            dim=2,
            bounds=[(0.0, 1.0), (-1.0, 2.0)],
            maxeval=1000,
            ftol=1e-4,
        )
        assert isinstance(mod, Module)
        assert mod.version == 3
        assert len(mod.instructions) > 0
        assert len(mod.constants) > 0
        assert mod.metadata["algorithm"] == "DIRECT"

    def test_esch_module_valid(self):
        mod = generate_esch_module(
            dim=3,
            bounds=[(0.0, 10.0)] * 3,
            pop_size=20,
            maxeval=5000,
        )
        assert isinstance(mod, Module)
        assert mod.version == 3
        assert len(mod.instructions) > 0
        assert len(mod.constants) > 0
        assert mod.metadata["algorithm"] == "ESCH"

    def test_crs2lm_module_valid(self):
        mod = generate_crs2lm_module(
            dim=4,
            bounds=[(-5.0, 5.0)] * 4,
            pop_size=8,
        )
        assert isinstance(mod, Module)
        assert mod.version == 3
        assert len(mod.instructions) > 0
        assert len(mod.constants) > 0
        assert mod.metadata["algorithm"] == "CRS2-LM"

    def test_direct_bytecode_parsable(self):
        mod = generate_direct_module(
            dim=2, bounds=[(0.0, 1.0)] * 2, maxeval=100, ftol=1e-3
        )
        bc = mod.to_bytecode()
        assert bc.startswith(b"FLX3")
        # Header: 4 magic + 2 version + 2 const_count
        assert len(bc) >= 8
        # Should contain instructions after header + constants
        assert len(bc) > 8 + len(mod.constants) * 4

    def test_esch_bytecode_parsable(self):
        mod = generate_esch_module(
            dim=2, bounds=[(0.0, 1.0)] * 2, pop_size=10, maxeval=500
        )
        bc = mod.to_bytecode()
        assert bc.startswith(b"FLX3")
        assert len(bc) >= 8

    def test_crs2lm_bytecode_parsable(self):
        mod = generate_crs2lm_module(dim=3, bounds=[(-1.0, 1.0)] * 3, pop_size=6)
        bc = mod.to_bytecode()
        assert bc.startswith(b"FLX3")
        assert len(bc) >= 8


# ── Required opcodes ───────────────────────────────────────


class TestRequiredOpcodes:
    def test_direct_has_vecload(self):
        mod = generate_direct_module(2, [(0.0, 1.0)] * 2, 100, 1e-3)
        assert "VecLoad" in _collect_opcodes(mod)

    def test_direct_has_vecstore(self):
        mod = generate_direct_module(2, [(0.0, 1.0)] * 2, 100, 1e-3)
        assert "VecStore" in _collect_opcodes(mod)

    def test_direct_has_rangecheck(self):
        mod = generate_direct_module(2, [(0.0, 1.0)] * 2, 100, 1e-3)
        assert "RangeCheck" in _collect_opcodes(mod)

    def test_direct_has_callbounded(self):
        mod = generate_direct_module(2, [(0.0, 1.0)] * 2, 100, 1e-3)
        assert "CallBounded" in _collect_opcodes(mod)

    def test_esch_has_pardispatch(self):
        mod = generate_esch_module(2, [(0.0, 1.0)] * 2, 10, 500)
        assert "ParDispatch" in _collect_opcodes(mod)

    def test_esch_has_vecload(self):
        mod = generate_esch_module(2, [(0.0, 1.0)] * 2, 10, 500)
        assert "VecLoad" in _collect_opcodes(mod)

    def test_esch_has_vecstore(self):
        mod = generate_esch_module(2, [(0.0, 1.0)] * 2, 10, 500)
        assert "VecStore" in _collect_opcodes(mod)

    def test_esch_has_callbounded(self):
        mod = generate_esch_module(2, [(0.0, 1.0)] * 2, 10, 500)
        assert "CallBounded" in _collect_opcodes(mod)

    def test_crs2lm_has_vecload(self):
        mod = generate_crs2lm_module(2, [(0.0, 1.0)] * 2, 4)
        assert "VecLoad" in _collect_opcodes(mod)

    def test_crs2lm_has_vecstore(self):
        mod = generate_crs2lm_module(2, [(0.0, 1.0)] * 2, 4)
        assert "VecStore" in _collect_opcodes(mod)

    def test_crs2lm_has_rangecheck(self):
        mod = generate_crs2lm_module(2, [(0.0, 1.0)] * 2, 4)
        ops = _collect_opcodes(mod)
        assert "RangeCheck" in ops or "VecRangeCheck" in ops

    def test_crs2lm_has_vecreduce(self):
        mod = generate_crs2lm_module(2, [(0.0, 1.0)] * 2, 4)
        assert "VecReduce" in _collect_opcodes(mod)

    def test_crs2lm_has_callbounded(self):
        mod = generate_crs2lm_module(2, [(0.0, 1.0)] * 2, 4)
        assert "CallBounded" in _collect_opcodes(mod)


# ── Proof certificate structure ────────────────────────────


class TestProofCertificate:
    def test_direct_has_validate_hashcommit_seal(self):
        mod = generate_direct_module(2, [(0.0, 1.0)] * 2, 100, 1e-3)
        assert _has_opcode_sequence(mod, "Validate", "HashCommit", "Seal")

    def test_esch_has_validate_hashcommit_seal(self):
        mod = generate_esch_module(2, [(0.0, 1.0)] * 2, 10, 500)
        assert _has_opcode_sequence(mod, "Validate", "HashCommit", "Seal")

    def test_crs2lm_has_validate_hashcommit_seal(self):
        mod = generate_crs2lm_module(2, [(0.0, 1.0)] * 2, 4)
        assert _has_opcode_sequence(mod, "Validate", "HashCommit", "Seal")

    def test_direct_proof_before_halt(self):
        mod = generate_direct_module(2, [(0.0, 1.0)] * 2, 100, 1e-3)
        ops = _collect_opcodes(mod)
        seal_idx = ops.index("Seal")
        halt_idx = ops.index("Halt")
        assert seal_idx < halt_idx

    def test_esch_proof_before_halt(self):
        mod = generate_esch_module(2, [(0.0, 1.0)] * 2, 10, 500)
        ops = _collect_opcodes(mod)
        seal_idx = ops.index("Seal")
        halt_idx = ops.index("Halt")
        assert seal_idx < halt_idx


# ── Fixed-point scaling ──────────────────────────────────


class TestFixedPointScaling:
    def test_direct_has_scale_factor_in_metadata(self):
        mod = generate_direct_module(2, [(0.0, 100.0)] * 2, 100, 1e-3)
        assert "scale_factor" in mod.metadata
        assert mod.metadata["scale_factor"] > 0

    def test_direct_has_resolution(self):
        mod = generate_direct_module(2, [(0.0, 100.0)] * 2, 100, 1e-3)
        assert "resolution" in mod.metadata
        assert mod.metadata["resolution"] > 0

    def test_esch_has_scale_factor(self):
        mod = generate_esch_module(2, [(0.0, 100.0)] * 2, 10, 500)
        assert "scale_factor" in mod.metadata
        assert mod.metadata["scale_factor"] > 0

    def test_crs2lm_has_scale_factor(self):
        mod = generate_crs2lm_module(2, [(0.0, 100.0)] * 2, 4)
        assert "scale_factor" in mod.metadata
        assert mod.metadata["scale_factor"] > 0

    def test_direct_constants_non_empty(self):
        mod = generate_direct_module(2, [(0.0, 100.0)] * 2, 100, 1e-3)
        assert len(mod.constants) > 0
        # Constants should encode finite values
        assert all(isinstance(c, int) for c in mod.constants)


# ── High-level codegen class ───────────────────────────────


class TestFLUXOptimizerCodegen:
    def test_init_defaults(self):
        codegen = FLUXOptimizerCodegen()
        assert codegen.frac_bits == 16

    def test_custom_frac_bits(self):
        codegen = FLUXOptimizerCodegen(frac_bits=8)
        assert codegen.frac_bits == 8

    def test_direct_facade(self):
        codegen = FLUXOptimizerCodegen()
        mod = codegen.direct(3, [(0.0, 1.0)] * 3, 500, 1e-4)
        assert mod.metadata["algorithm"] == "DIRECT"
        assert mod.metadata["dim"] == 3

    def test_esch_facade(self):
        codegen = FLUXOptimizerCodegen()
        mod = codegen.esch(4, [(-5.0, 5.0)] * 4, 30, 2000)
        assert mod.metadata["algorithm"] == "ESCH"
        assert mod.metadata["pop_size"] == 30

    def test_crs2lm_facade(self):
        codegen = FLUXOptimizerCodegen()
        mod = codegen.crs2lm(2, [(0.0, 1.0)] * 2, 6)
        assert mod.metadata["algorithm"] == "CRS2-LM"
        assert mod.metadata["simplex_size"] >= 4

    def test_list_algorithms(self):
        codegen = FLUXOptimizerCodegen()
        algos = codegen.list_algorithms()
        assert "DIRECT" in algos
        assert "ESCH" in algos
        assert "CRS2-LM" in algos

    def test_facade_reuses_frac_bits(self):
        codegen = FLUXOptimizerCodegen(frac_bits=12)
        mod = codegen.direct(2, [(0.0, 1.0)] * 2, 100, 1e-3)
        assert mod.constraints[0].params["frac_bits"] == 12


# ── Error handling ─────────────────────────────────────────


class TestErrorCases:
    def test_direct_bad_bounds_length(self):
        with pytest.raises(ValueError, match="bounds length"):
            generate_direct_module(
                dim=3, bounds=[(0.0, 1.0)] * 2, maxeval=100, ftol=1e-3
            )

    def test_esch_bad_bounds_length(self):
        with pytest.raises(ValueError, match="bounds length"):
            generate_esch_module(
                dim=3, bounds=[(0.0, 1.0)] * 2, pop_size=10, maxeval=500
            )

    def test_crs2lm_bad_bounds_length(self):
        with pytest.raises(ValueError, match="bounds length"):
            generate_crs2lm_module(dim=3, bounds=[(0.0, 1.0)] * 2, pop_size=6)

    def test_direct_empty_bounds(self):
        with pytest.raises(ValueError):
            generate_direct_module(dim=0, bounds=[], maxeval=100, ftol=1e-3)


# ── Constraint metadata ────────────────────────────────────


class TestConstraintMetadata:
    def test_direct_constraint_def(self):
        mod = generate_direct_module(2, [(0.0, 1.0)] * 2, 100, 1e-3)
        assert len(mod.constraints) >= 1
        c = mod.constraints[0]
        assert c.kind == "custom"
        assert c.params["algorithm"] == "DIRECT"
        assert c.params["maxeval"] == 100

    def test_esch_constraint_def(self):
        mod = generate_esch_module(2, [(0.0, 1.0)] * 2, 10, 500)
        assert len(mod.constraints) >= 1
        c = mod.constraints[0]
        assert c.params["algorithm"] == "ESCH"
        assert c.params["pop_size"] == 10

    def test_crs2lm_constraint_def(self):
        mod = generate_crs2lm_module(2, [(0.0, 1.0)] * 2, 4)
        assert len(mod.constraints) >= 1
        c = mod.constraints[0]
        assert c.params["algorithm"] == "CRS2-LM"


# ── Disassembly sanity ─────────────────────────────────────


class TestDisassembly:
    def test_direct_disasm_contains_algorithm(self):
        mod = generate_direct_module(2, [(0.0, 1.0)] * 2, 100, 1e-3)
        asm = mod.disasm()
        assert "FLUX v3" in asm
        assert "instructions:" in asm

    def test_esch_disasm_non_empty(self):
        mod = generate_esch_module(2, [(0.0, 1.0)] * 2, 10, 500)
        asm = mod.disasm()
        lines = asm.strip().split("\n")
        assert len(lines) > 3

    def test_crs2lm_disasm_has_expected_ops(self):
        mod = generate_crs2lm_module(2, [(0.0, 1.0)] * 2, 4)
        asm = mod.disasm()
        assert "VecLoad" in asm
        assert "VecReduce" in asm
        assert "Seal" in asm


# ── Warnings ───────────────────────────────────────────────


class TestWarnings:
    def test_crs2lm_warns_on_large_simplex(self):
        mod = generate_crs2lm_module(dim=2, bounds=[(0.0, 1.0)] * 2, pop_size=20)
        # pop_size=20 > 8 triggers truncation warning
        assert any("simplex_size" in w for w in mod.warnings)

    def test_direct_no_warnings_for_normal_size(self):
        mod = generate_direct_module(2, [(0.0, 1.0)] * 2, 100, 1e-3)
        assert len(mod.warnings) == 0
