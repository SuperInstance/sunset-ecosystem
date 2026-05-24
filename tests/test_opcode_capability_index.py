"""Tests for the FLUX Opcode Capability Index.

Ensures every opcode from flux-vm-v3-temp/src/opcode.rs is registered,
categories are correct, status queries work, and the gap report is
well-formed.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from logos.opcode_capability_index import (
    FluxOpcode,
    OpcodeCapabilityIndex,
    OpcodeStatus,
    _OPCODES,
)


# ── fixture ───────────────────────────────────────────────────

@pytest.fixture
def index() -> OpcodeCapabilityIndex:
    return OpcodeCapabilityIndex()


# ── 1. coverage ───────────────────────────────────────────────

def test_all_58_opcodes_registered(index: OpcodeCapabilityIndex) -> None:
    """Every opcode from the Rust source must be in the index."""
    assert index.total_opcodes == 58


def test_can_lookup_by_name(index: OpcodeCapabilityIndex) -> None:
    assert index.get("Add") is not None
    assert index.get("Halt") is not None
    assert index.get("StreamClose") is not None


def test_can_lookup_by_number(index: OpcodeCapabilityIndex) -> None:
    assert index.get(0x09) is not None  # Add
    assert index.get(0x29) is not None  # Halt
    assert index.get(0x3a) is not None  # StreamClose


def test_unknown_opcode_returns_none(index: OpcodeCapabilityIndex) -> None:
    assert index.get("FakeOpcode") is None
    assert index.get(0xFF) is None


# ── 2. category counts ────────────────────────────────────────

def test_category_counts_match_rust_source(index: OpcodeCapabilityIndex) -> None:
    counts = index.count_by_category()
    assert counts["stack"] == 8
    assert counts["arithmetic"] == 8
    assert counts["memory"] == 8  # 4 register + 4 vector (load/store)
    assert counts["control"] == 14  # 6 control flow + 6 constraint + 2 vector control
    assert counts["crypto"] == 8  # 4 constraint-crypto + 4 provenance
    assert counts["io"] == 12  # 4 effects + 4 parallel + 4 streaming


def test_category_names_are_stable(index: OpcodeCapabilityIndex) -> None:
    counts = index.count_by_category()
    expected = {"stack", "arithmetic", "memory", "control", "crypto", "io"}
    assert set(counts.keys()) == expected


# ── 3. status queries ─────────────────────────────────────────

def test_can_use_from_python_returns_false_for_rust_only(index: OpcodeCapabilityIndex) -> None:
    rust_only = ["Prove", "SnapVerify", "ParDispatch", "VecLoad", "StreamOpen", "LoadReg"]
    for name in rust_only:
        assert index.can_use_from_python(name) is False, f"{name} should be RUST_ONLY"


def test_can_use_from_python_returns_true_for_safe(index: OpcodeCapabilityIndex) -> None:
    safe = ["Add", "Sub", "Push", "Pop", "Halt", "Nop", "RangeCheck", "Validate", "EmitEvent"]
    for name in safe:
        assert index.can_use_from_python(name) is True, f"{name} should be PYTHON_SAFE"


def test_get_safe_opcodes_returns_only_safe(index: OpcodeCapabilityIndex) -> None:
    safe = index.get_safe_opcodes()
    for op in safe:
        assert op.status == OpcodeStatus.PYTHON_SAFE or op.name in index._overrides


def test_get_safe_opcodes_filters_by_category(index: OpcodeCapabilityIndex) -> None:
    safe_arithmetic = index.get_safe_opcodes(category="arithmetic")
    for op in safe_arithmetic:
        assert op.category == "arithmetic"
        assert index.can_use_from_python(op.name)


def test_get_rust_only_opcodes_filters_by_category(index: OpcodeCapabilityIndex) -> None:
    rust_io = index.get_rust_only_opcodes(category="io")
    for op in rust_io:
        assert op.category == "io"
        assert not index.can_use_from_python(op.name)


# ── 4. gap report ─────────────────────────────────────────────

def test_gap_report_includes_effort_estimates(index: OpcodeCapabilityIndex) -> None:
    gaps = index.get_gap_report()
    assert len(gaps) > 0
    for entry in gaps:
        assert "effort_estimate" in entry
        assert entry["effort_estimate"] in ("trivial", "low", "medium", "high", "blocked")


def test_gap_report_covers_all_rust_only(index: OpcodeCapabilityIndex) -> None:
    rust_only = index.get_rust_only_opcodes()
    gaps = index.get_gap_report()
    gap_names = {g["name"] for g in gaps}
    for op in rust_only:
        assert op.name in gap_names


def test_gap_report_reason_field_is_present(index: OpcodeCapabilityIndex) -> None:
    for entry in index.get_gap_report():
        assert "reason" in entry
        assert isinstance(entry["reason"], str)


# ── 5. path_a equivalents ─────────────────────────────────────

def test_suggest_path_a_maps_to_real_functions(index: OpcodeCapabilityIndex) -> None:
    """Mapped equivalents must be real method names on PythonFluxFallback."""
    valid_methods = {"check_candidate", "check_batch", "score_for_breeding", "record_violation"}
    for op in _OPCODES:
        equiv = index.suggest_path_a_equivalent(op.name)
        if equiv is not None:
            assert equiv in valid_methods, f"{op.name} maps to unknown method {equiv}"


def test_suggest_path_a_returns_none_for_rust_only(index: OpcodeCapabilityIndex) -> None:
    """Most RUST_ONLY opcodes have no Python fallback."""
    no_fallback = ["Prove", "VecLoad", "ParDispatch", "SnapHash", "StreamOpen"]
    for name in no_fallback:
        assert index.suggest_path_a_equivalent(name) is None


def test_suggest_path_a_by_number(index: OpcodeCapabilityIndex) -> None:
    assert index.suggest_path_a_equivalent(0x15) == "check_candidate"  # RangeCheck
    assert index.suggest_path_a_equivalent(0x2c) == "record_violation"  # EmitEvent


# ── 6. status overrides ───────────────────────────────────────

def test_update_status_changes_effective_status(index: OpcodeCapabilityIndex) -> None:
    index.update_status("Prove", OpcodeStatus.PYTHON_SAFE)
    assert index.can_use_from_python("Prove") is True


def test_reset_status_reverts_to_canonical(index: OpcodeCapabilityIndex) -> None:
    index.update_status("Prove", OpcodeStatus.PYTHON_SAFE)
    index.reset_status("Prove")
    assert index.can_use_from_python("Prove") is False


def test_update_status_unknown_opcode_raises(index: OpcodeCapabilityIndex) -> None:
    with pytest.raises(KeyError):
        index.update_status("FakeOpcode", OpcodeStatus.PYTHON_SAFE)


def test_count_by_status_reflects_overrides(index: OpcodeCapabilityIndex) -> None:
    before = index.count_by_status()
    index.update_status("Prove", OpcodeStatus.PYTHON_SAFE)
    index.update_status("VecLoad", OpcodeStatus.PYTHON_SAFE)
    after = index.count_by_status()
    assert after["PYTHON_SAFE"] == before["PYTHON_SAFE"] + 2
    assert after["RUST_ONLY"] == before["RUST_ONLY"] - 2


# ── 7. persistence ────────────────────────────────────────────

def test_save_and_roundtrip(index: OpcodeCapabilityIndex) -> None:
    index.update_status("Prove", OpcodeStatus.PYTHON_SAFE)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        index.save(path)
        loaded = OpcodeCapabilityIndex.load(path)
        assert loaded.total_opcodes == index.total_opcodes
        assert loaded.can_use_from_python("Prove") is True
        assert loaded.can_use_from_python("VecLoad") is False
    finally:
        Path(path).unlink()


def test_save_format_is_json(index: OpcodeCapabilityIndex) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        index.save(path)
        data = json.loads(Path(path).read_text())
        assert data["version"] == 1
        assert data["total_opcodes"] == 58
        assert len(data["canonical"]) == 58
    finally:
        Path(path).unlink()


# ── 8. edge cases ─────────────────────────────────────────────

def test_repr_is_human_readable(index: OpcodeCapabilityIndex) -> None:
    r = repr(index)
    assert "OpcodeCapabilityIndex(" in r
    assert "total=58" in r


def test_frozen_opcode_dataclass_is_hashable(index: OpcodeCapabilityIndex) -> None:
    op = index.get("Add")
    assert op is not None
    # frozen dataclasses are not auto-hashable unless eq=True and frozen=True
    # which they are — so this should work in a set
    s = {op}
    assert len(s) == 1


def test_div_opcode_has_overflow_guard_description(index: OpcodeCapabilityIndex) -> None:
    div = index.get("Div")
    assert div is not None
    assert "i32::MIN" in div.description or "guard" in div.description.lower()


def test_abs_opcode_has_overflow_guard_description(index: OpcodeCapabilityIndex) -> None:
    abs_op = index.get("Abs")
    assert abs_op is not None
    assert "i32::MIN" in abs_op.description or "guard" in abs_op.description.lower()


def test_call_bounded_has_default_note(index: OpcodeCapabilityIndex) -> None:
    cb = index.get("CallBounded")
    assert cb is not None
    assert "4096" in cb.description or "bound" in cb.description.lower()


# ── 9. batch sanity ───────────────────────────────────────────

def test_all_opcodes_have_unique_numbers(index: OpcodeCapabilityIndex) -> None:
    numbers = [op.opcode_number for op in index._by_name.values()]
    assert len(numbers) == len(set(numbers))


def test_all_opcodes_in_valid_range(index: OpcodeCapabilityIndex) -> None:
    for op in index._by_name.values():
        assert 0x01 <= op.opcode_number <= 0x3a


def test_no_deprecated_opcodes_by_default(index: OpcodeCapabilityIndex) -> None:
    counts = index.count_by_status()
    assert counts.get("DEPRECATED", 0) == 0


def test_no_untested_opcodes_by_default(index: OpcodeCapabilityIndex) -> None:
    """Every opcode in the canonical list has an explicit status."""
    counts = index.count_by_status()
    assert counts.get("UNTESTED", 0) == 0
