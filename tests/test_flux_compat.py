"""Tests for the FLUX v2 → v3 compat layer.

Covers at least 5 creative translation scenarios plus happy-path and
edge-case coverage.
"""

from __future__ import annotations

import os
import struct
import tempfile
import warnings

import pytest

from flux_compat import load_v2, Module
from flux_compat.v2_bytecode import parse_v2
from flux_compat.v3_module import Instruction


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_v2(
    constants: list[int] = None,
    instructions: list[tuple[int, int | None]] = None,
    constraints: list[tuple[int, bytes]] = None,
    flags: int = 0,
) -> str:
    """Serialize a minimal v2 bytecode file to a temp path and return it."""
    constants = constants or []
    instructions = instructions or []
    constraints = constraints or []

    if constraints:
        flags |= 0x01

    buf = b"FLX2"
    buf += struct.pack("<B", 0)  # version minor
    buf += struct.pack("<B", flags)  # flags
    buf += struct.pack("<H", len(constants))
    for c in constants:
        buf += struct.pack("<i", c)
    buf += struct.pack("<H", len(instructions))
    for op, imm in instructions:
        buf += struct.pack("<B", op)
        if imm is not None:
            # Use 2-byte immediate for most; 4-byte for Push/LoadConst
            if op in (0x01, 0x06):
                buf += struct.pack("<i", imm)
            elif op in (0x40, 0x41, 0x42):
                buf += struct.pack("<h", imm)
            elif op == 0x32:  # Range — packed i8+i8
                buf += struct.pack("<h", imm)
            elif op in (0x33, 0x50, 0x51):  # Batch, Read, Write
                buf += struct.pack("<B", imm)
            elif op in (0x20, 0x21):  # LoadReg, StoreReg
                buf += struct.pack("<B", imm)
            else:
                buf += struct.pack("<h", imm)

    if constraints:
        buf += struct.pack("<H", len(constraints))
        for kind, payload in constraints:
            buf += struct.pack("<B", kind)
            buf += struct.pack("<H", len(payload))
            buf += payload

    fd, path = tempfile.mkstemp(suffix=".flux")
    os.write(fd, buf)
    os.close(fd)
    return path


# ------------------------------------------------------------------
# Scenario 1 — 1:1 direct translation (happy path)
# ------------------------------------------------------------------


def test_direct_arithmetic_translation():
    """Simple stack + arithmetic opcodes map 1:1 with no warnings."""
    path = _make_v2(
        constants=[42, 7],
        instructions=[
            (0x01, 42),  # Push 42
            (0x01, 7),  # Push 7
            (0x10, None),  # Add
            (0x02, None),  # Pop
        ],
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = load_v2(path)

    assert mod.version == 3
    assert mod.constants == [42, 7]
    assert len(mod.instructions) == 4
    assert [i.opcode for i in mod.instructions] == ["Push", "Push", "Add", "Pop"]
    assert mod.instructions[0].operand == 42
    assert mod.instructions[1].operand == 7
    # No deprecation warnings for 1:1 mapping
    assert not any(issubclass(x.category, DeprecationWarning) for x in w)
    os.unlink(path)


# ------------------------------------------------------------------
# Scenario 2 — v2 "Check" → v3 RangeCheck + Validate (creative #1)
# ------------------------------------------------------------------


def test_check_to_range_check_plus_validate():
    """v2 Check expands to two v3 instructions and emits a warning."""
    path = _make_v2(
        instructions=[
            (0x01, 100),  # Push 100
            (0x30, None),  # Check
        ],
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = load_v2(path)

    ops = [i.opcode for i in mod.instructions]
    assert ops == ["Push", "RangeCheck", "Validate"]
    assert mod.instructions[1].operand is None
    assert mod.instructions[2].operand is None

    warns = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("Check' split into RangeCheck+Validate" in m for m in warns)
    os.unlink(path)


# ------------------------------------------------------------------
# Scenario 3 — v2 "Assert" → v3 Prove + HashCommit (creative #2)
# ------------------------------------------------------------------


def test_assert_to_prove_plus_hashcommit():
    """v2 Assert (hard trap) becomes verifiable proof sequence in v3."""
    path = _make_v2(
        instructions=[
            (0x01, 0),  # Push 0
            (0x31, None),  # Assert
        ],
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = load_v2(path)

    ops = [i.opcode for i in mod.instructions]
    assert ops == ["Push", "Prove", "HashCommit"]
    warns = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("proof certificates replace hard traps" in m for m in warns)
    os.unlink(path)


# ------------------------------------------------------------------
# Scenario 4 — v2 "Range" bounds unpacking (creative #3)
# ------------------------------------------------------------------


def test_range_bounds_unpacking():
    """v2 Range packs two i8 bounds into a 2-byte operand.
    We unpack them and map to RangeCheck + ClassifySeverity."""
    # Pack lower=-10 (0xF6) and upper=50 (0x32) into a signed 16-bit word
    packed = ((-10 & 0xFF) << 8) | (50 & 0xFF)
    # struct.pack '<h' writes little-endian so we need to think carefully:
    # We want bytes [0xF6, 0x32] as the 2-byte immediate.
    # In little-endian 0x32F6 = -13,226 which is not what we want.
    # The v2 parser reads the 2 bytes as a little-endian h.
    # So we need the little-endian int16 whose bytes are [0xF6, 0x32].
    # That's 0x32F6 = 13042 unsigned, but signed it's -13,226.
    # Let's instead encode with struct and verify round-trip.
    raw = struct.pack("<bb", -10, 50)  # little-endian two bytes
    operand = struct.unpack("<h", raw)[0]

    path = _make_v2(
        instructions=[
            (0x01, 25),  # Push 25 (value to range-check)
            (0x32, operand),  # Range
        ],
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = load_v2(path)

    ops = [i.opcode for i in mod.instructions]
    assert "RangeCheck" in ops
    assert "ClassifySeverity" in ops
    warns = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("unpacked bounds" in m for m in warns)
    os.unlink(path)


# ------------------------------------------------------------------
# Scenario 5 — v2 "Call" → v3 CallBounded with synthetic limit (creative #7)
# ------------------------------------------------------------------


def test_call_to_call_bounded():
    """v2 unbounded Call gets a synthetic 4096 cycle limit in v3."""
    path = _make_v2(
        instructions=[
            (0x42, 0x0100),  # Call target=256
        ],
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = load_v2(path)

    assert len(mod.instructions) == 1
    assert mod.instructions[0].opcode == "CallBounded"
    assert mod.instructions[0].operand == 0x0100

    warns = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("synthetic cycle limit 4096" in m for m in warns)
    os.unlink(path)


# ------------------------------------------------------------------
# Scenario 6 — v2 backward Jump → v3 FwdJump + Nop pad (creative #5)
# ------------------------------------------------------------------


def test_backward_jump_forward_only():
    """v2 backward jumps are rejected by v3's forward-only FwdJump.
    Compat layer pads with Nop and warns."""
    path = _make_v2(
        instructions=[
            (0x40, -4),  # Jump backward 4 (loop head)
        ],
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = load_v2(path)

    ops = [i.opcode for i in mod.instructions]
    # Should get Nop padding then FwdJump with absolute offset
    assert ops[-1] == "FwdJump"
    warns = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("backward Jump" in m for m in warns)
    assert any("review loop structure for v3 termination" in m for m in warns)
    os.unlink(path)


# ------------------------------------------------------------------
# Scenario 7 — v2 I/O Read/Write → v3 Streaming (creative #8)
# ------------------------------------------------------------------


def test_read_to_stream_sequence():
    """v2 port Read becomes StreamOpen+StreamCheck+StreamBatch."""
    path = _make_v2(
        instructions=[
            (0x50, 3),  # Read port 3
        ],
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = load_v2(path)

    ops = [i.opcode for i in mod.instructions]
    assert ops == ["StreamOpen", "StreamCheck", "StreamBatch"]
    assert mod.instructions[0].operand == 3
    warns = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("port I/O does not exist in v3" in m for m in warns)
    os.unlink(path)


def test_write_to_stream_sequence():
    """v2 port Write becomes StreamOpen+StreamBatch+StreamClose."""
    path = _make_v2(
        instructions=[
            (0x51, 1),  # Write port 1
        ],
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = load_v2(path)

    ops = [i.opcode for i in mod.instructions]
    assert ops == ["StreamOpen", "StreamBatch", "StreamClose"]
    warns = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("port I/O does not exist in v3" in m for m in warns)
    os.unlink(path)


# ------------------------------------------------------------------
# Scenario 8 — Deprecated opcode removal: Break
# ------------------------------------------------------------------


def test_break_removed():
    """v2 Break has no v3 equivalent; instruction is dropped with warning."""
    path = _make_v2(
        instructions=[
            (0x01, 1),  # Push 1
            (0x60, None),  # Break
            (0x02, None),  # Pop
        ],
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = load_v2(path)

    ops = [i.opcode for i in mod.instructions]
    assert "Break" not in ops
    assert ops == ["Push", "Pop"]
    warns = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("REMOVED" in m and "Break" in m for m in warns)
    os.unlink(path)


# ------------------------------------------------------------------
# Scenario 9 — Constraint preservation round-trip
# ------------------------------------------------------------------


def test_constraints_round_trip():
    """v2 constraint payloads are carried forward into v3 ConstraintDefs."""
    path = _make_v2(
        instructions=[(0x44, None)],  # Halt
        constraints=[
            (0x01, b"alt_min=1000;alt_max=40000"),  # aviation
            (0x02, b"temp_min=-40;temp_max=85"),  # temperature
        ],
        flags=0x01,
    )
    mod = load_v2(path)

    assert len(mod.constraints) == 2
    assert mod.constraints[0].kind == "aviation"
    assert mod.constraints[1].kind == "temperature"
    raw_back = bytes.fromhex(mod.constraints[0].params["raw"])
    assert b"alt_min" in raw_back
    os.unlink(path)


# ------------------------------------------------------------------
# Scenario 10 — Module serialization produces valid v3 header
# ------------------------------------------------------------------


def test_module_to_bytecode_header():
    """Module.to_bytecode() emits a valid v3 header."""
    path = _make_v2(
        constants=[1, 2, 3],
        instructions=[
            (0x01, 1),  # Push 1
            (0x02, None),  # Pop
        ],
    )
    mod = load_v2(path)
    blob = mod.to_bytecode()
    assert blob[:4] == b"FLX3"
    version, const_count = struct.unpack("<HH", blob[4:8])
    assert version == 3
    assert const_count == 3
    os.unlink(path)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_unknown_opcode_raises():
    """Parser rejects truly unknown v2 opcodes."""
    path = _make_v2(instructions=[(0xFF, None)])
    with pytest.raises(ValueError, match="Unknown v2 opcode"):
        parse_v2(path)
    os.unlink(path)


def test_bad_magic_raises():
    """Parser rejects files without FLX2 magic."""
    fd, path = tempfile.mkstemp(suffix=".flux")
    os.write(fd, b"DEAD")
    os.close(fd)
    with pytest.raises(ValueError, match="Bad v2 magic"):
        parse_v2(path)
    os.unlink(path)


def test_disassembly_output():
    """Module.disasm() produces human-readable text."""
    path = _make_v2(
        instructions=[(0x01, 99), (0x10, None)],
    )
    mod = load_v2(path)
    text = mod.disasm()
    assert "FLUX v3 module" in text
    assert "Push 99" in text
    assert "Add" in text
    os.unlink(path)


def test_metadata_populated():
    """load_v2 populates metadata with provenance."""
    path = _make_v2(
        instructions=[(0x44, None)],
        # flags=0: no constraints, no debug — let _make_v2 handle flags
    )
    mod = load_v2(path)
    assert mod.metadata["translated_from"] == "v2"
    assert mod.metadata["v2_flags"] == 0x00
    os.unlink(path)
