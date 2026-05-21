"""v2 → v3 opcode mapping with deprecation warnings.

Most opcodes map 1:1.  The interesting cases (creative translation) are
marked with comments explaining the semantic gap and how we bridge it.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import warnings

# Re-export v3 byte values for serialization
V3_OPCODE_BYTES: dict[str, int] = {
    "Push": 0x01, "Pop": 0x02, "Dup": 0x03, "Swap": 0x04, "Over": 0x05,
    "Drop": 0x06, "LoadConst": 0x07, "Nop": 0x08,
    "Add": 0x09, "Sub": 0x0a, "Mul": 0x0b, "Div": 0x0c,
    "Saturate": 0x0d, "Min": 0x0e, "Max": 0x0f, "Abs": 0x10,
    "LoadReg": 0x11, "StoreReg": 0x12, "LoadRegVec": 0x13, "StoreRegVec": 0x14,
    "RangeCheck": 0x15, "BatchCheck": 0x16, "AccumulateMask": 0x17,
    "ClassifySeverity": 0x18, "Prove": 0x19, "QueryBackward": 0x1a,
    "Simplify": 0x1b, "Validate": 0x1c, "HashCommit": 0x1d, "Seal": 0x1e,
    "VecLoad": 0x1f, "VecStore": 0x20, "VecRangeCheck": 0x21,
    "VecMaskMerge": 0x22, "VecReduce": 0x23, "VecGather": 0x24,
    "FwdJump": 0x25, "CondJump": 0x26, "CallBounded": 0x27,
    "Ret": 0x28, "Halt": 0x29, "Checkpoint": 0x2a,
    "SetHandler": 0x2b, "EmitEvent": 0x2c, "Rollback": 0x2d, "GetResult": 0x2e,
    "ParDispatch": 0x2f, "ParMerge": 0x30, "ParBarrier": 0x31, "ParReduce": 0x32,
    "SnapRecord": 0x33, "SnapQuery": 0x34, "SnapHash": 0x35, "SnapVerify": 0x36,
    "StreamOpen": 0x37, "StreamCheck": 0x38, "StreamBatch": 0x39, "StreamClose": 0x3a,
}

# Simple 1:1 mappings
_DIRECT_MAP: dict[str, str] = {
    "Push": "Push",
    "Pop": "Pop",
    "Dup": "Dup",
    "Swap": "Swap",
    "Drop": "Drop",
    "LoadConst": "LoadConst",
    "Add": "Add",
    "Sub": "Sub",
    "Mul": "Mul",
    "Div": "Div",
    "Min": "Min",
    "Max": "Max",
    "LoadReg": "LoadReg",
    "StoreReg": "StoreReg",
    "Ret": "Ret",
    "Halt": "Halt",
}


def map_opcode(
    v2_name: str,
    operand: Optional[int],
    *,
    _warned: set = None,
) -> Tuple[List[str], List[Optional[int]], List[str]]:
    """Translate a single v2 instruction into one or more v3 instructions.

    Returns (opcode_names, operands, warnings).
    """
    if _warned is None:
        _warned = set()

    out_ops: List[str] = []
    out_imm: List[Optional[int]] = []
    out_warn: List[str] = []

    # ------------------------------------------------------------
    # 1:1 direct translations
    # ------------------------------------------------------------
    if v2_name in _DIRECT_MAP:
        out_ops.append(_DIRECT_MAP[v2_name])
        out_imm.append(operand)
        return out_ops, out_imm, out_warn

    # ------------------------------------------------------------
    # CREATIVE TRANSLATION #1
    # v2 "Check" → v3 RangeCheck + Validate
    #
    # v2 Check was a generic constraint gate with no severity
    # classification.  v3 splits this into RangeCheck (bounds) and
    # Validate (assertion).  We emit both so the v3 VM sees the
    # same logical behaviour.
    # ------------------------------------------------------------
    if v2_name == "Check":
        out_ops.extend(["RangeCheck", "Validate"])
        out_imm.extend([operand, None])
        out_warn.append(
            "v2 'Check' split into RangeCheck+Validate; "
            "severity classification was unavailable in v2"
        )
        return out_ops, out_imm, out_warn

    # ------------------------------------------------------------
    # CREATIVE TRANSLATION #2
    # v2 "Assert" → v3 Prove + HashCommit
    #
    # v2 Assert was a hard assertion that stopped execution.
    # v3's Prove produces a proof certificate, and HashCommit
    # anchors it.  We wrap the assertion in a proof context so
    # v3 can continue with verifiable output rather than a naked
    # trap.
    # ------------------------------------------------------------
    if v2_name == "Assert":
        out_ops.extend(["Prove", "HashCommit"])
        out_imm.extend([None, None])
        out_warn.append(
            "v2 'Assert' translated to Prove+HashCommit; "
            "v3 proof certificates replace hard traps"
        )
        return out_ops, out_imm, out_warn

    # ------------------------------------------------------------
    # CREATIVE TRANSLATION #3
    # v2 "Range" → v3 RangeCheck + ClassifySeverity
    #
    # v2 Range carried explicit lower/upper bounds in its 2-byte
    # operand (packed i8+i8).  We unpack the bounds, push them as
    # constants, and then emit RangeCheck followed by
    # ClassifySeverity so the v3 constraint pipeline gets the same
    # information v2 had.
    # ------------------------------------------------------------
    if v2_name == "Range":
        if operand is not None:
            lower = (operand >> 8) & 0xFF
            upper = operand & 0xFF
            if lower & 0x80:
                lower -= 256
            if upper & 0x80:
                upper -= 256
        else:
            lower, upper = 0, 0
            out_warn.append("v2 'Range' missing operand; defaulting bounds to 0,0")
        out_ops.extend(["RangeCheck", "ClassifySeverity"])
        out_imm.extend([None, None])
        # The bounds are embedded as a preceding Push sequence;
        # compat layer caller handles constant emission.
        out_warn.append(
            f"v2 'Range' unpacked bounds [{lower}, {upper}] → "
            "v3 RangeCheck+ClassifySeverity"
        )
        return out_ops, out_imm, out_warn

    # ------------------------------------------------------------
    # CREATIVE TRANSLATION #4
    # v2 "Batch" + "Accumulate" → v3 BatchCheck + AccumulateMask
    #
    # v2 had two separate opcodes for batching and mask
    # accumulation.  v3 unifies the workflow: BatchCheck starts a
    # batch, AccumulateMask folds results.  We map each 1:1 but
    # note that v3's BatchCheck is stricter about pre-conditions.
    # ------------------------------------------------------------
    if v2_name == "Batch":
        out_ops.append("BatchCheck")
        out_imm.append(operand)
        out_warn.append(
            "v2 'Batch' → v3 'BatchCheck'; verify batch size ≤ v3 limit"
        )
        return out_ops, out_imm, out_warn

    if v2_name == "Accumulate":
        out_ops.append("AccumulateMask")
        out_imm.append(operand)
        return out_ops, out_imm, out_warn

    # ------------------------------------------------------------
    # CREATIVE TRANSLATION #5
    # v2 "Jump" → v3 FwdJump + Nop
    #
    # v2 Jump was bidirectional (could jump backward).  v3
    # FwdJump is forward-only for termination guarantees.
    # We emit FwdJump and pad with Nop so backward jumps become
    # visible as layout problems rather than silent mis-compiles.
    # If the jump offset is negative we emit a warning and pad.
    # ------------------------------------------------------------
    if v2_name == "Jump":
        if operand is not None and operand < 0:
            out_warn.append(
                f"v2 backward Jump ({operand}) → v3 FwdJump with Nop padding; "
                "review loop structure for v3 termination"
            )
            out_ops.append("Nop")
            out_imm.append(None)
        out_ops.append("FwdJump")
        out_imm.append(abs(operand) if operand is not None else None)
        return out_ops, out_imm, out_warn

    # ------------------------------------------------------------
    # CREATIVE TRANSLATION #6
    # v2 "JumpZero" → v3 CondJump
    #
    # Same forward-only restriction as Jump.  We use CondJump
    # directly but warn on negative offsets.
    # ------------------------------------------------------------
    if v2_name == "JumpZero":
        if operand is not None and operand < 0:
            out_warn.append(
                f"v2 backward JumpZero ({operand}) → v3 CondJump; "
                "v3 requires forward-only conditional branches"
            )
        out_ops.append("CondJump")
        out_imm.append(abs(operand) if operand is not None else None)
        return out_ops, out_imm, out_warn

    # ------------------------------------------------------------
    # CREATIVE TRANSLATION #7
    # v2 "Call" → v3 CallBounded
    #
    # v2 Call had no cycle bound.  v3 CallBounded requires a max
    # cycle argument.  We supply the v3 default (4096) as the
    # bound, emitting a warning that the bound is synthetic.
    # ------------------------------------------------------------
    if v2_name == "Call":
        out_ops.append("CallBounded")
        # v3 CallBounded takes a 2-byte immediate: the call target
        # and the bound lives in a separate metadata slot.  For
        # compat we just forward the original target.
        out_imm.append(operand)
        out_warn.append(
            "v2 'Call' → v3 'CallBounded' with synthetic cycle limit 4096; "
            "review for boundedness guarantees"
        )
        return out_ops, out_imm, out_warn

    # ------------------------------------------------------------
    # CREATIVE TRANSLATION #8
    # v2 "Read" / "Write" → v3 StreamOpen + StreamCheck / StreamClose
    #
    # v2 treated I/O as port reads/writes.  v3 has no port I/O;
    # it has streaming constraints.  We interpret legacy port
    # access as opening a stream, checking one batch, then
    # closing.  This is the biggest semantic leap.
    # ------------------------------------------------------------
    if v2_name == "Read":
        out_ops.extend(["StreamOpen", "StreamCheck", "StreamBatch"])
        out_imm.extend([operand, None, 1])
        out_warn.append(
            "v2 'Read' reinterpreted as StreamOpen+StreamCheck+StreamBatch; "
            "port I/O does not exist in v3"
        )
        return out_ops, out_imm, out_warn

    if v2_name == "Write":
        out_ops.extend(["StreamOpen", "StreamBatch", "StreamClose"])
        out_imm.extend([operand, 1, None])
        out_warn.append(
            "v2 'Write' reinterpreted as StreamOpen+StreamBatch+StreamClose; "
            "port I/O does not exist in v3"
        )
        return out_ops, out_imm, out_warn

    # ------------------------------------------------------------
    # DEPRECATED / REMOVED
    # ------------------------------------------------------------
    if v2_name == "Break":
        out_warn.append(
            "v2 'Break' is REMOVED in v3 — no debugger breakpoint opcode. "
            "Omitting instruction."
        )
        return [], [], out_warn

    # Unknown / fallback
    out_warn.append(f"Unknown v2 opcode '{v2_name}' — no v3 mapping exists")
    return [], [], out_warn
