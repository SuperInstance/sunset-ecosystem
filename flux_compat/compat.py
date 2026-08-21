"""FLUX v2 → v3 compat shim.

Implements `load_v2(path) -> v3.Module` per SPEC-FLUX-RESOLUTION.md.
"""

from __future__ import annotations

import os
import warnings
from typing import List

from .v2_bytecode import parse_v2, V2Instruction
from .v3_module import Module, Instruction, ConstraintDef
from .opcode_map import map_opcode


def load_v2(path: str) -> Module:
    """Load a legacy v2 bytecode file and translate it to a v3 Module.

    Parameters
    ----------
    path : str
        Filesystem path to a `.flux` or `.fc` v2 bytecode file.

    Returns
    -------
    Module
        A v3 Module ready for inspection, disassembly, or VM ingestion.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is not a valid v2 bytecode container.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"v2 bytecode not found: {path}")

    v2 = parse_v2(path)

    v3_instructions: List[Instruction] = []
    v3_warnings: List[str] = []
    v3_constraints: List[ConstraintDef] = []

    # Translate constants verbatim — the constant pool format is identical
    v3_constants = list(v2.constants)

    # Translate constraints (simple 1:1 kind mapping)
    for c in v2.constraints:
        v3_constraints.append(
            ConstraintDef(kind=c.kind, params={"raw": c.payload.hex()})
        )

    # Translate instructions opcode-by-opcode
    for idx, inst in enumerate(v2.instructions):
        op_names, op_imms, op_warns = map_opcode(inst.opcode, inst.operand)

        for w in op_warns:
            full = f"[pc={idx:04x} op={inst.opcode!r}] {w}"
            if full not in v3_warnings:
                v3_warnings.append(full)
                warnings.warn(full, DeprecationWarning, stacklevel=2)

        for name, imm in zip(op_names, op_imms):
            v3_instructions.append(Instruction(opcode=name, operand=imm))

    return Module(
        version=3,
        constants=v3_constants,
        instructions=v3_instructions,
        constraints=v3_constraints,
        metadata={
            "translated_from": "v2",
            "v2_version_minor": v2.version_minor,
            "v2_flags": v2.flags,
            "v2_instruction_count": len(v2.instructions),
        },
        warnings=v3_warnings,
    )
