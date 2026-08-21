"""FLUX v2 → v3 compatibility layer.

Public API:
    load_v2(path) -> v3.Module

See docs/SPEC-FLUX-RESOLUTION.md for architecture context.
"""

from .compat import load_v2
from .v3_module import Module, Instruction, ConstraintDef

__all__ = ["load_v2", "Module", "Instruction", "ConstraintDef"]
