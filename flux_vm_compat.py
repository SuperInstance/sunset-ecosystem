"""Flux VM Compat — Redirect shim for legacy v2 → v3 migration.

Per SPEC-FLUX-RESOLUTION §5:
  Importing this module triggers an automatic redirect so that
  code expecting `flux_vm_v2` transparently gets `flux_vm_v3`.

  Example:
      from flux_vm_compat import FluxVM  # → loads v3 under the hood
"""
from __future__ import annotations

import warnings

# Emit a single deprecation notice on first import
warnings.warn(
    "flux_vm_compat is a compatibility shim. "
    "Prefer direct imports from flux_vm_v3 or flux_compat.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the canonical v3 interface
try:
    from flux_compat.compat import load_v2
    from flux_compat.v3_module import Instruction, ConstraintDef, Module
except ImportError as _e:
    raise ImportError(
        "flux_vm_compat requires flux_compat to be installed. "
        "Ensure flux_compat/ is on PYTHONPATH."
    ) from _e

__all__ = [
    "load_v2",
    "Instruction",
    "ConstraintDef",
    "Module",
]
