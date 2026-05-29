---
title: "FLUX VM Integration — Path A vs Path B Decision"
category: architecture
type: knowledge-track
date: 2026-05-25
affected_files:
  - docs/FLUX_OPCODE_ALIGNMENT.md
  - swarm/flux_compiler.py
  - sunset/flux_vm_bridge.py
  - swarm/flux_vm_gating.py
---

# FLUX VM Integration — Path A vs Path B Decision

## Context

The FLUX VM (Rust, 60 opcodes) was built as a full virtual machine with proof-carrying, checkpoint/rollback, streaming, and parallel dispatch features. However, the Python integration only used a single FFI function (`flux_check_batch()`), treating the Rust side as a constraint library rather than a VM.

This created an architectural disconnect: the VM's advanced features were completely inaccessible from Python.

## Decision

**Chose Path B (Full VM Integration) with hybrid fallback.**

- Path A (Library): Accept FLUX as constraint library, keep `flux_check_batch()`
- Path B (Full VM): Build Python→FLUX bytecode compiler, wire VM lifecycle, enable proofs/checkpoints/streaming

**Rationale:**
- Path B unlocks verifiable proof certificates (SHA-256) for every breeding decision
- Path B enables checkpoint/rollback for fault recovery
- Path B allows streaming constraint checks for large batches
- The fleet already has 80% of Path B built (compiler, bridge, runner); only the AST frontend is missing
- Python fallback (`FluxVMRunner`) provides full functionality until Rust FFI is complete

## Implementation

### What Already Exists

| Component | Status | Location |
|-----------|--------|----------|
| Bytecode compiler | ✅ | `swarm/flux_compiler.py` |
| Python fallback VM | ✅ | `swarm/flux_vm_runner.py` |
| Rust FFI bridge | ✅ | `sunset/flux_vm_bridge.py` |
| VM gating checker | ✅ | `swarm/flux_vm_gating.py` |
| Rust VM (libflux_vm_v3.so) | ✅ | `flux_vm/libflux_vm_v3.so` |

### What's Missing

| Component | Effort | Status |
|-----------|--------|--------|
| Python AST frontend | ~3 hours | `sunset/flux_ast_compiler.py` (design complete) |
| Rust FFI: `flux_vm_load_const_pool()` | Blocked on FM | Needs cargo rebuild |
| Rust FFI: `flux_vm_set_register()` | Blocked on FM | Needs cargo rebuild |

### Design Document

Full compiler design: `docs/DESIGN_FLUX_PYTHON_COMPILER.md`

## When to Apply

- Any new breeding constraint that needs proof certificates
- Any constraint that requires checkpoint/rollback (fault recovery)
- Any batch check that exceeds Python performance limits (use Rust VM)
- Any constraint that must be auditable (regulatory/compliance requirements)

## Examples

### Simple Range Check (High-Level Opcode)

```python
from sunset.flux_compiler_frontend import compile_lambda

bc, pool, asm = compile_lambda(
    "lambda x: x > 0 and x < 100",
    prefer_range_check=True,  # single RangeCheck opcode
)

# Disassembly:
# 0000  LoadConst  0          ; load x
# 0002  RangeCheck [0, 100]   ; push 1.0 if pass, 0.0 if fail
# 0011  Validate              ; trap if TOS == 0
# 0012  Halt
```

### Complex Constraint (Low-Level Arithmetic + Branch)

```python
bc, pool, asm = compile_lambda(
    "lambda w, t: abs(w) < 5.0 and t < 0.95",
    prefer_range_check=False,  # arithmetic + CondJump
)
```

## Prevention

- Always compile constraints to bytecode (don't use Python eval at runtime)
- Use `prefer_range_check=True` for simple bounds (smaller bytecode)
- Use `prefer_range_check=False` for complex logic (debuggable step-by-step)
- Test with Python fallback first, then verify with Rust VM
- Document constraint semantics in `docs/solutions/` for future agents

## Related

- `docs/DESIGN_FLUX_PYTHON_COMPILER.md` — Full compiler design
- `docs/FLUX_OPCODE_ALIGNMENT.md` — Original audit (60 opcodes, 0 used)
- `swarm/flux_compiler.py` — Existing bytecode compiler
- `sunset/flux_vm_bridge.py` — FFI bridge to Rust VM
