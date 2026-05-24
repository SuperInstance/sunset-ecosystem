# FLUX Opcode Capability Index

**Purpose:** Prevent "compile and crash" failures by tracking which of the 58 FLUX-C v3 opcodes can safely be used from Python today.

**Scope:** Reference / tracking module. No compiler. No Rust code. Pure Python registry.

---

## Background

The FLUX VM audit (`flux-vm-v3-temp/FLUX-VM-V3-AUDIT.md`) found **58 opcodes** in the Rust VM and **zero** called from Python. Two integration paths exist:

| Path | Approach | Status |
|------|----------|--------|
| **A** | Constraint library — Python calls `check_candidate` / `check_batch` via FFI or pure-Python fallback | **Implemented** in `swarm/flux_gating.py` |
| **B** | Full VM compiler — Rust opcodes → Python bytecode | **Blocked** on Fleet Manager decision |

This index is the ground truth for Path A. Before any integration work emits a FLUX opcode, consult `can_use_from_python()`.

---

## Status Legend

| Status | Meaning |
|--------|---------|
| `PYTHON_SAFE` | Has a working Python fallback. Safe to use from Python. |
| `RUST_ONLY` | Requires the Rust VM runtime. No Python equivalent exists. |
| `PLANNED` | Python fallback is on the roadmap but not yet built. |
| `DEPRECATED` | Do not use in new code. |
| `UNTESTED` | No explicit status assigned yet. |

---

## Opcode Table

### Stack (8) — all PYTHON_SAFE

| # | Name | Hex | Status | Path A Equivalent |
|---|------|-----|--------|-------------------|
| 1 | Push | `0x01` | PYTHON_SAFE | — |
| 2 | Pop | `0x02` | PYTHON_SAFE | — |
| 3 | Dup | `0x03` | PYTHON_SAFE | — |
| 4 | Swap | `0x04` | PYTHON_SAFE | — |
| 5 | Over | `0x05` | PYTHON_SAFE | — |
| 6 | Drop | `0x06` | PYTHON_SAFE | — |
| 7 | LoadConst | `0x07` | PYTHON_SAFE | — |
| 8 | Nop | `0x08` | PYTHON_SAFE | — |

### Arithmetic (8) — all PYTHON_SAFE

| # | Name | Hex | Status | Path A Equivalent |
|---|------|-----|--------|-------------------|
| 9 | Add | `0x09` | PYTHON_SAFE | — |
| 10 | Sub | `0x0a` | PYTHON_SAFE | — |
| 11 | Mul | `0x0b` | PYTHON_SAFE | — |
| 12 | Div | `0x0c` | PYTHON_SAFE | — |
| 13 | Saturate | `0x0d` | PYTHON_SAFE | — |
| 14 | Min | `0x0e` | PYTHON_SAFE | — |
| 15 | Max | `0x0f` | PYTHON_SAFE | — |
| 16 | Abs | `0x10` | PYTHON_SAFE | — |

> **Note:** `Div` and `Abs` guard against `i32::MIN` edge cases (overflow panics in Rust). Python's `abs()` and `//` do not panic, so these are safe but semantically different.

### Register / Memory (4) — all RUST_ONLY

| # | Name | Hex | Status | Effort | Path A Equivalent |
|---|------|-----|--------|--------|-------------------|
| 17 | LoadReg | `0x11` | RUST_ONLY | medium | — |
| 18 | StoreReg | `0x12` | RUST_ONLY | medium | — |
| 19 | LoadRegVec | `0x13` | RUST_ONLY | high | — |
| 20 | StoreRegVec | `0x14` | RUST_ONLY | high | — |

### Constraint (10) — mixed

| # | Name | Hex | Status | Path A Equivalent |
|---|------|-----|--------|-------------------|
| 21 | RangeCheck | `0x15` | PYTHON_SAFE | `check_candidate` |
| 22 | BatchCheck | `0x16` | RUST_ONLY | `check_batch` |
| 23 | AccumulateMask | `0x17` | RUST_ONLY | `check_batch` |
| 24 | ClassifySeverity | `0x18` | PYTHON_SAFE | `check_candidate` |
| 25 | Prove | `0x19` | RUST_ONLY | — |
| 26 | QueryBackward | `0x1a` | RUST_ONLY | — |
| 27 | Simplify | `0x1b` | RUST_ONLY | — |
| 28 | Validate | `0x1c` | PYTHON_SAFE | `check_candidate` |
| 29 | HashCommit | `0x1d` | RUST_ONLY | — |
| 30 | Seal | `0x1e` | RUST_ONLY | — |

> **Gap:** BatchCheck and AccumulateMask have *named* Path A equivalents (`check_batch`) but the Python fallback does not actually emit these opcodes — it runs numpy-based constraint checks. A true Python batch pipeline would require medium effort to build.

### Vector / SIMD (6) — all RUST_ONLY

| # | Name | Hex | Status | Effort |
|---|------|-----|--------|--------|
| 31 | VecLoad | `0x1f` | RUST_ONLY | high |
| 32 | VecStore | `0x20` | RUST_ONLY | high |
| 33 | VecRangeCheck | `0x21` | RUST_ONLY | high |
| 34 | VecMaskMerge | `0x22` | RUST_ONLY | high |
| 35 | VecReduce | `0x23` | RUST_ONLY | medium |
| 36 | VecGather | `0x24` | RUST_ONLY | high |

### Control Flow (6) — mixed

| # | Name | Hex | Status | Notes |
|---|------|-----|--------|-------|
| 37 | FwdJump | `0x25` | PYTHON_SAFE | Forward-only; trivial to implement |
| 38 | CondJump | `0x26` | PYTHON_SAFE | Forward-only; trivial to implement |
| 39 | CallBounded | `0x27` | RUST_ONLY | Needs cycle-bound enforcement |
| 40 | Ret | `0x28` | PYTHON_SAFE | Trivial subroutine return |
| 41 | Halt | `0x29` | PYTHON_SAFE | Trivial stop |
| 42 | Checkpoint | `0x2a` | RUST_ONLY | Needs checkpoint stack |

### Effects (4) — mostly RUST_ONLY

| # | Name | Hex | Status | Path A Equivalent |
|---|------|-----|--------|-------------------|
| 43 | SetHandler | `0x2b` | RUST_ONLY | — |
| 44 | EmitEvent | `0x2c` | PYTHON_SAFE | `record_violation` |
| 45 | Rollback | `0x2d` | RUST_ONLY | — |
| 46 | GetResult | `0x2e` | RUST_ONLY | — |

### Parallel (4) — all RUST_ONLY

| # | Name | Hex | Status | Effort |
|---|------|-----|--------|--------|
| 47 | ParDispatch | `0x2f` | RUST_ONLY | high |
| 48 | ParMerge | `0x30` | RUST_ONLY | high |
| 49 | ParBarrier | `0x31` | RUST_ONLY | high |
| 50 | ParReduce | `0x32` | RUST_ONLY | high |

> **Note:** These depend on Rayon. A Python equivalent would need `multiprocessing` or `concurrent.futures` reimplementation — high effort.

### Provenance (4) — all RUST_ONLY

| # | Name | Hex | Status | Effort |
|---|------|-----|--------|--------|
| 51 | SnapRecord | `0x33` | RUST_ONLY | medium |
| 52 | SnapQuery | `0x34` | RUST_ONLY | medium |
| 53 | SnapHash | `0x35` | RUST_ONLY | medium |
| 54 | SnapVerify | `0x36` | RUST_ONLY | low |

> **Note:** `SnapVerify` is a stub in the Rust VM (always returns `1`). Even so, a Python fallback would need the same SHA-256 chain infrastructure, so it stays RUST_ONLY until that exists.

### Streaming (4) — all RUST_ONLY

| # | Name | Hex | Status | Effort |
|---|------|-----|--------|--------|
| 55 | StreamOpen | `0x37` | RUST_ONLY | medium |
| 56 | StreamCheck | `0x38` | RUST_ONLY | medium |
| 57 | StreamBatch | `0x39` | RUST_ONLY | medium |
| 58 | StreamClose | `0x3a` | RUST_ONLY | medium |

---

## Summary

| Status | Count | Fraction |
|--------|-------|----------|
| **PYTHON_SAFE** | 22 | 38 % |
| **RUST_ONLY** | 36 | 62 % |
| DEPRECATED | 0 | 0 % |
| PLANNED | 0 | 0 % |
| UNTESTED | 0 | 0 % |

**Key insight:** Only 22 of 58 opcodes can be used from Python today. Any integration work that assumes the full VM is available will crash at runtime when it hits a RUST_ONLY opcode.

---

## Gap Report — Priority Order

Opcodes sorted by effort estimate for a Python fallback:

### Low effort (could be done in a single session)
- **SnapVerify** — stub already; Python `hashlib.sha256` verification is trivial
- **EmitEvent** — WAL logging already exists; just needs formal opcode wrapper

### Medium effort (1–2 sessions)
- **Checkpoint / Rollback** — Python `copy.deepcopy` stack + context manager
- **BatchCheck / AccumulateMask** — extend `PythonFluxFallback.check_batch()`
- **SnapRecord / SnapQuery / SnapHash** — ring buffer in Python
- **Streaming quartet** — Python `io.BytesIO` buffer + iterator protocol
- **LoadReg / StoreReg** — simple dict-based register file

### High effort (blocked on architecture)
- **Vector / SIMD sextet** — needs numpy SIMD or `numba` integration
- **Parallel quartet** — needs multiprocessing orchestration layer
- **Prove / QueryBackward / HashCommit / Seal** — needs full proof-carrying infrastructure
- **SetHandler / GetResult** — needs effects system reimplementation
- **CallBounded** — needs cycle-counter instrumentation

---

## API Quick Reference

```python
from logos.opcode_capability_index import OpcodeCapabilityIndex

idx = OpcodeCapabilityIndex()

# Can I use this opcode from Python?
idx.can_use_from_python("RangeCheck")   # → True
idx.can_use_from_python("Prove")        # → False

# List all safe opcodes in a category
idx.get_safe_opcodes(category="arithmetic")

# Get a structured gap report for planning
idx.get_gap_report()

# Suggest the Python fallback function name
idx.suggest_path_a_equivalent("RangeCheck")  # → "check_candidate"

# Override after manual testing
from logos.opcode_capability_index import OpcodeStatus
idx.update_status("SnapVerify", OpcodeStatus.PYTHON_SAFE)

# Persist
idx.save("/tmp/flux_opcode_index.json")
loaded = OpcodeCapabilityIndex.load("/tmp/flux_opcode_index.json")
```

---

## Files

| File | Role |
|------|------|
| `logos/opcode_capability_index.py` | Registry + queries + persistence |
| `tests/test_opcode_capability_index.py` | 21 tests covering coverage, categories, status, gaps, overrides, persistence |
| `docs/OPCODE_CAPABILITY_INDEX.md` | This document |

---

*Index version: 1 | Canonical source: `flux-vm-v3-temp/src/opcode.rs` | Last updated: 2026-05-25*
