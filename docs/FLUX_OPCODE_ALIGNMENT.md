# FLUX Opcode Alignment Audit

**Scope:** flux-vm-v3 (Rust VM) ↔ sunset-ecosystem (Python)  
**Date:** 2026-05-25  
**Auditor:** Technical Integration Scout (Cocapn Fleet)  
**Files examined:**
- `flux_vm/ffi.py` — Python FFI wrapper
- `sunset/compiler.py` — Python "compiler"
- `sunset/flux_integration.py` — Integration bridge
- `flux-vm-v3-temp/src/opcode.rs` — Rust VM opcode definitions
- `flux-vm-v3-temp/src/vm.rs` — Rust VM core
- `flux-vm-v3-temp/src/lib.rs` — Rust public API

---

## Executive Summary

**There is no opcode alignment because the two systems do not speak at the bytecode layer.**

The Rust VM defines 60 opcodes and a full stack-machine execution engine. The Python side does not emit a single opcode, nor does it load bytecode into the VM, nor does it invoke the VM's execution loop. The integration is entirely at the FFI function-call level (`flux_check_batch()`), treating the Rust side as a constraint-checking library rather than a virtual machine.

---

## 1. Rust VM Opcode Inventory

The Rust VM (`flux-vm-v3-temp/src/opcode.rs`) defines **60 opcodes** across 10 categories:

| Category | Count | Opcodes |
|----------|-------|---------|
| Stack | 8 | `Push`, `Pop`, `Dup`, `Swap`, `Over`, `Drop`, `LoadConst`, `Nop` |
| Arithmetic | 8 | `Add`, `Sub`, `Mul`, `Div`, `Saturate`, `Min`, `Max`, `Abs` |
| Register | 4 | `LoadReg`, `StoreReg`, `LoadRegVec`, `StoreRegVec` |
| Constraint | 10 | `RangeCheck`, `BatchCheck`, `AccumulateMask`, `ClassifySeverity`, `Prove`, `QueryBackward`, `Simplify`, `Validate`, `HashCommit`, `Seal` |
| Vector/SIMD | 6 | `VecLoad`, `VecStore`, `VecRangeCheck`, `VecMaskMerge`, `VecReduce`, `VecGather` |
| Control | 6 | `FwdJump`, `CondJump`, `CallBounded`, `Ret`, `Halt`, `Checkpoint` |
| Effects | 4 | `SetHandler`, `EmitEvent`, `Rollback`, `GetResult` |
| Parallel | 4 | `ParDispatch`, `ParMerge`, `ParBarrier`, `ParReduce` |
| Provenance | 4 | `SnapRecord`, `SnapQuery`, `SnapHash`, `SnapVerify` |
| Streaming | 4 | `StreamOpen`, `StreamCheck`, `StreamBatch`, `StreamClose` |

**VM execution entry points** (`src/vm.rs`):
- `FluxVM::load_bytecode(&mut self, bc: &[u8])` — loads bytecode
- `FluxVM::run(&mut self) -> VmResult` — executes the opcode loop
- `FluxVM::step(&mut self) -> FluxResult<()>` — single-step execution

**Public API** (`src/lib.rs` re-exports): `FluxVM`, `OpCode`, `VmResult`, etc.

---

## 2. Python "Compiler" — Opcode Emission Analysis

`sunset/compiler.py` is **not a bytecode compiler**. It is an **agentic JIT profiler** that:
1. Watches Python function calls at runtime
2. Identifies hot paths via statistical profiling
3. Recompiles them to Numba LLVM, Rust FFI, or CUDA kernels via `CodeGenerator`
4. Hot-swaps the Python function with the compiled version

**Opcode emission:** **ZERO**. The file contains no opcode definitions, no bytecode generation, no `OpCode` enum, no `.flux` file writer. The word "bytecode" appears only in the context of Python's built-in `types.CodeType` (for hot-swapping Python code objects).

**Relevant classes:**
- `Compiler` — profiler + hot-swap orchestrator
- `NumbaBackend`, `RustBackend` — JIT compilation targets (not FLUX VM targets)
- `hot_swap()` — replaces a Python function at runtime

**Conclusion:** `sunset/compiler.py` cannot emit FLUX opcodes and was never designed to.

---

## 3. Integration Layer — How the Python Side Actually Uses Rust

`sunset/flux_integration.py` provides `FluxConstraintChecker`, which bridges `RoomGrid.tick()` to constraint checking.

**What it calls:**
- `_RustBackend.check_batch()` → `ctypes.CDLL("libflux_vm.so").flux_check_batch(...)`

**What it does NOT call:**
- `FluxVM::new()` — never instantiates the VM
- `FluxVM::load_bytecode()` — never loads bytecode
- `FluxVM::run()` — never executes opcodes
- `FluxVM::step()` — never single-steps

The FFI function `flux_check_batch()` (exposed in `ffi.rs`) is a **direct constraint-checking routine** that performs bounds/L2/variance checks in native code. It is NOT a VM entry point — it does not decode or execute opcodes. It is a thin native wrapper around the same logic implemented in `_PythonBackend`.

**Conclusion:** The integration bypasses the VM entirely. All 60 opcodes are unused.

---

## 4. The Missing Piece — Where Is the Real FLUX Compiler?

There **is** a legitimate FLUX bytecode compiler: `flux-compiler-v0.1.0/guardc/src/compiler.rs`

Pipeline: `GUARD source → Parser → Typechecker → CIR → Lowering → LCIR → Codegen → FLUX bytecode`

**Status:** Not wired into sunset-ecosystem.

The Python integration layer knows nothing about it. There is no Python wrapper for `guardc`, no `compile_guard_to_bytecode()` function, no path from `sunset/compiler.py` to FLUX opcodes.

---

## 5. Opcode Alignment Table

| Rust Opcode | Python Equivalent | Status | Notes |
|-------------|-------------------|--------|-------|
| `Push` | — | **missing** | No bytecode emitter |
| `Pop` | — | **missing** | No bytecode emitter |
| `Dup` | — | **missing** | No bytecode emitter |
| `Swap` | — | **missing** | No bytecode emitter |
| `Over` | — | **missing** | No bytecode emitter |
| `Drop` | — | **missing** | No bytecode emitter |
| `LoadConst` | — | **missing** | No bytecode emitter |
| `Nop` | — | **missing** | No bytecode emitter |
| `Add` | — | **missing** | No bytecode emitter |
| `Sub` | — | **missing** | No bytecode emitter |
| `Mul` | — | **missing** | No bytecode emitter |
| `Div` | — | **missing** | No bytecode emitter |
| `Saturate` | — | **missing** | No bytecode emitter |
| `Min` | — | **missing** | No bytecode emitter |
| `Max` | — | **missing** | No bytecode emitter |
| `Abs` | — | **missing** | No bytecode emitter |
| `LoadReg` | — | **missing** | No bytecode emitter |
| `StoreReg` | — | **missing** | No bytecode emitter |
| `LoadRegVec` | — | **missing** | No bytecode emitter |
| `StoreRegVec` | — | **missing** | No bytecode emitter |
| `RangeCheck` | `_PythonBackend.check_batch()` bounds check | **partial** | Logic exists in Python, but not as VM opcode |
| `BatchCheck` | `_RustBackend.check_batch()` | **partial** | Rust FFI performs batch check, but NOT via opcode execution |
| `AccumulateMask` | `np.zeros(n, dtype=bool)` accumulation | **partial** | Mask logic in Python, not as opcode |
| `ClassifySeverity` | `ConstraintViolation.severity` calculation | **partial** | Severity computed in Python post-hoc |
| `Prove` | — | **missing** | No proof generation on Python side |
| `QueryBackward` | — | **missing** | No backward query capability exposed |
| `Simplify` | — | **missing** | No simplifier |
| `Validate` | `_validate()` in Compiler | **partial** | A/B test validation exists, but unrelated to VM `Validate` opcode |
| `HashCommit` | — | **missing** | No hashing of constraint state |
| `Seal` | — | **missing** | No sealing mechanism |
| `VecLoad` | — | **missing** | No vector/SIMD opcode emitter |
| `VecStore` | — | **missing** | No vector/SIMD opcode emitter |
| `VecRangeCheck` | — | **missing** | No vector/SIMD opcode emitter |
| `VecMaskMerge` | — | **missing** | No vector/SIMD opcode emitter |
| `VecReduce` | — | **missing** | No vector/SIMD opcode emitter |
| `VecGather` | — | **missing** | No vector/SIMD opcode emitter |
| `FwdJump` | — | **missing** | No control-flow codegen |
| `CondJump` | — | **missing** | No control-flow codegen |
| `CallBounded` | — | **missing** | No bounded call mechanism |
| `Ret` | — | **missing** | No return opcode |
| `Halt` | — | **missing** | No halt opcode |
| `Checkpoint` | — | **missing** | No checkpoint/rollback via VM |
| `SetHandler` | — | **missing** | No effect handler registration |
| `EmitEvent` | logging / print statements | **partial** | Events emitted via Python logging, not VM effects |
| `Rollback` | `hot_swap_restore()` | **partial** | Rollback exists for function hot-swap, NOT VM state rollback |
| `GetResult` | `check_batch()` return value | **partial** | Results retrieved from FFI call, not VM register |
| `ParDispatch` | — | **missing** | No parallel dispatch |
| `ParMerge` | — | **missing** | No parallel merge |
| `ParBarrier` | — | **missing** | No parallel barrier |
| `ParReduce` | — | **missing** | No parallel reduce |
| `SnapRecord` | — | **missing** | No provenance recording |
| `SnapQuery` | — | **missing** | No provenance query |
| `SnapHash` | — | **missing** | No provenance hash |
| `SnapVerify` | — | **missing** | No provenance verification |
| `StreamOpen` | — | **missing** | No streaming support |
| `StreamCheck` | — | **missing** | No streaming support |
| `StreamBatch` | — | **missing** | No streaming support |
| `StreamClose` | — | **missing** | No streaming support |

---

## 6. Mismatches & Fix Suggestions

### P0 — Architectural Disconnect

**Issue:** The Python side treats FLUX as a constraint library; the Rust side is a full VM. The integration never loads bytecode or executes opcodes.

**Impact:** The VM's proof-carrying, checkpoint/rollback, streaming, parallel dispatch, and provenance features are completely inaccessible from Python.

**Fix suggestion (choose one path):**

**Path A — Embrace the Library Model (simpler)**
- Accept that FLUX is a constraint library for the fleet
- Remove VM complexity from the Rust side if unused
- Keep `flux_check_batch()` as the stable API boundary
- *Effort:* Low. Status quo with documentation.

**Path B — Full VM Integration (harder, more powerful)**
- Build a Python → FLUX bytecode compiler that translates constraint expressions to `OpCode` sequences
- Add `FluxVM.load_bytecode()` and `.run()` calls to `flux_integration.py`
- Wire `guardc` compiler into the Python pipeline
- Expose VM features (proofs, checkpoints, streaming) to Python
- *Effort:* High. Requires new Python module `sunset/flux_codegen.py` + FFI extensions.

### P1 — Missing Compiler Bridge

**Issue:** `flux-compiler-v0.1.0/guardc` can produce FLUX bytecode, but there is no Python interface to it.

**Fix suggestion:**
- Add a `FluxCompiler` Python class that wraps `guardc` via subprocess or a Rust PyO3 module
- Input: constraint expressions (Python AST or string)
- Output: `bytes` ready for `FluxVM.load_bytecode()`
- *Effort:* Medium. Depends on whether `guardc` is stable.

### P1 — Opacity of `flux_check_batch()`

**Issue:** The FFI function performs the same checks as the VM's constraint opcodes, but without the VM's traceability (no proof certificate, no provenance log).

**Fix suggestion:**
- If staying on Path A: document this tradeoff explicitly
- If moving to Path B: replace `flux_check_batch()` with a VM-run that executes a pre-compiled `RangeCheck` + `BatchCheck` + `AccumulateMask` opcode sequence, yielding a `ProofCertificate`

### P2 — JIT Compiler Name Collision

**Issue:** `sunset/compiler.py` is called a "compiler" but has nothing to do with FLUX bytecode. This confuses the architecture.

**Fix suggestion:**
- Rename `sunset/compiler.py` → `sunset/jit_profiler.py` or similar
- Reserve "compiler" for the FLUX bytecode compiler (whether `guardc` or a future Python one)

---

## 7. Integration Bridge Code Needed

If the fleet chooses **Path B** (full VM integration), the following Python code would be the minimal bridge:

```python
# sunset/flux_vm_bridge.py — hypothetical

import ctypes
from pathlib import Path
import numpy as np


class FluxVMBridge:
    """Python wrapper for the full FLUX VM (not just check_batch)."""

    def __init__(self, so_path: str):
        self._lib = ctypes.CDLL(so_path)
        # FFI signatures for VM lifecycle
        self._lib.flux_vm_new.argtypes = []
        self._lib.flux_vm_new.restype = ctypes.c_void_p
        self._lib.flux_vm_load_bytecode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self._lib.flux_vm_run.argtypes = [ctypes.c_void_p]
        self._lib.flux_vm_run.restype = ctypes.c_int
        self._lib.flux_vm_free.argtypes = [ctypes.c_void_p]

        self._vm = self._lib.flux_vm_new()

    def load(self, bytecode: bytes):
        arr = (ctypes.c_uint8 * len(bytecode))(*bytecode)
        self._lib.flux_vm_load_bytecode(self._vm, arr, len(bytecode))

    def run(self) -> int:
        return self._lib.flux_vm_run(self._vm)

    def __del__(self):
        if hasattr(self, "_vm") and self._vm:
            self._lib.flux_vm_free(self._vm)
```

**Note:** The Rust FFI (`src/ffi.rs`) currently does NOT export `flux_vm_new`, `flux_vm_load_bytecode`, or `flux_vm_run`. It only exports `flux_check_batch`. Adding these exports would be required for Path B.

---

## 8. Priority Ranking

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| **P0** | Decide Path A (library) vs Path B (full VM) | Low | Defines all subsequent work |
| **P0** | Document the architectural disconnect | Low | Prevents future confusion |
| **P1** | Wire `guardc` compiler to Python (if Path B) | Medium | Enables bytecode generation |
| **P1** | Extend FFI to expose VM lifecycle functions (if Path B) | Medium | Enables VM execution from Python |
| **P1** | Rename `sunset/compiler.py` to avoid confusion | Low | Clarity |
| **P2** | Build `sunset/flux_codegen.py` bytecode emitter (if Path B) | High | Full opcode integration |
| **P2** | Add provenance/proof streaming to Python API (if Path B) | High | Leverages VM features |

---

## 9. Conclusion

The "opcode alignment" question is moot in the current architecture because the two systems do not interact at the opcode layer. The Rust VM has 60 well-defined opcodes; the Python side has zero. The integration is a single FFI function call (`flux_check_batch`) that treats the Rust component as a native constraint library, not as a VM.

**No opcodes are aligned. No opcodes are emitted. No opcodes are executed from Python.**

The real question for the fleet is: *do we want FLUX to be a library or a VM?* The answer determines whether to invest in Path A (文档 + 简化) or Path B (full compiler + VM lifecycle bridge).

---

*Audit complete. Report path:* `docs/FLUX_OPCODE_ALIGNMENT.md`
