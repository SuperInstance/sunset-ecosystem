# SPEC-FLUX-RESOLUTION.md
**Author:** CCC (Systems Architect)  
**Date:** 2026-05-21  
**Status:** DECISION — v3 is the forward path

---

## 1. Situation

The FLUX ecosystem has a compiler/VM split that needs resolving:

| Component | Location | State |
|-----------|----------|-------|
| `flux-compiler` | `/flux-compiler/` | **Empty shell** — only README + CONTRIBUTING, no source code |
| `flux-vm` (v2) | `/flux-vm/` | ISA definitions only (mini, edge, std, thor) — no Rust runtime |
| `flux-vm-v3` | `/flux-vm-v3/` | **2673 lines of working Rust** — full VM, SIMD, JIT, proof, parallel |
| `forgemaster/flux-compiler` | `/forgemaster/flux-compiler/` | Forgemaster's copy (likely mirrors `flux-compiler` — needs verification) |

## 2. Decision: v3 is Forward

**flux-vm-v3 is the canonical runtime.** Rationale:

1. **It has source code.** 2,673 lines across 15 modules. The others are empty or ISA-only.
2. **It has the full feature set:** stack machine, 50+ opcodes, vector/SIMD unit, proof certificates, provenance logging, streaming constraints, parallel batch, JIT stubs (x86), checkpoint/rollback, bounded memory.
3. **It has real deps:** `sha2` for proof hashing, `rayon` for parallelism, `criterion` for benchmarking.
4. **It has the constraint-check domain right:** `check.rs` includes aviation and temperature presets — this is a *certifiable constraint checker*, not a general-purpose VM.

The `flux-compiler` repo was the original commit (GUARD DSL → FLUX-C compilation via `guard2mask`), but the actual compiler source was never pushed. The README describes what it should do; v3 implements the VM that receives the compiler's output.

## 3. Architecture: Compiler + VM Split

```
GUARD DSL source
    │
    ▼
flux-compiler (TO BE BUILT)
    │  PEG grammar (pest) → AST → FLUX bytecode
    ▼
flux-vm-v3 (EXISTS)
    │  Executes bytecode, checks constraints, produces proof certificates
    ▼
Verified output + ProofCertificate
```

The compiler is **not built yet**. The VM is. Ship order: stabilize v3 → build compiler against v3's opcode set.

## 4. Repository Actions

### Phase 1: Consolidate (immediate)

```bash
# 1. flux-compiler is empty — mark as archived/pending
cd /home/phoenix/.openclaw/workspace/flux-compiler
echo "Source pending. VM runtime is at flux-vm-v3/. See SPEC-FLUX-RESOLUTION." > STATUS.md
git add STATUS.md && git commit -m "archive: mark as pending compiler, v3 is VM runtime"

# 2. flux-vm (v2) is ISA-only — merge ISAs into v3
# v2 has: flux-isa, flux-isa-mini, flux-isa-edge, flux-isa-std, flux-isa-thor, flux-ast
# These are opcode DEFINITIONS, not implementations.
# ACTION: Copy ISA specs into flux-vm-v3/isa/ as reference docs
mkdir -p /home/phoenix/.openclaw/workspace/flux-vm-v3/isa/
cp -r /home/phoenix/.openclaw/workspace/flux-vm/flux-isa* /home/phoenix/.openclaw/workspace/flux-vm-v3/isa/
cp -r /home/phoenix/.openclaw/workspace/flux-vm/flux-ast /home/phoenix/.openclaw/workspace/flux-vm-v3/isa/

# 3. Tag v3 as canonical
cd /home/phoenix/.openclaw/workspace/flux-vm-v3
git tag v3-canonical
```

### Phase 2: Clean up duplicates

The FLUX ecosystem has massive duplication across fleet-*, forgemaster/*, quality-gate-stream/*, etc. Each contains copies of flux-compiler, flux-vm, flux-docs, flux-hardware. These are subtree clones from the Forgemaster era.

```bash
# These repos contain duplicate flux-* dirs that should be replaced with 
# symlinks or git submodules pointing to flux-vm-v3:
# - fleet-health-monitor/flux-{compiler,docs,hardware,vm}
# - fleet-murmur/flux-{compiler,docs,hardware,vm}
# - quality-gate-stream/flux-{compiler,docs,hardware,vm}
# - forgemaster/flux-* (20+ copies)

# ACTION: For each, delete the duplicate dir and add a submodule:
# cd <repo> && rm -rf flux-vm && git submodule add <flux-vm-v3-url> flux-vm
# This requires flux-vm-v3 to be pushed to GitHub first.
```

### Phase 3: Build the compiler

When the compiler is built, it targets v3's opcode set. The opcode enum in `src/opcode.rs` (149 lines) defines the instruction set. The compiler must emit these opcodes.

## 5. Compat Shim

If any code imports from `flux-vm` (v2), it needs a redirect:

```python
# flux_vm_compat.py — temporary shim
"""
Redirect old flux_vm imports to flux_vm_v3.
Only needed if external consumers reference the old path.
"""

import sys
import importlib


def _redirect(old_name, new_name):
    """Make old module name an alias for new."""
    mod = importlib.import_module(new_name)
    sys.modules[old_name] = mod


# Rust FFI: no compat needed — v3 has the same C ABI entry points
# Python: if flux-sdk-python references old paths, update import paths
```

In practice, since v2 has no source code, there's nothing to be compatible *with*. The shim is insurance for the SDK layer (`flux-sdk-python`).

## 6. Opcode Audit (v3 → compiler contract)

From `opcode.rs`, v3 has these opcode groups:

| Group | Opcodes | Count |
|-------|---------|-------|
| Stack | Push, Pop, Dup, Swap, Over, Drop, LoadConst, Nop | 8 |
| Arithmetic | Add, Sub, Mul, Div, Saturate, Min, Max, Abs | 8 |
| Register | LoadReg, StoreReg, LoadRegVec, StoreRegVec | 4 |
| Constraint | RangeCheck, BatchCheck, AccumulateMask, ClassifySeverity, Prove, QueryBackward, Simplify, Validate, HashCommit, Seal | 10 |
| Vector/SIMD | VecLoad, VecStore, VecRangeCheck, VecMaskMerge, VecReduce, VecGather | 6 |
| Control | FwdJump, CondJump, CallBounded, Ret, Halt, Checkpoint | 6 |
| Effects | SetHandler, EmitEvent, Rollback, GetResult | 4 |
| Parallel | ParDispatch, ParMerge, ParBarrier, ParReduce | 4 |
| Provenance | SnapRecord, SnapQuery, SnapHash, SnapVerify | 4 |
| Streaming | StreamOpen, StreamCheck, StreamBatch, StreamClose | 4 |

**Total: 58 opcodes.** The compiler must emit a subset of these. The `Simplify` opcode is currently identity — this is where constraint-specific simplification logic should plug in.

## 7. Summary

| What | Decision |
|------|----------|
| Forward VM | `flux-vm-v3` (2673 LOC, 15 modules, working) |
| Old `flux-vm` | ISA specs → merge into v3 as reference docs |
| `flux-compiler` | Empty — archive, build later targeting v3 opcodes |
| Duplicate flux-* dirs | Replace with git submodules to v3 |
| Compat shim | Minimal — v2 has no consumers (no source) |
| Next step | Push v3 to GitHub, then build compiler |
