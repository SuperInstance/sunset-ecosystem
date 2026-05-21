# SPEC-FLUX-RESOLUTION.md — Resolve the FLUX Compiler Split

## Problem

FLUX exists as two independent codebases that share a name but not an ISA:

1. **flux-compiler** (`/flux-compiler/`) — initial commit only, empty repo (just `.git/`). The "v2" that was supposed to be a constraint compiler but never materialized.
2. **flux-vm-v3** (`/flux-vm-v3/`) — active Rust crate with 60 opcodes, stack machine VM, JIT (x86), proof certificates, provenance logging, streaming, vector unit, parallel execution, and benchmarks.

There are also **6 copies** of flux-compiler scattered across subprojects:
- `flux-research/flux-compiler/` — empty
- `.local-plato/twin/flux-compiler/` — 2 files (stub)
- `quality-gate-stream/flux-compiler/` — empty
- `fleet-health-monitor/flux-compiler/` — empty
- `fleet-murmur/flux-compiler/` — empty
- `forgemaster/flux-compiler/` — 2 files (stub)

The v3 codebase is production-quality (see `flux-vm-v3/src/`):
- `opcode.rs`: 60 opcodes (stack, arithmetic, register, constraint, vector, control flow, provenance)
- `vm.rs`: Stack machine with bounded memory, cycle limit (4096), checkpoints
- `jit.rs` / `jit_x86.rs`: JIT compilation layer
- `proof.rs`: Proof certificates for constraint verification
- `provenance.rs`: Full provenance chain
- `effects.rs`: Effect handler system
- `streaming.rs`: Stream state management
- `vector.rs`: SIMD vector unit
- `parallel.rs`: Parallel constraint evaluation
- `memory.rs`: Bounded memory with STACK_LIMIT
- `check.rs`: Constraint types with aviation preset
- `bench.rs`: Criterion benchmarks

## Ground-Level Code

### Files to move/rename/delete

```
# DELETE — empty flux-compiler copies
flux-research/flux-compiler/          → DELETE (empty repo)
.local-plato/twin/flux-compiler/      → DELETE (2-file stub)
quality-gate-stream/flux-compiler/    → DELETE (empty)
fleet-health-monitor/flux-compiler/   → DELETE (empty)
fleet-murmur/flux-compiler/           → DELETE (empty)
forgemaster/flux-compiler/            → DELETE (2-file stub)

# RENAME — flux-compiler becomes flux-vm-v2-archive
flux-compiler/                        → mv flux-compiler flux-vm-v2-archive
                                        (preserves git history, marks as dead)

# CANONICAL — flux-vm-v3 IS the FLUX compiler going forward
flux-vm-v3/                           → RENAME to flux-compiler/
                                        (it earns the name)
```

### Compat layer (new file)

Create `flux-vm-v3/src/compat.rs`:

```rust
//! Compatibility shim for any v2-era tooling that expected
//! the old flux-compiler Python interface.
//!
//! v2 never shipped code, so this is purely documentation.
//! Any external references to "flux-compiler" should resolve
//! to flux-vm-v3 (now renamed flux-compiler).

/// v2 never defined an ISA. v3's ISA starts here.
/// If you need to port a v2 concept, map it to:
///   - Constraint checks → OpCode::RangeCheck, BatchCheck
///   - Safety proofs → proof::ProofCertificate
///   - Memory bounds → memory::BoundedMemory
pub const V2_COMPAT_NOTE: &str = "v2 had no ISA. Use v3 opcodes directly.";
```

## Decision

**v3 is the forward path. v2 is legacy maintenance (but really just archival).**

Rationale:
- v2 (`flux-compiler`) never shipped a single source file. It's an empty git repo.
- v3 (`flux-vm-v3`) has 15 source files, JIT compilation, proof certificates, benchmarks.
- 6 scattered copies of v2 are debris from cross-repo cloning, not meaningful forks.
- There is no compat burden because v2 has no consumers.

**Action**: Rename `flux-vm-v3/` → `flux-compiler/`. Archive the old empty `flux-compiler/` as `flux-vm-v2-archive/`. Delete all 6 stub copies.

## Implementation Order

1. `mv flux-compiler flux-vm-v2-archive` — preserve old empty repo
2. `mv flux-vm-v3 flux-compiler` — v3 takes the canonical name
3. Update `Cargo.toml` name from `flux-vm-v3` to `flux-compiler`
4. Delete 6 stub copies in subprojects
5. Create `src/compat.rs` with documentation shim
6. Update any imports referencing `flux_vm_v3` → `flux_compiler`
7. `cargo test` in the renamed directory
8. Update STRUCTURAL-SURVEY.md to reflect single FLUX codebase

## Success Criteria

- [ ] `flux-compiler/` contains the v3 Rust crate (15+ source files)
- [ ] `flux-vm-v2-archive/` exists with old empty repo
- [ ] No duplicate flux-compiler directories in any subproject
- [ ] `cargo test` passes in renamed `flux-compiler/`
- [ ] `cargo bench` benchmarks still run
- [ ] STRUCTURAL-SURVEY.md updated to show single FLUX codebase
