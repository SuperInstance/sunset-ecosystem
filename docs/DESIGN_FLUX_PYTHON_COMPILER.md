# Python → FLUX VM Bytecode Compiler Design Document

**Fleet:** Cocapn | **Author:** kimi1 (Fleet Orchestrator)  
**Date:** 2026-05-29 | **Status:** Design Complete — Awaits Implementation  
**Scope:** Path B — Full VM Integration (Python AST → FLUX bytecode → Rust VM execution)

---

## 1. Executive Summary

The fleet has already built a **partial** Python→FLUX compiler (`swarm/flux_compiler.py`) and a **full VM bridge** (`sunset/flux_vm_bridge.py`). What's missing is the **Python `ast` frontend** — a module that takes real Python lambda expressions, parses them with `ast.parse`, and feeds them into the existing compiler infrastructure.

This document defines the missing link: a Python `ast` → `FluxCompiler` AST adapter, plus the full compilation pipeline from Python source to Rust VM execution.

**What already exists:**
- `swarm/flux_compiler.py` — Bytecode emitter + custom AST compiler
- `swarm/flux_vm_runner.py` — Python fallback VM interpreter
- `sunset/flux_vm_bridge.py` — Full FFI bridge to Rust VM (new, load, run, proof)
- `flux_vm/libflux_vm_v3.so` — Rust VM with 60 opcodes, lifecycle functions exported

**What this design adds:**
- `sunset/flux_ast_compiler.py` — Python `ast` → FLUX bytecode compiler
- `sunset/flux_compiler_frontend.py` — User-facing API (lambdas, functions, strings)
- FFI completion plan — what's exported vs what's still needed on the Rust side

---

## 2. Architecture Overview

```
Python source (lambda/function)
       │
       ▼
  ast.parse()  ──►  ast.AST  ──►  [ast → Flux AST adapter]  ──►  FluxCompiler
       │                                                              │
       │                                                              ▼
       │                                                    bytecode bytes
       │                                                              │
       │                                                              ▼
       │                                              ┌─────────────────────────────┐
       │                                              │  Python: FluxVMRunner       │
       │                                              │  (fallback / debug)           │
       │                                              └─────────────────────────────┘
       │                                                              │
       │                                                              ▼
       │                                              ┌─────────────────────────────┐
       │                                              │  Rust FFI: FluxVMBridge       │
       │                                              │  flux_vm_new()                │
       │                                              │  flux_vm_load_bytecode()      │
       │                                              │  flux_vm_run()                │
       │                                              │  flux_vm_get_proof()          │
       │                                              └─────────────────────────────┘
       │                                                              │
       ▼                                                              ▼
   Proof Certificate (SHA-256)                             pass/fail + cycles
```

---

## 3. Opcode Mapping Table (Python AST → FLUX Opcode)

### 3.1 Arithmetic Expressions

| Python AST | FLUX Opcode | Stack Effect | Notes |
|-----------|-------------|--------------|-------|
| `ast.Constant` (int/float) | `Push <f32>` | `→ value` | Immediate or constant pool |
| `ast.Name` (variable) | `LoadConst <u8>` | `→ value` | Resolved from constant pool |
| `ast.BinOp` with `Add` | `Add` | `a b → a+b` | |
| `ast.BinOp` with `Sub` | `Sub` | `a b → a-b` | |
| `ast.BinOp` with `Mult` | `Mul` | `a b → a*b` | |
| `ast.BinOp` with `Div` | `Div` | `a b → a/b` | Zero-div → inf (VM behavior) |
| `ast.UnaryOp` with `USub` | `Push -1.0` + `Mul` | `a → -a` | |
| `ast.UnaryOp` with `UAdd` | `Nop` | `a → a` | No-op |
| `ast.Call` `abs(x)` | `Abs` | `a → |a|` | |
| `ast.Call` `min(a,b)` | `Min` | `a b → min` | |
| `ast.Call` `max(a,b)` | `Max` | `a b → max` | |
| `ast.Call` `saturate(x, lo, hi)` | `Saturate <f32> <f32>` | `x → clamp(x)` | 9-byte instruction |

### 3.2 Comparison Expressions

| Python AST | FLUX Opcode Pattern | Semantics |
|-----------|---------------------|-----------|
| `ast.Compare` with `Lt` | `Sub` + `CondJump` | `a < b` → `a - b < 0` → jump if `≤ 0` |
| `ast.Compare` with `LtE` | `Sub` + `CondJump` | `a ≤ b` → `a - b ≤ 0` → jump if `≤ 0` |
| `ast.Compare` with `Gt` | `Swap` + `Sub` + `CondJump` | `a > b` → `b - a < 0` → jump if `≤ 0` |
| `ast.Compare` with `GtE` | `Swap` + `Sub` + `CondJump` | `a ≥ b` → `b - a ≤ 0` → jump if `≤ 0` |
| `ast.Compare` with `Eq` | `Sub` + `Abs` + `Push ε` + `Sub` + `CondJump` | `|a-b| ≤ ε` (float tolerance) |
| `ast.Compare` with `NotEq` | `Sub` + `Abs` + `Push ε` + `Sub` + `FwdJump` | Inverted Eq pattern |
| `ast.Compare` with `In` | `RangeCheck <lo> <hi>` | `lo ≤ a ≤ hi` → pushes 1.0 or 0.0 |

**Note:** `CondJump` semantics in the FLUX VM: **jump if TOS ≤ 0**. This is a signed comparison on the floating-point top-of-stack value. The compiler inverts comparisons by swapping operands or adding `FwdJump` fall-through logic.

### 3.3 Boolean Logic

| Python AST | FLUX Pattern | Semantics |
|-----------|-------------|-----------|
| `ast.BoolOp` with `And` | `left` + `CondJump else` + `right` + `FwdJump end` | Short-circuit: if left is false (≤0), skip right |
| `ast.BoolOp` with `Or` | `left` + `CondJump end` + `right` + `FwdJump end` | Short-circuit: if left is true (>0), skip right |
| `ast.UnaryOp` with `Not` | `Push 0` + `Swap` + `Sub` + `Saturate 0 1` | `not a` → `0 - a` clamped to [0,1] |

**Truthiness convention:** FLUX uses `> 0` as true, `≤ 0` as false. Boolean results are 1.0 (true) or 0.0 (false).

### 3.4 High-Level Constraint Opcodes (Compiler Optimization)

The compiler should recognize common patterns and emit single opcodes instead of multi-instruction sequences:

| Python Pattern | Optimized FLUX Opcode | When to Use |
|---------------|----------------------|-------------|
| `lo ≤ x ≤ hi` | `RangeCheck <lo> <hi>` | `prefer_range_check=True` (default) |
| `all(lo ≤ x[i] ≤ hi for i in range(n))` | `BatchCheck` + `AccumulateMask` | Vectorized batch checks |
| `x[i] ∈ [lo, hi]` for SIMD | `VecRangeCheck` + `VecMaskMerge` | When `n` is a multiple of 8 |
| `np.linalg.norm(x) < max_l2` | `VecLoad` + `VecReduce` + `Sub` + `CondJump` | L2 norm check |
| Severity classification | `ClassifySeverity <u8>` | Post-check severity tagging |
| Proof generation | `Prove` + `HashCommit` + `Seal` | When `collect_proof=True` |
| Provenance logging | `SnapRecord` + `SnapHash` | Metronome heartbeat traces |

---

## 4. Compilation Pipeline Stages

### Stage 1: Source Acquisition

```python
# Three input modes supported:

# A. Lambda expression (most common)
source = "lambda x: x > 0 and x < 100"

# B. Function AST (from inspect)
import inspect
tree = ast.parse(inspect.getsource(my_func))

# C. String expression (with variable binding)
source = "x > 0 and x < 100"
vars = {"x": 5.0}
```

**Stage 1 responsibilities:**
- Parse with `ast.parse(source, mode='eval')` for lambdas/expressions
- Parse with `ast.parse(source, mode='exec')` for function definitions
- Extract `ast.Lambda` or `ast.FunctionDef` body
- Validate: only supported AST nodes (no `ast.For`, `ast.While`, `ast.Import`, etc.)

### Stage 2: AST Normalization (Python `ast` → FLUX-compatible AST)

```python
class FluxASTNormalizer(ast.NodeTransformer):
    """Transform Python ast into a form the FLUX compiler can consume."""
```

**Normalization passes:**
1. **Lambda lifting:** Extract lambda body, treat parameters as `Var` nodes
2. **Comparison flattening:** Chain `a < b < c` into `a < b and b < c`
3. **Boolean short-circuit expansion:** `And`/`Or` → explicit `IfNode` structures
4. **Call resolution:** `abs(x)` → `UnaryOp("Abs", x)`, `min(a,b)` → `BinOp("Min", a, b)`
5. **Constant folding:** `3 + 4` → `Const(7.0)` at compile time
6. **Name resolution:** Map Python names to constant pool slots or register indices

**Unsupported nodes (raise `FluxCompileError`):**
- `ast.For`, `ast.While`, `ast.AsyncFor` — no loops in FLUX VM
- `ast.Import`, `ast.ImportFrom` — no imports
- `ast.ClassDef` — no classes
- `ast.Try`, `ast.With` — no exception handling
- `ast.Subscript` with non-constant index — arrays require `VecLoad`/`VecGather`
- `ast.ListComp`, `ast.DictComp` — no comprehensions (use `VecRangeCheck` instead)

### Stage 3: AST Translation (Normalized Python AST → `FluxCompiler` AST)

The existing `swarm/flux_compiler.py` defines its own AST nodes:
- `Const(value: float)`
- `Var(name: str)`
- `BinOp(op: str, left: Expr, right: Expr)`
- `UnaryOp(op: str, operand: Expr)`
- `RangeCheckNode(expr: Expr, lo: float, hi: float)`
- `IfNode(cond: CmpOp, then_expr: Expr, else_expr: Expr)`
- `CmpOp(op: str, left: Expr, right: Expr)`

**Translation table:**

```python
def translate(node: ast.AST) -> Expr:
    if isinstance(node, ast.Constant):
        return Const(float(node.value))
    elif isinstance(node, ast.Name):
        return Var(node.id)
    elif isinstance(node, ast.BinOp):
        return BinOp(op_map[type(node.op)], translate(node.left), translate(node.right))
    elif isinstance(node, ast.Compare):
        # Flatten chain: a < b < c → a < b and b < c
        return flatten_compare(node)
    elif isinstance(node, ast.BoolOp):
        # And/Or → IfNode with CondJump
        return flatten_boolop(node)
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return BinOp("Mul", Const(-1.0), translate(node.operand))
        elif isinstance(node.op, ast.Not):
            return IfNode(
                CmpOp("LE", translate(node.operand), Const(0.0)),
                Const(1.0), Const(0.0)
            )
    elif isinstance(node, ast.Call):
        return translate_call(node)
    elif isinstance(node, ast.IfExp):
        return IfNode(
            CmpOp("GT", translate(node.test), Const(0.0)),
            translate(node.body), translate(node.orelse)
        )
    # ... etc
```

### Stage 4: Bytecode Generation (Existing `FluxCompiler.compile_expr()`)

This stage is **already implemented** in `swarm/flux_compiler.py`. The `FluxCompiler` class:
1. Walks the `FluxCompiler` AST
2. Emits opcodes via `BytecodeEmitter`
3. Resolves labels for forward/backward jumps
4. Produces a `bytes` object + constant pool

**Key feature:** `prefer_range_check` flag
- `True` (default): `RangeCheckNode` → single `RangeCheck` opcode (5 bytes)
- `False`: `RangeCheckNode` → arithmetic + `CondJump` sequence (15+ bytes, but debuggable)

### Stage 5: VM Loading & Execution

**Python fallback path:**
```python
from swarm.flux_vm_runner import FluxVMRunner
runner = FluxVMRunner(const_pool)
result = runner.run(bytecode)  # float: 1.0 = pass, 0.0 = fail
```

**Rust VM path:**
```python
from sunset.flux_vm_bridge import FluxVMBridge
bridge = FluxVMBridge()
bridge.new()
bridge.load_bytecode(bytecode)
bridge.load_constraint(lo, hi)  # for BatchCheck/RangeCheck
for value in room_values:
    bridge.push_value(int(value * scale))  # fixed-point
passed = bridge.run()  # bool
proof_hash = bridge.get_proof_hash()  # 32 bytes
```

---

## 5. FFI Interface: Python → Rust VM

### 5.1 Already Exported (from `flux_vm/ffi.py` and `sunset/flux_vm_bridge.py`)

```c
// VM lifecycle
void* flux_vm_new(void);
void  flux_vm_free(void* vm);
int   flux_vm_reset(void* vm);

// Bytecode loading
int flux_vm_load_bytecode(void* vm, uint8_t* bytecode, uint32_t len);

// Constraint pre-loading (for RangeCheck / BatchCheck)
int flux_vm_load_constraint(void* vm, int32_t lo, int32_t hi);

// Stack loading
int flux_vm_push_value(void* vm, int32_t value);

// Execution
int flux_vm_run(void* vm);  // returns 1=pass, 0=fail, <0=error

// Result extraction
int flux_vm_get_result(void* vm, uint64_t* cycles, int* pass_flag);

// Proof certificate
int flux_vm_get_proof(void* vm, uint8_t* buf, uint32_t buf_len);  // returns 32 on success

// Provenance
int flux_vm_get_provenance_len(void* vm);
```

### 5.2 Still Needed (for full Path B)

| Function | Signature | Why Needed |
|----------|-----------|------------|
| `flux_vm_load_const_pool` | `int(void* vm, float* pool, uint32_t n)` | Currently only `Push` with immediates works. `LoadConst` needs a constant pool loaded into VM memory. |
| `flux_vm_set_register` | `int(void* vm, uint8_t reg, float value)` | `LoadReg`/`StoreReg` opcodes need register file access from Python. |
| `flux_vm_get_stack_top` | `int(void* vm, float* out)` | Debug: peek at stack after execution. |
| `flux_vm_step` | `int(void* vm)` | Single-step execution for debugging (already in Rust `vm.rs`, not exported). |
| `flux_vm_get_cycles` | `uint64_t(void* vm)` | Cycle counter read (currently only via `get_result`). |

---

## 6. Example: `lambda x: x > 0 and x < 100` → FLUX Bytecode

### 6.1 Python Source

```python
constraint = lambda x: x > 0 and x < 100
```

### 6.2 AST (simplified)

```python
ast.BoolOp(
    op=ast.And(),
    values=[
        ast.Compare(
            left=ast.Name(id='x', ctx=ast.Load()),
            ops=[ast.Gt()],
            comparators=[ast.Constant(value=0)]
        ),
        ast.Compare(
            left=ast.Name(id='x', ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[ast.Constant(value=100)]
        )
    ]
)
```

### 6.3 Normalized AST (after flattening)

```python
IfNode(
    cond=CmpOp("GT", Var("x"), Const(0.0)),  # x > 0
    then_expr=IfNode(
        cond=CmpOp("LT", Var("x"), Const(100.0)),  # x < 100
        then_expr=Const(1.0),   # pass
        else_expr=Const(0.0)    # fail (x ≥ 100)
    ),
    else_expr=Const(0.0)        # fail (x ≤ 0)
)
```

### 6.4 Bytecode (low-level path, `prefer_range_check=False`)

```text
; lambda x: x > 0 and x < 100
; Stack: [x] (pushed by caller before run)

Offset  Opcode        Operand(s)        Comment
────────────────────────────────────────────────────────────────
0000    LoadConst     0                 ; load x from const pool slot 0
0002    Push          0.0               ; compare constant
0007    Sub                             ; TOS = x - 0 = x
0008    CondJump      +14               ; if x ≤ 0, jump to fail_0
0011    LoadConst     0                 ; reload x
0013    Push          100.0             ; upper bound
0018    Sub                             ; TOS = x - 100
0019    CondJump      +7                ; if x - 100 ≤ 0 (i.e., x ≤ 100), jump to pass
                                        ; wait — this is wrong for "x < 100"
                                        ; correction: x < 100 means x - 100 < 0
                                        ; CondJump jumps if TOS ≤ 0, so this jumps on x ≤ 100
                                        ; We need to invert: jump if x - 100 ≥ 0 (x ≥ 100) to fail

; CORRECTED sequence for "x < 100":
0011    LoadConst     0                 ; x
0013    Push          100.0             ; 100
0018    Swap                            ; stack: [100, x]
0019    Sub                             ; TOS = 100 - x
0020    CondJump      +7                ; if 100 - x ≤ 0 (i.e., x ≥ 100), jump to fail_100
0023    Push          1.0               ; pass: x > 0 and x < 100
0028    Halt

0029    CondJump target: fail_0
0029    Push          0.0               ; fail: x ≤ 0
0034    ClassifySeverity 2              ; CRITICAL
0036    Validate                        ; trap if TOS == 0
0037    Halt

0038    CondJump target: fail_100
0038    Push          0.0               ; fail: x ≥ 100
0043    ClassifySeverity 2              ; CRITICAL
0045    Validate                        ; trap if TOS == 0
0046    Halt
```

### 6.5 Bytecode (high-level path, `prefer_range_check=True`)

```text
; Optimized: single RangeCheck opcode

Offset  Opcode        Operand(s)        Comment
────────────────────────────────────────────────────────────────
0000    LoadConst     0                 ; load x from const pool slot 0
0002    RangeCheck    0.0, 100.0        ; push 1.0 if 0 ≤ x ≤ 100, else 0.0
0011    Validate                        ; trap if TOS == 0
0012    Halt
```

### 6.6 Python Compilation Code

```python
from sunset.flux_compiler_frontend import compile_lambda
from sunset.flux_vm_bridge import FluxVMBridge

# 1. Compile
bytecode, const_pool, disasm = compile_lambda(
    "lambda x: x > 0 and x < 100",
    prefer_range_check=True,   # emit single RangeCheck
    with_validate=True,        # trap on failure
    with_halt=True,
)

# 2. Run in Python fallback
from swarm.flux_vm_runner import FluxVMRunner
runner = FluxVMRunner(const_pool)
result = runner.run(bytecode)  # 1.0 = pass, 0.0 = fail (but Validate traps on fail)

# 3. Run in Rust VM
bridge = FluxVMBridge()
bridge.new()
bridge.load_bytecode(bytecode)
bridge.load_constraint(0, 100)  # for RangeCheck
bridge.push_value(50)           # x = 50
passed = bridge.run()         # True
proof = bridge.get_proof()    # FluxVMProof with SHA-256 hash
```

---

## 7. Module Design: `sunset/flux_ast_compiler.py`

```python
"""Python ast → FLUX bytecode compiler.

The missing link: takes real Python expressions (lambda, function AST)
and produces FLUX VM bytecode via the existing FluxCompiler infrastructure.
"""

from __future__ import annotations

import ast
import inspect
from typing import Callable, Tuple, List

from swarm.flux_compiler import (
    FluxCompiler,
    BytecodeEmitter,
    Const, Var, BinOp, UnaryOp, RangeCheckNode,
    IfNode, CmpOp,
    Expr,
)


class FluxCompileError(Exception):
    """Raised when a Python construct cannot be compiled to FLUX."""
    pass


class PythonASTAdapter:
    """Translate Python ast.AST into FluxCompiler's internal AST."""

    def __init__(self, var_defaults: dict[str, float] | None = None):
        self.var_defaults = var_defaults or {}

    def translate(self, node: ast.AST) -> Expr:
        """Translate a Python AST node to a FluxCompiler Expr."""
        if isinstance(node, ast.Constant):
            return Const(float(node.value))
        elif isinstance(node, ast.Name):
            return Var(node.id)
        elif isinstance(node, ast.BinOp):
            return self._binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self._unaryop(node)
        elif isinstance(node, ast.Compare):
            return self._compare(node)
        elif isinstance(node, ast.BoolOp):
            return self._boolop(node)
        elif isinstance(node, ast.IfExp):
            return self._ifexp(node)
        elif isinstance(node, ast.Call):
            return self._call(node)
        else:
            raise FluxCompileError(
                f"Unsupported Python AST node: {type(node).__name__}"
            )

    def _binop(self, node: ast.BinOp) -> Expr:
        op_map = {
            ast.Add: "Add", ast.Sub: "Sub", ast.Mult: "Mul", ast.Div: "Div",
            ast.Mod: "Mod",  # will raise if not in PYTHON_SAFE_OPCODES
        }
        op = op_map.get(type(node.op))
        if op is None:
            raise FluxCompileError(f"Unsupported binary operator: {type(node.op).__name__}")
        return BinOp(op, self.translate(node.left), self.translate(node.right))

    def _compare(self, node: ast.Compare) -> Expr:
        # Flatten chain: a < b < c → a < b and b < c
        if len(node.ops) == 1:
            return self._single_compare(node.ops[0], node.left, node.comparators[0])
        else:
            # Chain: combine with And
            left = self._single_compare(node.ops[0], node.left, node.comparators[0])
            for i in range(1, len(node.ops)):
                right = self._single_compare(
                    node.ops[i], node.comparators[i-1], node.comparators[i]
                )
                left = BinOp("And", left, right)  # And is not a real opcode — handled in BoolOp
            return left

    def _single_compare(self, op: ast.cmpop, left: ast.AST, right: ast.AST) -> Expr:
        cmp_map = {
            ast.Lt: "LT", ast.LtE: "LE", ast.Gt: "GT", ast.GtE: "GE",
            ast.Eq: "EQ", ast.NotEq: "NE",
        }
        op_str = cmp_map.get(type(op))
        if op_str is None:
            raise FluxCompileError(f"Unsupported comparison: {type(op).__name__}")
        return CmpOp(op_str, self.translate(left), self.translate(right))

    def _boolop(self, node: ast.BoolOp) -> Expr:
        # And/Or → IfNode (short-circuit)
        if isinstance(node.op, ast.And):
            # a and b and c → if a then (if b then c else 0) else 0
            result: Expr = self.translate(node.values[-1])
            for val in reversed(node.values[:-1]):
                result = IfNode(
                    CmpOp("GT", self.translate(val), Const(0.0)),
                    result, Const(0.0)
                )
            return result
        elif isinstance(node.op, ast.Or):
            result = self.translate(node.values[-1])
            for val in reversed(node.values[:-1]):
                result = IfNode(
                    CmpOp("GT", self.translate(val), Const(0.0)),
                    Const(1.0), result
                )
            return result
        else:
            raise FluxCompileError(f"Unsupported boolean op: {type(node.op).__name__}")

    def _unaryop(self, node: ast.UnaryOp) -> Expr:
        if isinstance(node.op, ast.USub):
            return BinOp("Mul", Const(-1.0), self.translate(node.operand))
        elif isinstance(node.op, ast.UAdd):
            return self.translate(node.operand)
        elif isinstance(node.op, ast.Not):
            return IfNode(
                CmpOp("LE", self.translate(node.operand), Const(0.0)),
                Const(1.0), Const(0.0)
            )
        else:
            raise FluxCompileError(f"Unsupported unary op: {type(node.op).__name__}")

    def _ifexp(self, node: ast.IfExp) -> Expr:
        return IfNode(
            CmpOp("GT", self.translate(node.test), Const(0.0)),
            self.translate(node.body), self.translate(node.orelse)
        )

    def _call(self, node: ast.Call) -> Expr:
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname == "abs" and len(node.args) == 1:
                return UnaryOp("Abs", self.translate(node.args[0]))
            elif fname == "min" and len(node.args) == 2:
                return BinOp("Min", self.translate(node.args[0]), self.translate(node.args[1]))
            elif fname == "max" and len(node.args) == 2:
                return BinOp("Max", self.translate(node.args[0]), self.translate(node.args[1]))
            elif fname == "saturate" and len(node.args) == 3:
                return RangeCheckNode(
                    self.translate(node.args[0]),
                    self._extract_const(node.args[1]),
                    self._extract_const(node.args[2]),
                )
        raise FluxCompileError(f"Unsupported call: {ast.dump(node.func)}")

    def _extract_const(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value)
        raise FluxCompileError("Expected constant")


def compile_lambda(
    source: str,
    *,
    prefer_range_check: bool = True,
    with_validate: bool = True,
    var_defaults: dict[str, float] | None = None,
) -> Tuple[bytes, List[float], List[str]]:
    """Compile a Python lambda string to FLUX bytecode.

    Returns (bytecode, constant_pool, disassembly).

    Example::
        bc, pool, asm = compile_lambda("lambda x: x > 0 and x < 100")
    """
    tree = ast.parse(source, mode="eval")
    lam = tree.body
    if not isinstance(lam, ast.Lambda):
        raise FluxCompileError("Source must be a lambda expression")

    adapter = PythonASTAdapter(var_defaults)
    expr = adapter.translate(lam.body)

    compiler = FluxCompiler(prefer_range_check=prefer_range_check)
    emitter = compiler.compile_constraint(expr, with_validate=with_validate, with_halt=True)
    return emitter.to_bytes(), emitter.const_pool, emitter.disassemble()


def compile_function(
    func: Callable,
    *,
    prefer_range_check: bool = True,
    with_validate: bool = True,
) -> Tuple[bytes, List[float], List[str]]:
    """Compile a Python function to FLUX bytecode.

    Only the function body is compiled. The function must consist of
    a single return statement with a constraint expression.

    Example::
        def check(x):
            return x > 0 and x < 100
        bc, pool, asm = compile_function(check)
    """
    source = inspect.getsource(func)
    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise FluxCompileError("Expected a function definition")
    if len(func_def.body) != 1 or not isinstance(func_def.body[0], ast.Return):
        raise FluxCompileError("Function must have exactly one return statement")

    adapter = PythonASTAdapter()
    expr = adapter.translate(func_def.body[0].value)

    compiler = FluxCompiler(prefer_range_check=prefer_range_check)
    emitter = compiler.compile_constraint(expr, with_validate=with_validate, with_halt=True)
    return emitter.to_bytes(), emitter.const_pool, emitter.disassemble()
```

---

## 8. Integration with Existing Fleet Infrastructure

### 8.1 Breeding Pipeline (`swarm/breeder_daemon_v2.py`)

Currently `select_parents()` calls `_check_flux()` which uses `FluxVMGatingChecker`. With the AST compiler, the breeder can accept Python lambdas as constraints:

```python
from sunset.flux_compiler_frontend import compile_lambda
from swarm.flux_vm_gating import FluxVMGatingChecker

# Before: hardcoded constraints in FluxGatingConfig
# After: user-defined Python lambda
constraint = "lambda w: w > -5.0 and w < 5.0 and abs(w) < 3.0"
bc, pool, _ = compile_lambda(constraint)

checker = FluxVMGatingChecker()
checker.load_bytecode(bc)  # new method: load pre-compiled bytecode
# ... rest of check_candidate unchanged
```

### 8.2 FLUX Preset Library (`swarm/flux_preset_library.py`)

Presets can be expressed as Python lambdas and compiled to bytecode at module load time:

```python
PRESETS = {
    "tight_bounds": compile_lambda("lambda w: -1.0 < w < 1.0"),
    "thermal_safe": compile_lambda("lambda w, t: abs(w) < 5.0 and t < 0.95"),
    # ...
}
```

### 8.3 Opcode Capability Index (`logos/opcode_capability_index.py`)

The compiler should check `PYTHON_SAFE_OPCODES` before emitting any opcode. If an opcode is not in the safe set, the compiler should either:
- Raise `FluxCompileError` (strict mode)
- Emit a warning and use the Python fallback (lenient mode)

---

## 9. Implementation Plan

### Phase 1: AST Adapter (1-2 hours, direct build)
- [ ] Create `sunset/flux_ast_compiler.py` with `PythonASTAdapter`
- [ ] Implement all node translators (`_binop`, `_compare`, `_boolop`, `_unaryop`, `_call`, `_ifexp`)
- [ ] Add `compile_lambda()` and `compile_function()` convenience APIs
- [ ] Write tests: `tests/test_flux_ast_compiler.py` (10+ tests)

### Phase 2: Integration Wiring (1-2 hours, direct build)
- [ ] Wire `FluxVMGatingChecker` to accept pre-compiled bytecode
- [ ] Add `load_bytecode()` method to `FluxVMGatingChecker`
- [ ] Update `FluxPresetLibrary` to use compiled lambdas
- [ ] Write tests: `tests/test_flux_ast_integration.py` (5+ tests)

### Phase 3: Rust FFI Completion (blocked on FM's cargo build)
- [ ] Export `flux_vm_load_const_pool()` from Rust FFI
- [ ] Export `flux_vm_set_register()` from Rust FFI
- [ ] Export `flux_vm_get_stack_top()` from Rust FFI
- [ ] Rebuild `libflux_vm_v3.so` with new exports

### Phase 4: Documentation & Examples (30 min, direct build)
- [ ] Add `docs/FLUX_PYTHON_COMPILER.md` (this document, committed)
- [ ] Add example in `examples/flux_compile_and_run.py`
- [ ] Update `README.md` with Path B usage

---

## 10. Design Decisions

### 10.1 Why `ast.parse()` instead of a custom parser?

Python already has a perfect parser. Reusing it means:
- Syntax errors are handled by Python (familiar tracebacks)
- We get constant folding, operator precedence, and comparison chaining for free
- Future Python features (walrus operator, pattern matching) are available automatically
- No grammar maintenance

### 10.2 Why two ASTs (Python `ast` → `FluxCompiler` AST → bytecode)?

Instead of emitting bytecode directly from Python `ast`, we translate to the existing `FluxCompiler` AST first. This is **not** waste — it provides:
- **Separation of concerns:** Python-specific logic (name resolution, defaults) lives in the adapter; VM-specific logic (stack effects, jump patching) lives in `FluxCompiler`
- **Reusability:** `FluxCompiler` already works. We don't rewrite it.
- **Testability:** We can test the adapter independently of the bytecode emitter
- **Extensibility:** Future frontends (JSON, YAML, GUARD) can target the same `FluxCompiler` AST

### 10.3 Why `CondJump` uses `≤ 0` semantics?

The FLUX VM defines `CondJump` as "jump if TOS ≤ 0". This is a signed comparison. For `x > 0`:
- We compute `0 - x` (or `x - 0` and use `FwdJump` fall-through)
- If `x ≤ 0`, the subtraction yields `≤ 0`, so `CondJump` fires → fail path
- If `x > 0`, the subtraction yields `> 0`, so `CondJump` falls through → continue

This is the same semantics as `JLE` (jump if less-or-equal) in x86 assembly. The compiler inverts comparisons by swapping operands or reversing jump targets.

### 10.4 Fixed-Point vs Floating-Point

The Rust VM uses `i32` internally (fixed-point with configurable scale). The Python compiler emits `f32` bytecode (`Push <f32>`). The VM bridge converts:
- Python float → `int(value * scale)` → `push_value(int32)`
- VM result → `float(result) / scale`

Scale defaults to 1000 in `FluxVMConfig`. This is consistent with the existing `flux_vm_gating.py` implementation.

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `ast.parse()` produces nodes we don't handle | Medium | Low | Clear `FluxCompileError` with node type and line number |
| Constant pool overflows 256 entries | Low | Medium | Use `LoadConst` only for variables; `Push` for all constants. Split large programs. |
| Jump offset exceeds u16 (65,535 bytes) | Low | Low | Constraints are small. If needed, add `FwdJump32` opcode. |
| Rust FFI `flux_vm_load_const_pool` missing | High | Medium | Phase 3 is blocked on FM. Meanwhile, Python fallback `FluxVMRunner` works. |
| `ast.BinOp` `And` not same as `ast.BoolOp` `And` | Medium | Low | `ast.BitAnd` (`&`) is unsupported. `ast.BoolOp` (`and`) is the only And. |
| Python `in` operator vs `ast.Compare` `In` | Low | Medium | `x in [0, 100]` → `RangeCheck`. `x in list` → unsupported (raise error). |

---

## 12. Conclusion

The fleet has 80% of Path B already built. What's missing is the **Python AST frontend** — a ~200-line adapter that translates `ast.parse()` output into the existing `FluxCompiler` AST. Once this adapter exists, any Python lambda or function can be compiled to FLUX bytecode and executed in the Rust VM with proof certificates.

**The design is ready. Implementation is Phase 1 + 2 (direct build, ~3 hours, no subagents needed).**

The Rust FFI completion (Phase 3) requires FM's cargo build, but the Python fallback (`FluxVMRunner`) provides full functionality until then.

**Next action:** Implement `sunset/flux_ast_compiler.py` and `tests/test_flux_ast_compiler.py`.

---

*Design document: `docs/DESIGN_FLUX_PYTHON_COMPILER.md`*  
*Related: `docs/FLUX_OPCODE_ALIGNMENT.md` (audit), `swarm/flux_compiler.py` (existing compiler), `sunset/flux_vm_bridge.py` (FFI bridge)*
