# LESSON-15: FLUX Constraint Checking

**Domain:** flux
**Prerequisites:** [11, 14]
**Agent Templates:** [lore-keeper, distill-teacher]
**Estimated Ticks:** 400

---

## Concept
Stack machine, proof certificates, terminating by design. FLUX is the compiler's conscience.

FLUX (Formally-proven Lightweight Unified eXecution) is a stack-based virtual machine
that runs constraint-checking bytecode. Every operation has a proof certificate — a witness
that the operation preserved some invariant. This makes FLUX programs auditable and
terminating by construction.

The v3 runtime (flux-compiler/) is the canonical implementation. The v2 runtime
(flux-vm-v2/) is archived but loadable via the compat layer (flux_compat/compat.py).

Key concepts:
- **Stack machine**: All operations push/pop from a single stack — no registers, no memory aliasing
- **Constraint check**: Each instruction verifies preconditions before executing
- **Proof certificate**: A compact witness that the check passed
- **Termination**: The instruction set is designed so all programs halt (no loops, no recursion)

FLUX is used in the ecosystem to verify that agent code meets safety constraints
before deployment. An agent whose code fails FLUX checking cannot compile to COMPILED state.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from flux_compat.v3_module import Module, Instruction

# Build a simple FLUX module
mod = Module(
    version=3,
    constants=[42, 100],
    instructions=[
        Instruction(opcode="Push", operand=0),  # push const[0] = 42
        Instruction(opcode="Push", operand=1),  # push const[1] = 100
        Instruction(opcode="RangeCheck"),  # verify 42 < 100
    ],
    constraints=[],
    metadata={"agent_id": "demo-001"},
)

print(mod.disasm())
print(f"Bytecode length: {len(mod.to_bytecode())} bytes")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Design a FLUX program that verifies: (1) an agent's DNA vector has dimension 128,
(2) all entries are in [-1, 1], (3) the vector is normalized (L2 norm = 1.0).
What instructions would you need? Are they in the current opcode set?

---
**Next:** LESSON-16
