"""FLUX Bytecode Generator for Optimization Algorithms (Proposal 1).

Generates FLUX v3 Module objects for three NLopt gradient-free global
algorithms: DIRECT, ESCH, and CRS2-LM.  Each module is a self-contained
bytecode program with:

- Fixed-point scaling via constants
- Vector load/store for algorithm state
- RangeCheck / CallBounded / ParDispatch as required
- Validate + HashCommit + Seal proof certificate

Reference: docs/EXOTICA_NLOPT_RESEARCH_BRIEF.md (§2.2, §2.4)
"""

from __future__ import annotations

__all__ = [
    "FLUXOptimizerCodegen",
    "generate_direct_module",
    "generate_esch_module",
    "generate_crs2lm_module",
]

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from flux_compat.fixed_point_bridge import FixedPointBridge
from flux_compat.v3_module import Instruction, Module, ConstraintDef


# ── Opcode constants (mirrors sunset.flux_codegen.OpCode) ──
class _Op:
    Push = 0x01
    Pop = 0x02
    Dup = 0x03
    Swap = 0x04
    Over = 0x05
    Drop = 0x06
    LoadConst = 0x07
    Nop = 0x08
    Add = 0x09
    Sub = 0x0A
    Mul = 0x0B
    Div = 0x0C
    Saturate = 0x0D
    Min = 0x0E
    Max = 0x0F
    Abs = 0x10
    LoadReg = 0x11
    StoreReg = 0x12
    LoadRegVec = 0x13
    StoreRegVec = 0x14
    RangeCheck = 0x15
    BatchCheck = 0x16
    AccumulateMask = 0x17
    ClassifySeverity = 0x18
    Prove = 0x19
    QueryBackward = 0x1A
    Simplify = 0x1B
    Validate = 0x1C
    HashCommit = 0x1D
    Seal = 0x1E
    VecLoad = 0x1F
    VecStore = 0x20
    VecRangeCheck = 0x21
    VecMaskMerge = 0x22
    VecReduce = 0x23
    VecGather = 0x24
    FwdJump = 0x25
    CondJump = 0x26
    CallBounded = 0x27
    Ret = 0x28
    Halt = 0x29
    Checkpoint = 0x2A
    SetHandler = 0x2B
    EmitEvent = 0x2C
    Rollback = 0x2D
    GetResult = 0x2E
    ParDispatch = 0x2F
    ParMerge = 0x30
    ParBarrier = 0x31
    ParReduce = 0x32
    SnapRecord = 0x33
    SnapQuery = 0x34
    SnapHash = 0x35
    SnapVerify = 0x36
    StreamOpen = 0x37
    StreamCheck = 0x38
    StreamBatch = 0x39
    StreamClose = 0x3A


# Mapping from numeric opcode to mnemonic string
_OPCODE_NAME = {v: k for k, v in vars(_Op).items() if not k.startswith("_")}


def _i(opcode: int, operand: Optional[int] = None) -> Instruction:
    """Build a v3 Instruction from a numeric opcode."""
    name = _OPCODE_NAME.get(opcode, "Nop")
    return Instruction(opcode=name, operand=operand)


# ── helper: proof certificate tail ─────────────────────────


def _proof_tail() -> List[Instruction]:
    """Standard proof-carrying suffix: Validate, HashCommit, Seal."""
    return [
        _i(_Op.Validate),
        _i(_Op.HashCommit),
        _i(_Op.Seal),
    ]


# ── helper: fixed-point constants block ─────────────────────


def _fp_constants(bridge: FixedPointBridge, *values: float) -> List[int]:
    """Encode a sequence of floats as fixed-point constants."""
    return [bridge.encode(v) for v in values]


# ────────────────────────────────────────────────────────────
#  DIRECT  (Dividing RECTangles)
# ────────────────────────────────────────────────────────────


def generate_direct_module(
    dim: int,
    bounds: List[Tuple[float, float]],
    maxeval: int,
    ftol: float,
    frac_bits: int = 16,
) -> Module:
    """Generate a FLUX module for the DIRECT global optimization algorithm.

    The produced bytecode sets up:
      1. Load rectangle lower/upper bounds as vectors
      2. Evaluate center via CallBounded (objective function bridge)
      3. RangeCheck on rectangle size vs ftol
      4. VecStore of subdivided rectangles
      5. Global best tracking (Min + StoreReg)
      6. Proof certificate (Validate + HashCommit + Seal)

    Parameters
    ----------
    dim:
        Problem dimension (number of variables).
    bounds:
        List of ``(lower, upper)`` pairs, length = dim.
    maxeval:
        Maximum function evaluations (becomes CallBounded cycle limit).
    ftol:
        Relative function-value tolerance.
    frac_bits:
        Fixed-point fractional bits.

    Returns
    -------
    flux.Module
        A v3 module ready for VM ingestion or compilation.
    """
    if len(bounds) != dim:
        raise ValueError(f"bounds length ({len(bounds)}) must equal dim ({dim})")

    # Build fixed-point bridge from boundary-corner pilot
    pilot = []
    for lo, hi in bounds:
        pilot.extend([lo, hi, (lo + hi) / 2.0])
    bridge = FixedPointBridge.auto_scale(pilot, frac_bits=frac_bits)

    constants: List[int] = []
    instructions: List[Instruction] = []
    warnings: List[str] = []

    # ── metadata header ──
    instructions.append(_i(_Op.Push, maxeval))
    instructions.append(_i(_Op.StoreReg, 0))  # r0 = maxeval counter

    # ── fixed-point scale factor as constant ──
    scale_fp = bridge.encode(bridge.scale_factor)
    constants.append(scale_fp)
    instructions.append(_i(_Op.LoadConst, scale_fp))
    instructions.append(_i(_Op.StoreReg, 1))  # r1 = scale_factor

    # ── ftol as fixed-point constant ──
    ftol_fp = bridge.encode(ftol)
    constants.append(ftol_fp)

    # ── load rectangle bounds as vectors ──
    # VecLoad reads a vector from the stack (dim values) into vec reg 0
    for lo, hi in bounds:
        lo_fp = bridge.encode(lo)
        hi_fp = bridge.encode(hi)
        constants.extend([lo_fp, hi_fp])
        instructions.extend(
            [
                _i(_Op.LoadConst, lo_fp),
                _i(_Op.LoadConst, hi_fp),
            ]
        )
    # Push dimension for vector ops
    instructions.extend(
        [
            _i(_Op.Push, dim * 2),  # total bound values loaded
            _i(_Op.VecLoad),  # load into vec reg 0 (implicit)
        ]
    )

    # ── compute rectangle center (lo+hi)/2 via VecReduce ──
    # Simplified: average the bounds pair-wise
    instructions.extend(
        [
            _i(_Op.Push, dim),
            _i(_Op.VecReduce),  # sum of bounds → stack top
            _i(_Op.Push, 2),
            _i(_Op.Div),  # /2 for average
            _i(_Op.Saturate),  # clamp to fixed-point range
        ]
    )

    # ── evaluate objective at center via CallBounded ──
    # Cycle limit = maxeval; target = 0 (native objective bridge)
    instructions.extend(
        [
            _i(_Op.Push, 0),  # call target 0 = objective bridge
            _i(_Op.Push, maxeval),  # cycle limit
            _i(_Op.CallBounded),
        ]
    )

    # ── check rectangle size vs ftol ──
    # Push ftol constant, then RangeCheck on rectangle width
    instructions.extend(
        [
            _i(_Op.LoadConst, ftol_fp),
            _i(_Op.RangeCheck),
            _i(_Op.Validate),
        ]
    )

    # ── store evaluated result and update global best ──
    instructions.extend(
        [
            _i(_Op.StoreReg, 2),  # r2 = current objective value
            _i(_Op.LoadReg, 2),
            _i(_Op.LoadReg, 3),  # r3 = best so far (0 initially)
            _i(_Op.Min),
            _i(_Op.StoreReg, 3),  # r3 = updated best
        ]
    )

    # ── VecStore of current center point ──
    instructions.extend(
        [
            _i(_Op.Push, dim),
            _i(_Op.VecStore),  # store vec reg 0 → memory
        ]
    )

    # ── proof certificate ──
    instructions.extend(_proof_tail())

    # ── result and halt ──
    instructions.extend(
        [
            _i(_Op.GetResult),
            _i(_Op.Halt),
        ]
    )

    # ── constraints metadata ──
    constraints = [
        ConstraintDef(
            kind="custom",
            params={
                "algorithm": "DIRECT",
                "dim": dim,
                "maxeval": maxeval,
                "ftol": ftol,
                "frac_bits": frac_bits,
                "scale_factor": bridge.scale_factor,
            },
        )
    ]

    return Module(
        version=3,
        constants=constants,
        instructions=instructions,
        constraints=constraints,
        metadata={
            "algorithm": "DIRECT",
            "dim": dim,
            "maxeval": maxeval,
            "ftol": ftol,
            "scale_factor": bridge.scale_factor,
            "resolution": bridge.resolution,
        },
        warnings=warnings,
    )


# ────────────────────────────────────────────────────────────
#  ESCH  (Evolutionary Strategy with Cauchy mutation)
# ────────────────────────────────────────────────────────────


def generate_esch_module(
    dim: int,
    bounds: List[Tuple[float, float]],
    pop_size: int,
    maxeval: int,
    frac_bits: int = 16,
) -> Module:
    """Generate a FLUX module for the ESCH evolutionary strategy.

    The produced bytecode sets up:
      1. Population vectors via VecLoad / VecStore into register bank
      2. Pre-generated Cauchy mutation constants in constant pool
      3. Mutation arithmetic: Add + Mul with constants
      4. Selection: Min over fitness register array
      5. Parallel eval: ParDispatch across population members
      6. Proof certificate (Validate + HashCommit + Seal)

    Parameters
    ----------
    dim:
        Problem dimension.
    bounds:
        List of ``(lower, upper)`` pairs, length = dim.
    pop_size:
        Population size (μ).
    maxeval:
        Maximum function evaluations.
    frac_bits:
        Fixed-point fractional bits.

    Returns
    -------
    flux.Module
    """
    if len(bounds) != dim:
        raise ValueError(f"bounds length ({len(bounds)}) must equal dim ({dim})")

    pilot = []
    for lo, hi in bounds:
        pilot.extend([lo, hi])
    # Add population-size headroom for mutation constants
    pilot.extend([1.0, -1.0] * min(pop_size, 10))
    bridge = FixedPointBridge.auto_scale(pilot, frac_bits=frac_bits)

    constants: List[int] = []
    instructions: List[Instruction] = []
    warnings: List[str] = []

    # ── init population size and eval counter ──
    instructions.extend(
        [
            _i(_Op.Push, pop_size),
            _i(_Op.StoreReg, 0),  # r0 = pop_size
            _i(_Op.Push, maxeval),
            _i(_Op.StoreReg, 1),  # r1 = remaining evals
        ]
    )

    # ── load population member 0 as template via VecLoad ──
    for lo, hi in bounds:
        lo_fp = bridge.encode(lo)
        hi_fp = bridge.encode(hi)
        constants.extend([lo_fp, hi_fp])
        instructions.extend(
            [
                _i(_Op.LoadConst, lo_fp),
                _i(_Op.LoadConst, hi_fp),
            ]
        )
    instructions.extend(
        [
            _i(_Op.Push, dim * 2),
            _i(_Op.VecLoad),
        ]
    )

    # ── Cauchy mutation constants (pre-truncated to [-M, M]) ──
    # Generate a small pool of mutation constants; 99% of Cauchy mass
    # within [-5, 5] for typical scales.
    M = 5.0
    mutation_constants = []
    for i in range(min(pop_size, 16)):
        # Pseudo-Cauchy sample: simple deterministic pattern
        val = M * math.sin((i + 1) * 1.618)  # golden-angle jitter
        val = max(-M, min(M, val))
        mutation_constants.append(val)

    for val in mutation_constants:
        v_fp = bridge.encode(val)
        constants.append(v_fp)
        instructions.extend(
            [
                _i(_Op.LoadConst, v_fp),
                _i(_Op.Push, dim),
                _i(_Op.VecStore),
            ]
        )

    # ── mutation: Add + Mul with constant ──
    instructions.extend(
        [
            _i(_Op.Push, dim),
            _i(_Op.VecLoad),  # load current individual
            _i(_Op.LoadConst, constants[-1] if constants else 0),
            _i(_Op.Mul),  # scale by mutation constant
            _i(_Op.Add),  # add to current individual
            _i(_Op.Saturate),  # clamp
        ]
    )

    # ── evaluate mutated individual via CallBounded ──
    instructions.extend(
        [
            _i(_Op.Push, 0),  # target 0 = objective bridge
            _i(_Op.Push, maxeval // max(pop_size, 1)),
            _i(_Op.CallBounded),
        ]
    )

    # ── selection: Min over fitness ──
    instructions.extend(
        [
            _i(_Op.StoreReg, 2),  # r2 = current fitness
            _i(_Op.LoadReg, 2),
            _i(_Op.LoadReg, 3),  # r3 = best fitness
            _i(_Op.Min),
            _i(_Op.StoreReg, 3),
        ]
    )

    # ── parallel dispatch across population ──
    instructions.extend(
        [
            _i(_Op.Push, pop_size),
            _i(_Op.ParDispatch),  # dispatch population evaluations
            _i(_Op.ParBarrier),  # wait for all
        ]
    )

    # ── proof certificate ──
    instructions.extend(_proof_tail())

    instructions.extend(
        [
            _i(_Op.GetResult),
            _i(_Op.Halt),
        ]
    )

    constraints = [
        ConstraintDef(
            kind="custom",
            params={
                "algorithm": "ESCH",
                "dim": dim,
                "pop_size": pop_size,
                "maxeval": maxeval,
                "frac_bits": frac_bits,
                "scale_factor": bridge.scale_factor,
            },
        )
    ]

    return Module(
        version=3,
        constants=constants,
        instructions=instructions,
        constraints=constraints,
        metadata={
            "algorithm": "ESCH",
            "dim": dim,
            "pop_size": pop_size,
            "maxeval": maxeval,
            "scale_factor": bridge.scale_factor,
            "resolution": bridge.resolution,
        },
        warnings=warnings,
    )


# ────────────────────────────────────────────────────────────
#  CRS2-LM  (Controlled Random Search)
# ────────────────────────────────────────────────────────────


def generate_crs2lm_module(
    dim: int,
    bounds: List[Tuple[float, float]],
    pop_size: int,
    frac_bits: int = 16,
) -> Module:
    """Generate a FLUX module for the CRS2-LM controlled random search.

    The produced bytecode sets up:
      1. Simplex of 2*dim points via VecLoad / VecStore
      2. Centroid computation via VecReduce (mean)
      3. Reflection: Sub + Mul + Add
      4. RangeCheck on reflected point against bounds
      5. Proof certificate (Validate + HashCommit + Seal)

    Parameters
    ----------
    dim:
        Problem dimension.
    bounds:
        List of ``(lower, upper)`` pairs, length = dim.
    pop_size:
        Population / simplex size (typically 2*dim).
    frac_bits:
        Fixed-point fractional bits.

    Returns
    -------
    flux.Module
    """
    if len(bounds) != dim:
        raise ValueError(f"bounds length ({len(bounds)}) must equal dim ({dim})")

    pilot = []
    for lo, hi in bounds:
        pilot.extend([lo, hi])
    bridge = FixedPointBridge.auto_scale(pilot, frac_bits=frac_bits)

    constants: List[int] = []
    instructions: List[Instruction] = []
    warnings: List[str] = []

    # ── init simplex size ──
    simplex_size = max(pop_size, dim * 2)
    instructions.extend(
        [
            _i(_Op.Push, simplex_size),
            _i(_Op.StoreReg, 0),  # r0 = simplex_size
        ]
    )

    # ── load bounds as vectors ──
    for lo, hi in bounds:
        lo_fp = bridge.encode(lo)
        hi_fp = bridge.encode(hi)
        constants.extend([lo_fp, hi_fp])
        instructions.extend(
            [
                _i(_Op.LoadConst, lo_fp),
                _i(_Op.LoadConst, hi_fp),
            ]
        )
    instructions.extend(
        [
            _i(_Op.Push, dim * 2),
            _i(_Op.VecLoad),
            _i(_Op.VecStore),  # store bounds vec for later RangeCheck
        ]
    )

    # ── load simplex points via VecLoad ──
    # Each point has dim coordinates; load one point at a time
    for i in range(min(simplex_size, 8)):  # cap at 8 for bytecode size
        for d in range(dim):
            lo, hi = bounds[d]
            # Deterministic initial simplex: linear interpolation
            t = (i + 1) / (simplex_size + 1)
            val = lo + t * (hi - lo)
            val_fp = bridge.encode(val)
            constants.append(val_fp)
            instructions.append(_i(_Op.LoadConst, val_fp))
        instructions.extend(
            [
                _i(_Op.Push, dim),
                _i(_Op.VecLoad),
                _i(_Op.VecStore),  # store point i
            ]
        )

    if simplex_size > 8:
        warnings.append(
            f"simplex_size={simplex_size} > 8; truncated to 8 points in bytecode. "
            "Runtime loop should unroll remainder."
        )

    # ── centroid via VecReduce (mean) ──
    # Sum all simplex points then divide by simplex_size
    instructions.extend(
        [
            _i(_Op.Push, dim),
            _i(_Op.VecReduce),  # sum → stack
            _i(_Op.Push, simplex_size),
            _i(_Op.Div),  # mean
            _i(_Op.Saturate),
        ]
    )

    # ── reflection: centroid + alpha*(centroid - worst_point) ──
    # alpha = 1.5 as fixed-point constant
    alpha = 1.5
    alpha_fp = bridge.encode(alpha)
    constants.append(alpha_fp)
    instructions.extend(
        [
            _i(_Op.LoadConst, alpha_fp),
            _i(_Op.Mul),
            _i(_Op.Add),
            _i(_Op.Saturate),
        ]
    )

    # ── RangeCheck reflected point against bounds ──
    instructions.extend(
        [
            _i(_Op.Push, dim),
            _i(_Op.VecRangeCheck),
            _i(_Op.Validate),
        ]
    )

    # ── evaluate reflected point via CallBounded ──
    instructions.extend(
        [
            _i(_Op.Push, 0),  # target 0 = objective bridge
            _i(_Op.Push, simplex_size * 10),  # cycle limit
            _i(_Op.CallBounded),
            _i(_Op.StoreReg, 1),  # r1 = reflected fitness
        ]
    )

    # ── proof certificate ──
    instructions.extend(_proof_tail())

    instructions.extend(
        [
            _i(_Op.GetResult),
            _i(_Op.Halt),
        ]
    )

    constraints = [
        ConstraintDef(
            kind="custom",
            params={
                "algorithm": "CRS2-LM",
                "dim": dim,
                "pop_size": pop_size,
                "simplex_size": simplex_size,
                "frac_bits": frac_bits,
                "scale_factor": bridge.scale_factor,
            },
        )
    ]

    return Module(
        version=3,
        constants=constants,
        instructions=instructions,
        constraints=constraints,
        metadata={
            "algorithm": "CRS2-LM",
            "dim": dim,
            "pop_size": pop_size,
            "simplex_size": simplex_size,
            "scale_factor": bridge.scale_factor,
            "resolution": bridge.resolution,
        },
        warnings=warnings,
    )


# ────────────────────────────────────────────────────────────
#  Unified codegen class
# ────────────────────────────────────────────────────────────


@dataclass
class FLUXOptimizerCodegen:
    """High-level codegen facade for NLopt → FLUX optimization modules.

    Usage
    -----
        codegen = FLUXOptimizerCodegen(frac_bits=16)
        mod = codegen.direct(dim=4, bounds=[(0,1)]*4, maxeval=1000, ftol=1e-4)
        mod = codegen.esch(dim=4, bounds=[(0,1)]*4, pop_size=20, maxeval=5000)
        mod = codegen.crs2lm(dim=4, bounds=[(0,1)]*4, pop_size=8)
    """

    frac_bits: int = 16

    def direct(
        self,
        dim: int,
        bounds: List[Tuple[float, float]],
        maxeval: int,
        ftol: float,
    ) -> Module:
        return generate_direct_module(dim, bounds, maxeval, ftol, self.frac_bits)

    def esch(
        self,
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int,
        maxeval: int,
    ) -> Module:
        return generate_esch_module(dim, bounds, pop_size, maxeval, self.frac_bits)

    def crs2lm(
        self,
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int,
    ) -> Module:
        return generate_crs2lm_module(dim, bounds, pop_size, self.frac_bits)

    def list_algorithms(self) -> List[str]:
        return ["DIRECT", "ESCH", "CRS2-LM"]
