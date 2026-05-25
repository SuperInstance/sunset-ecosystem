"""FLUX VM Runner — production bytecode interpreter for FLUX constraints.

Extracted and hardened from ``MiniFluxVM`` in the test suite.
Executes FLUX bytecode compiled by ``FluxCompiler`` and returns
the top-of-stack result (or raises ``FluxTrap`` on Validate failure).
"""

from __future__ import annotations

import struct
from typing import List

from swarm.flux_compiler import FluxOpcode


class FluxTrap(Exception):
    """Raised when FLUX Validate opcode sees 0.0 on the stack."""
    pass


class FluxVMRunner:
    """Production bytecode interpreter for the PYTHON_SAFE FLUX opcode subset.

    Usage::

        runner = FluxVMRunner(const_pool)
        result = runner.run(bytecode)

    The runner is stateless aside from the constant pool — safe to reuse.
    """

    def __init__(self, const_pool: List[float]) -> None:
        self.const_pool = const_pool

    def run(self, bc: bytes) -> float:
        """Execute *bc* and return the top-of-stack value.

        Raises
        ------
        FluxTrap
            If a ``Validate`` opcode pops 0.0.
        ValueError
            On unknown opcode.
        """
        stack: List[float] = []
        i = 0
        while i < len(bc):
            op = bc[i]

            if op == FluxOpcode.Push:
                val = struct.unpack("<f", bc[i + 1 : i + 5])[0]
                stack.append(val)
                i += 5

            elif op == FluxOpcode.Pop:
                stack.pop()
                i += 1

            elif op == FluxOpcode.Dup:
                stack.append(stack[-1])
                i += 1

            elif op == FluxOpcode.Swap:
                a, b = stack.pop(), stack.pop()
                stack.extend([a, b])
                i += 1

            elif op == FluxOpcode.LoadConst:
                idx = bc[i + 1]
                stack.append(self.const_pool[idx])
                i += 2

            elif op == FluxOpcode.Add:
                b, a = stack.pop(), stack.pop()
                stack.append(a + b)
                i += 1

            elif op == FluxOpcode.Sub:
                b, a = stack.pop(), stack.pop()
                stack.append(a - b)
                i += 1

            elif op == FluxOpcode.Mul:
                b, a = stack.pop(), stack.pop()
                stack.append(a * b)
                i += 1

            elif op == FluxOpcode.Div:
                b, a = stack.pop(), stack.pop()
                stack.append(a / b if b != 0 else float("inf"))
                i += 1

            elif op == FluxOpcode.Min:
                b, a = stack.pop(), stack.pop()
                stack.append(min(a, b))
                i += 1

            elif op == FluxOpcode.Max:
                b, a = stack.pop(), stack.pop()
                stack.append(max(a, b))
                i += 1

            elif op == FluxOpcode.Abs:
                stack.append(abs(stack.pop()))
                i += 1

            elif op == FluxOpcode.Saturate:
                lo = struct.unpack("<f", bc[i + 1 : i + 5])[0]
                hi = struct.unpack("<f", bc[i + 5 : i + 9])[0]
                val = stack.pop()
                stack.append(max(lo, min(hi, val)))
                i += 9

            elif op == FluxOpcode.RangeCheck:
                lo = struct.unpack("<f", bc[i + 1 : i + 5])[0]
                hi = struct.unpack("<f", bc[i + 5 : i + 9])[0]
                val = stack.pop()
                stack.append(1.0 if lo <= val <= hi else 0.0)
                i += 9

            elif op == FluxOpcode.ClassifySeverity:
                # Pop and push back for stack balance
                val = stack.pop()
                stack.append(val)
                i += 2

            elif op == FluxOpcode.Validate:
                val = stack.pop()
                if val == 0.0:
                    raise FluxTrap("Validate failed")
                stack.append(val)
                i += 1

            elif op == FluxOpcode.FwdJump:
                off = struct.unpack("<H", bc[i + 1 : i + 3])[0]
                i += 3 + off

            elif op == FluxOpcode.CondJump:
                off = struct.unpack("<H", bc[i + 1 : i + 3])[0]
                val = stack.pop()
                if val <= 0:
                    i += 3 + off
                else:
                    i += 3

            elif op == FluxOpcode.Halt:
                break

            elif op == FluxOpcode.Nop:
                i += 1

            else:
                raise ValueError(f"Unhandled opcode 0x{op:02x} at offset {i}")

        return stack[-1] if stack else 0.0
