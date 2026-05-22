"""Built-in plugins for the superinstance-runtime event bus."""

from __future__ import annotations

__all__ = [
    "ConstraintCollector",
    "ConstraintSelector",
    "ConstraintCompiler",
    "PlatoCollector",
    "PlatoSelector",
    "PlatoCompiler",
]

try:
    from superinstance.plugins.constraint import (
        ConstraintCollector,
        ConstraintCompiler,
        ConstraintSelector,
    )
except ImportError:
    ConstraintCollector = None  # type: ignore[misc, assignment]
    ConstraintSelector = None  # type: ignore[misc, assignment]
    ConstraintCompiler = None  # type: ignore[misc, assignment]

try:
    from superinstance.plugins.plato import (
        PlatoCollector,
        PlatoCompiler,
        PlatoSelector,
    )
except ImportError:
    PlatoCollector = None  # type: ignore[misc, assignment]
    PlatoSelector = None  # type: ignore[misc, assignment]
    PlatoCompiler = None  # type: ignore[misc, assignment]
