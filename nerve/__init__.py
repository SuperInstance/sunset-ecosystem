"""Nerve Fiber Architecture — Micro-models as sensory pathways.

Like putting on shoes — you feel every edge at first, then muscle memory
takes over and you stop noticing. Nerve fibers are the sensory interface
between raw signals and reasoning agents.
"""

from .fiber import NerveFiber, FiberState, SensoryTile
from .routing import RoutingLayer, Route, HebbianChannel
from .adaptation import AdaptationEngine, ShoeTracker
from .templates import AgentTemplate, TemplateRegistry, BUILTIN_TEMPLATES
from .topology import NerveTopology, TickResult

__all__ = [
    "NerveFiber",
    "FiberState",
    "SensoryTile",
    "RoutingLayer",
    "Route",
    "HebbianChannel",
    "AdaptationEngine",
    "ShoeTracker",
    "AgentTemplate",
    "TemplateRegistry",
    "BUILTIN_TEMPLATES",
    "NerveTopology",
    "TickResult",
]
