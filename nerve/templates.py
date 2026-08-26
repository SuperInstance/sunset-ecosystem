"""Agent Templates — Configuration presets for specialized agent breeds.

Each template defines the starting personality (ethos/pathos/logos bias),
exploration rate, and routing tags for a specific agent type.
"""

from __future__ import annotations

__all__ = [
    "AgentTemplate",
    "TemplateRegistry",
    "BUILTIN_TEMPLATES",
]

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class AgentTemplate:
    """Configuration preset for a specialized agent.

    Attributes:
        name: Unique identifier, e.g. "mud-expert", "arena-analyst".
        ethos_bias: Starting hardware-awareness [0, 1].
        pathos_bias: Starting human-empathy [0, 1].
        logos_bias: Starting code-reasoning [0, 1].
        input_projection: How the 64-dim signal is built from raw input.
        chaos_initial: Exploration rate at birth (0 = fully deterministic).
        hint_level: 10 = fully hinted, 0 = autonomous exploration.
        tags: Routing / filtering tags for the swarm query layer.
    """

    name: str
    ethos_bias: float = 0.5
    pathos_bias: float = 0.5
    logos_bias: float = 0.5
    input_projection: str = "identity"
    chaos_initial: float = 0.3
    hint_level: int = 5
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for axis, val in (
            ("ethos_bias", self.ethos_bias),
            ("pathos_bias", self.pathos_bias),
            ("logos_bias", self.logos_bias),
        ):
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{axis} must be in [0, 1], got {val}")
        if not (0.0 <= self.chaos_initial <= 1.0):
            raise ValueError(
                f"chaos_initial must be in [0, 1], got {self.chaos_initial}"
            )

    @property
    def trinity_signature(self) -> tuple[float, float, float]:
        """Return (ethos, pathos, logos) as a tuple."""
        return (self.ethos_bias, self.pathos_bias, self.logos_bias)

    @property
    def mean_bias(self) -> float:
        """Average of the three trinity axes."""
        return sum(self.trinity_signature) / 3.0

    def __repr__(self) -> str:
        return (
            f"AgentTemplate({self.name!r}, "
            f"E={self.ethos_bias:.2f} P={self.pathos_bias:.2f} L={self.logos_bias:.2f}, "
            f"chaos={self.chaos_initial:.2f}, hints={self.hint_level})"
        )


class TemplateRegistry:
    """In-memory store of named agent templates."""

    def __init__(self) -> None:
        self._templates: Dict[str, AgentTemplate] = {}
        for tmpl in BUILTIN_TEMPLATES:
            self.register(tmpl)

    def register(self, tmpl: AgentTemplate) -> None:
        """Add or overwrite a template."""
        self._templates[tmpl.name] = tmpl

    def get(self, name: str) -> AgentTemplate:
        """Retrieve a template by name. Raises KeyError if missing."""
        return self._templates[name]

    def list_names(self) -> List[str]:
        """Return sorted list of all registered template names."""
        return sorted(self._templates.keys())

    def filter_by_tag(self, tag: str) -> List[AgentTemplate]:
        """Return all templates that include *tag*."""
        return [t for t in self._templates.values() if tag in t.tags]

    def __contains__(self, name: str) -> bool:
        return name in self._templates

    def __repr__(self) -> str:
        return f"TemplateRegistry({len(self._templates)} templates)"


# ── Built-in Templates ──────────────────────────────────────────
# Per SPEC-BREEDER §2 "Built-in Templates"

BUILTIN_TEMPLATES: List[AgentTemplate] = [
    AgentTemplate(
        name="mud-expert",
        ethos_bias=0.3,
        pathos_bias=0.7,
        logos_bias=0.9,
        input_projection="mud_sparse",
        chaos_initial=0.3,
        hint_level=7,
        tags=["plato", "mud", "lore", "interactive"],
    ),
    AgentTemplate(
        name="arena-analyst",
        ethos_bias=0.5,
        pathos_bias=0.4,
        logos_bias=0.9,
        input_projection="combat_vector",
        chaos_initial=0.2,
        hint_level=4,
        tags=["pvp", "ranking", "analytical", "dispassionate"],
    ),
    AgentTemplate(
        name="lore-keeper",
        ethos_bias=0.8,
        pathos_bias=0.8,
        logos_bias=0.5,
        input_projection="narrative_embedding",
        chaos_initial=0.4,
        hint_level=8,
        tags=["history", "world-building", "values", "emotion"],
    ),
    AgentTemplate(
        name="distill-teacher",
        ethos_bias=0.6,
        pathos_bias=0.3,
        logos_bias=0.7,
        input_projection="curriculum_signal",
        chaos_initial=0.25,
        hint_level=6,
        tags=["education", "hint-schedule", "challenge", "pedagogy"],
    ),
    AgentTemplate(
        name="swarm-router",
        ethos_bias=0.3,
        pathos_bias=0.3,
        logos_bias=0.9,
        input_projection="task_allocation",
        chaos_initial=0.15,
        hint_level=3,
        tags=["routing", "allocation", "logic", "orchestration"],
    ),
    AgentTemplate(
        name="generic",
        ethos_bias=0.5,
        pathos_bias=0.5,
        logos_bias=0.5,
        input_projection="identity",
        chaos_initial=0.3,
        hint_level=5,
        tags=["default", "general-purpose"],
    ),
]
