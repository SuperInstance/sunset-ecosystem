"""fleet/deck.py — Presentation deck generation for breeding reports.

Cross-pollinated from ccc-os/deck.py.  Extended with fleet-specific
templates: breeding reports, FLUX gate decisions, fleet status snapshots.

Usage
-----
    from fleet.deck import breeding_report, fleet_status

    md = breeding_report(
        generation=42,
        pool_size=50,
        pass_rate=0.85,
        top_score=0.12,
        flux_gate_blocks=3,
        thermal_violations=0,
        proof_count=47,
    )
    print(md)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class Slide:
    title: str
    bullets: List[str]
    quote: Optional[str] = None


class Deck:
    """A presentation deck rendered as Markdown."""

    def __init__(self, title: str, deck_type: str):
        self.title = title
        self.deck_type = deck_type
        self.slides: list[Slide] = []
        self.meta = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "sunset-deck-system",
            "template": deck_type,
        }

    def add(self, slide: Slide) -> None:
        self.slides.append(slide)

    def render(self) -> str:
        lines = [
            f"# {self.title}",
            f"_Type: {self.deck_type} | Generated: {self.meta['generated_at'][:16]}_",
            "",
            "---",
            "",
        ]
        for i, slide in enumerate(self.slides, 1):
            lines.append(f"## Slide {i}: {slide.title}")
            lines.append("")
            for b in slide.bullets:
                lines.append(f"- {b}")
            if slide.quote:
                lines.append("")
                lines.append(f"> {slide.quote}")
            lines.append("")
            lines.append("---")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "deck_type": self.deck_type,
            "slides": [
                {"title": s.title, "bullets": s.bullets, "quote": s.quote}
                for s in self.slides
            ],
            "meta": self.meta,
        }


# ═══════════════════════════════════════════════════════════════
# Fleet-specific templates
# ═══════════════════════════════════════════════════════════════

def breeding_report(
    generation: int,
    pool_size: int,
    pass_rate: float,
    top_score: float,
    flux_gate_blocks: int,
    thermal_violations: int,
    proof_count: int,
    avg_cycles: int = 14,
) -> str:
    """Template: Breeding cycle report."""
    deck = Deck(f"Breeding Report — Generation {generation}", "breeding_report")
    deck.add(Slide("Pool", [
        f"Pool size: {pool_size}",
        f"Pass rate: {pass_rate:.1%}",
        f"Top score: {top_score:.3f}",
    ]))
    deck.add(Slide("FLUX Gating", [
        f"Candidates blocked: {flux_gate_blocks}",
        f"Proof certificates: {proof_count}",
        f"Avg VM cycles: {avg_cycles}",
    ]))
    deck.add(Slide("Thermal", [
        f"Violations: {thermal_violations}",
        "Status: Normal" if thermal_violations == 0 else "⚠️ Elevated pressure detected",
    ]))
    deck.add(Slide("Next", [
        "Continue breeding cycle" if pass_rate > 0.5 else "Review mutation parameters",
    ]))
    return deck.render()


def flux_gate_decision(
    candidate_id: str,
    passed: bool,
    score: float,
    violations: dict[str, float],
    proof_hash: str | None,
    vm_cycles: int,
) -> str:
    """Template: FLUX gate decision record."""
    deck = Deck(f"FLUX Gate — {candidate_id}", "flux_gate_decision")
    status = "✅ PASS" if passed else "❌ FAIL"
    deck.add(Slide("Decision", [f"Result: {status}", f"Score: {score:.3f}"]))
    if violations:
        deck.add(Slide("Violations", [
            f"{k}: {v:.3f}" for k, v in violations.items()
        ]))
    else:
        deck.add(Slide("Violations", ["None."]))
    if proof_hash:
        deck.add(Slide("Proof", [
            f"Hash: {proof_hash[:16]}...",
            f"VM cycles: {vm_cycles}",
        ]))
    return deck.render()


def fleet_status(
    services_up: int,
    services_down: int,
    breeding_active: bool,
    last_proof: str | None,
    blockers: List[str],
) -> str:
    """Template: Fleet status snapshot."""
    deck = Deck("Fleet Status", "fleet_status")
    total = services_up + services_down
    deck.add(Slide("Services", [
        f"{services_up}/{total} UP",
        f"{services_down} DOWN" if services_down else "All services operational",
    ]))
    deck.add(Slide("Breeding", [
        "Active" if breeding_active else "Idle",
        f"Last proof: {last_proof[:16]}..." if last_proof else "No proofs yet",
    ]))
    if blockers:
        deck.add(Slide("Blockers", blockers))
    else:
        deck.add(Slide("Blockers", ["None."]))
    return deck.render()


def architecture_decision(
    title: str,
    problem: str,
    options: str,
    recommendation: str,
    risk: str,
    timeline: str,
) -> str:
    """Template: Architecture decision deck."""
    deck = Deck(title, "architecture_decision")
    deck.add(Slide("The Problem", [problem]))
    deck.add(Slide("Options", [options]))
    deck.add(Slide("Recommendation", [recommendation]))
    deck.add(Slide("Risk", [risk]))
    deck.add(Slide("Timeline", [timeline]))
    return deck.render()


def research_summary(
    title: str,
    what_learned: str,
    why_matters: str,
    what_to_do: str,
) -> str:
    """Template: Research summary."""
    deck = Deck(title, "research_summary")
    deck.add(Slide("What We Learned", [what_learned]))
    deck.add(Slide("Why It Matters", [why_matters]))
    deck.add(Slide("What To Do", [what_to_do]))
    return deck.render()
