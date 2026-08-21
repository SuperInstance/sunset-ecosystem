#!/usr/bin/env python3
"""Distillation Demo — 12 agents, 5 generations, quality tracking.

First concrete distillation experiment for the Sunset Ecosystem.
Uses existing modules: swarm, distill, ranking, sunset.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

# ── Ensure the project root is on sys.path ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sunset.agent import Agent, AgentPhase
from swarm.penrose import assign_positions, minimum_overlap, PenrosePosition
from swarm.broadcast import BroadcastingChannel, BroadcastMessage
from distill.prompt_history import PromptHistory, PromptRecord
from distill.hint_schedule import ExponentialBackoffSchedule
from distill.delta_tracker import DeltaTracker
from distill.backtest_runner import BacktestRunner
from ranking.feedback_loop import FeedbackLoop
from ranking.user_ranking import UserRanking
from ranking.ranked_response import RankedResponse
from ranking.personalization import PersonalizationStore
from distill.distillation_signal import DistillationSignal

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
NUM_AGENTS = 12
NUM_GENERATIONS = 5
HINT_LEVELS = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0]
SEED_PROMPTS = [
    "Explain the PLATO architecture and how it handles trinity rooms.",
    "How does the SUNSET ecosystem manage agent lifecycle phases?",
    "Describe the BroadcastingChannel Hebbian strengthening mechanism.",
    "What is the Penrose lattice and why is it used for agent diversity?",
    "How does the FeedbackLoop connect user rankings to hint schedules?",
]
REFERENCE_OUTPUTS = {
    SEED_PROMPTS[0]: (
        "PLATO is a cognitive architecture using trinity rooms — ethos, logos, pathos — "
        "to balance hardware reality, codebase state, and emotional context. Each room "
        "maintains its own state and contributes to a unified agent worldview. "
        "Trinity connections allow rooms to share ground-truths-for-now, creating "
        "a dynamic, responsive system that adapts to changing conditions."
    ),
    SEED_PROMPTS[1]: (
        "SUNSET agents pass through five phases: incubating, competing, breeding, "
        "sunsetting, and asleep. Each phase has distinct behaviors. Incubating agents "
        "read trinity rooms to build context. Competing agents find relevance through "
        "scoring. Breeding agents spawn children. Sunsetting agents write epilogues. "
        "Asleep agents are archived but searchable."
    ),
    SEED_PROMPTS[2]: (
        "BroadcastingChannel uses Hebbian learning: when a broadcast message is found "
        "useful, the channel weight between source and subscriber increases. This "
        "strengthens pathways that produce valuable content, creating emergent "
        "communication patterns. Feedback reports strengthen or weaken channels, "
        "and the weight influences future message routing."
    ),
    SEED_PROMPTS[3]: (
        "The Penrose lattice uses golden-angle spacing to position agents on a "
        "sunflower-like spiral. This aperiodic distribution guarantees that no two "
        "agents see the problem from the same perspective. The Vogel model ensures "
        "unique angles and distances, maximizing cognitive diversity across the swarm."
    ),
    SEED_PROMPTS[4]: (
        "FeedbackLoop is the central nervous system of distillation. It takes user "
        "rankings, extracts preference tags via PersonalizationStore, generates "
        "DistillationGuidance via DistillationSignal, and adjusts the HintSchedule. "
        "When distilled responses beat big-model outputs, hints are reduced, moving "
        "the system toward autonomy."
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# Simulated Agent Response Generation
# ═══════════════════════════════════════════════════════════════════════
def generate_response(
    agent_idx: int,
    prompt: str,
    hint_level: int,
    position: PenrosePosition,
    gen: int,
) -> tuple[str, float]:
    """Simulate an agent response.

    Quality depends on hint_level, generation progress, position diversity, and noise.
    Later generations improve because distillation knowledge accumulates.
    """
    reference = REFERENCE_OUTPUTS.get(prompt, "The system operates as designed.")

    # Agents with LESS hints actually IMPROVE faster (they learn distillation)
    # This is the key insight: hints are crutches. Remove them = force learning.

    # Distillation learning rate
    if hint_level <= 5:
        learn_rate = 0.04 * (6 - hint_level)
    else:
        learn_rate = 0.005 * (11 - hint_level)

    # Past knowledge accumulates
    past_knowledge = learn_rate * gen

    # Current hint contribution (high at start, diminishes)
    hint_contribution = 0.02 * hint_level * max(0, 1.0 - 0.1 * gen)

    # Position diversity bonus
    diversity_bonus = 0.04 / (1 + position.ring)

    # Random noise (bounded)
    noise = random.gauss(0, 0.02)

    quality = max(
        0.2,
        min(0.98, 0.35 + past_knowledge + hint_contribution + diversity_bonus + noise),
    )

    # Build response: proportion of reference words covered
    ref_words = reference.split()
    n_words = max(8, int(len(ref_words) * quality))
    words = ref_words[:n_words]
    words.append(f"(ring-{position.ring}, hl={hint_level})")

    return " ".join(words), quality


def score_response(response: str, reference: str) -> float:
    """Jaccard word overlap score."""
    words_a = set(response.lower().split())
    words_b = set(reference.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


# ═══════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════
def run_experiment() -> str:
    """Run the distillation experiment and return the report."""
    random.seed(42)
    report_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        report_lines.append(msg)

    log("# Distillation Experiment Report — gen-report-001")
    log("")
    log(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"**Agents:** {NUM_AGENTS} | **Generations:** {NUM_GENERATIONS}")
    log(f"**Hint levels:** {HINT_LEVELS}")
    log("")

    # ── Create agents ────────────────────────────────────────────────
    log("## Setup")
    log("")
    agents: list[Agent] = []
    for i in range(NUM_AGENTS):
        agents.append(
            Agent(generation=0, phase=AgentPhase.INCUBATING, room=f"agent-{i:02d}")
        )

    agent_ids = [a.id for a in agents]

    # ── Penrose positions ────────────────────────────────────────────
    positions = assign_positions(agent_ids)
    agent_positions = dict(zip(agent_ids, positions))
    log(f"**Penrose positions:** {len(positions)} agents on golden-angle spiral")
    log(f"  Min overlap: {minimum_overlap(positions):.3f} (lower = more diverse)")
    log("")

    # ── Broadcasting channel ─────────────────────────────────────────
    channel = BroadcastingChannel()
    for a in agents:
        channel.subscribe(a.id, "research-results")
    log(
        f"**BroadcastingChannel:** {channel.subscription_count} subscriptions on 'research-results'"
    )
    log("")

    # ── Prompt history ───────────────────────────────────────────────
    history = PromptHistory()
    for i, prompt in enumerate(SEED_PROMPTS):
        history.add(
            PromptRecord(
                prompt=prompt,
                response=REFERENCE_OUTPUTS[prompt],
                seed=42 + i,
                model="big-model-reference",
                hint_level=10,
                quality_score=0.95,
                application="plato-architecture",
            )
        )
    log(f"**PromptHistory:** {history.count()} seed prompts loaded")
    log("")

    # ── Hint schedule ────────────────────────────────────────────────
    hint_schedule = ExponentialBackoffSchedule(max_hints=10)
    log(f"**HintSchedule:** {hint_schedule}")
    log("")

    # ── Supporting systems ───────────────────────────────────────────
    personalization = PersonalizationStore()
    backtester = BacktestRunner(history)
    signal = DistillationSignal(backtester)
    feedback_loop = FeedbackLoop(hint_schedule, personalization, signal)
    delta_tracker = DeltaTracker()

    # Per-agent hint levels (static assignments)
    agent_hl = dict(zip(agent_ids, HINT_LEVELS))

    # ── Run generations ──────────────────────────────────────────────
    log("## Generation Results")
    log("")
    log("| Gen | Hint Lvl | Avg Quality | Best Quality | Δ from prev |")
    log("|-----|----------|-------------|--------------|-------------|")

    for gen in range(NUM_GENERATIONS):
        prompt = SEED_PROMPTS[gen % len(SEED_PROMPTS)]
        reference = REFERENCE_OUTPUTS[prompt]

        # Broadcast the prompt
        channel.broadcast(
            BroadcastMessage(
                content=prompt,
                source_agent="orchestrator",
                target_room="research-results",
                relevance_score=1.0,
            )
        )

        # Each agent generates a response
        results: list[tuple[int, Agent, str, float]] = []
        for idx, a in enumerate(agents):
            pos = agent_positions[a.id]
            hl = max(0, agent_hl[a.id] - gen)  # hints decay with generation
            text, raw_q = generate_response(idx, prompt, hl, pos, gen)
            score = score_response(text, reference)
            results.append((idx, a, text, score))

        # Rank by score
        ranked = sorted(results, key=lambda x: x[3], reverse=True)

        avg_quality = sum(s for _, _, _, s in results) / len(results)
        best_quality = ranked[0][3]

        # Build UserRanking: mark first 4 agents (which have highest hint levels) as
        # "big model" and the remaining 8 as "distilled".
        # This tests: do low-hint distilled agents beat high-hint big model agents?
        ranked_responses = []
        for rank_i, (idx, agent, text, score) in enumerate(ranked):
            # Agents 0-3 have highest hint levels (10,9,8,7) → "big model ref"
            # Agents 4-11 have lower hints (6,5,4,3,2,1,0,0) → "distilled"
            is_big_model = idx < 4
            source = "big-model-ref" if is_big_model else f"distilled-gen{gen}"
            ranked_responses.append(
                RankedResponse(
                    response=text,
                    source=source,
                    rank=rank_i + 1,
                    hint_level=agent_hl[agent.id],
                )
            )

        notes = (
            "thorough detailed correct helpful"
            if best_quality > 0.5
            else "correct helpful concise"
        )
        ranking = UserRanking(
            prompt=prompt,
            responses=ranked_responses,
            user_notes=notes,
        )

        # Feed through feedback loop
        guidance = feedback_loop.ingest(ranking)

        # Record in delta tracker
        delta_tracker.record(
            generation=gen,
            avg_quality=avg_quality,
            hint_level=hint_schedule.current_level(),
        )

        # Delta from previous
        prev_avg = (
            delta_tracker.snapshots[-2].avg_quality
            if len(delta_tracker.snapshots) >= 2
            else avg_quality
        )
        delta = avg_quality - prev_avg

        log(
            f"| {gen:3d} | {hint_schedule.current_level():8d} | {avg_quality:11.4f} | {best_quality:12.4f} | {delta:+11.4f} |"
        )

        # Advance agent phases
        for a in agents:
            if a.phase == AgentPhase.INCUBATING and gen >= 1:
                a.advance(AgentPhase.COMPETING)
            elif a.phase == AgentPhase.COMPETING and gen >= 3:
                a.advance(AgentPhase.BREEDING)

    log("")

    # ── Summary ──────────────────────────────────────────────────────
    log("## Summary")
    log("")
    best_gen = delta_tracker.best_generation()
    trend = delta_tracker.trend()
    status = feedback_loop.get_status()

    log(
        f"- **Best generation:** Gen {best_gen.generation} (quality: {best_gen.avg_quality:.4f}, hints: {best_gen.hint_level})"
    )
    log(
        f"- **Quality trend:** {trend:+.4f} per step ({'improving ↑' if trend > 0 else 'declining ↓' if trend < 0 else 'stable →'})"
    )
    log(
        f"- **Regression detected:** {'Yes ⚠️' if delta_tracker.is_regression() else 'No ✓'}"
    )
    log(
        f"- **Revert recommended:** {'Yes ⚠️' if delta_tracker.should_revert() else 'No ✓'}"
    )
    log(f"- **Final hint level:** {status['hint_level']}")
    log(f"- **Autonomous:** {'Yes ✓' if status['is_autonomous'] else 'Not yet'}")
    log(f"- **Total rankings:** {status['total_rankings']}")
    log(f"- **Hint reduction rate:** {status['reduction_rate']:.1%}")
    log("")

    # Quality progression
    log("## Quality Progression")
    log("```")
    for snap in delta_tracker.snapshots:
        bar = "█" * int(snap.avg_quality * 40)
        log(
            f"  Gen {snap.generation}: {bar} {snap.avg_quality:.4f} (hints={snap.hint_level})"
        )
    log("```")
    log("")

    # Agent states
    log("## Agent Final States")
    log("")
    phase_counts: dict[str, int] = {}
    for a in agents:
        phase_counts[a.phase.value] = phase_counts.get(a.phase.value, 0) + 1
    for phase, count in sorted(phase_counts.items()):
        log(f"- {phase}: {count} agents")
    log("")

    # Learned preferences
    log("## Learned Preferences")
    log("")
    for tag, weight in personalization.get_top_preferences(5):
        log(f"- {tag}: {weight:.2f}")
    log("")

    # Hebbian weights
    log("## Hebbian Channel Weights (top 5)")
    log("")
    weights = []
    for i in range(NUM_AGENTS):
        for j in range(NUM_AGENTS):
            if i != j:
                w = channel.get_channel_weight(agent_ids[i], agent_ids[j])
                if w > 0:
                    weights.append((agent_ids[i][:6], agent_ids[j][:6], w))
    weights.sort(key=lambda x: x[2], reverse=True)
    for src, dst, w in weights[:5]:
        log(f"- {src} → {dst}: {w:.4f}")
    log("")

    # Per-agent detail
    log("## Per-Agent Detail")
    log("")
    log("| Agent | Hint Level | Ring | Phase |")
    log("|-------|------------|------|-------|")
    for idx, a in enumerate(agents):
        pos = agent_positions[a.id]
        log(
            f"| {idx:2d} ({a.id[:6]}) | {agent_hl[a.id]:10d} | {pos.ring:4d} | {a.phase.value} |"
        )
    log("")

    log("---")
    log(
        f"*Report generated by distillation_demo.py at {time.strftime('%Y-%m-%d %H:%M:%S')}*"
    )

    return "\n".join(report_lines)


if __name__ == "__main__":
    report = run_experiment()
    report_path = Path(__file__).resolve().parent / "gen-report-001.md"
    report_path.write_text(report + "\n")
    print(f"\n📄 Report saved to: {report_path}")
