"""Generation runner — orchestrates one generation of the SUNSET lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sunset.agent import Agent, AgentPhase, ResourceBudget
from sunset.seed_bank import SeedBank
from sunset.sunset_documents import Epilogue, Onboarding, Summary
from sunset.tensor_archive import SunsetEntry, TensorArchive
from sunset.trinity_scorer import trinity_score


@dataclass
class GenerationReport:
    """Summary of a completed generation."""

    generation: int
    agents_spawned: int = 0
    agents_survived: int = 0
    agents_sunset: int = 0
    peak_score: float = 0.0
    mean_score: float = 0.0
    children_spawned: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def __repr__(self) -> str:
        return (
            f"GenerationReport(gen={self.generation}, "
            f"spawned={self.agents_spawned}, survived={self.agents_survived}, "
            f"sunset={self.agents_sunset}, peak={self.peak_score:.4f})"
        )


@dataclass
class EthosProfile:
    """Hardware / resource profile read from ethos room."""

    parallel_capacity: int = 4
    budget_per_agent: ResourceBudget = field(default_factory=ResourceBudget)
    survival_threshold: float = 0.01


class GenerationRunner:
    """Orchestrates one generation of the SUNSET lifecycle.

    1. Read ethos room → get hardware profile
    2. Spawn N agents (N = hardware parallel capacity)
    3. Each agent reads trinity rooms, competes for relevance
    4. Score trinity connections after time budget
    5. Sunset losers (they write their docs)
    6. Breed children from survivors
    7. Return GenerationReport
    """

    def __init__(
        self,
        seed_bank: Optional[SeedBank] = None,
        tensor_archive: Optional[TensorArchive] = None,
    ) -> None:
        self.seed_bank = seed_bank or SeedBank()
        self.tensor_archive = tensor_archive or TensorArchive()
        self._current_generation = 0

    def __repr__(self) -> str:
        return (
            f"GenerationRunner(gen={self._current_generation}, "
            f"seeds={len(self.seed_bank._entries)})"
        )

    def run_generation(
        self,
        ethos: Optional[EthosProfile] = None,
        trinity_scores: Optional[Dict[str, Tuple[float, float, float]]] = None,
        generation: Optional[int] = None,
    ) -> GenerationReport:
        """Run a single generation.

        Args:
            ethos: Hardware/resource profile. Defaults to EthosProfile().
            trinity_scores: Pre-computed (ethos, pathos, logos) per agent ID.
                In a real system agents compute their own; here we inject.
            generation: Generation number override.

        Returns:
            GenerationReport with lifecycle statistics.
        """
        ethos = ethos or EthosProfile()
        gen = generation if generation is not None else self._current_generation
        started = datetime.now(timezone.utc)

        # 1. Spawn N agents
        agents = self._spawn_agents(gen, ethos.parallel_capacity, ethos.budget_per_agent)
        if trinity_scores is None:
            trinity_scores = {}

        # 2. Agents compete — compute trinity scores
        for agent in agents:
            agent.advance(AgentPhase.COMPETING)
            scores = trinity_scores.get(agent.id)
            if scores:
                agent.trinity_score = trinity_score(*scores)

        # 3. Partition into survivors and sunset candidates
        survivors, sunsetting = self._partition(agents, ethos.survival_threshold)

        # 4. Sunset the losers — write their docs
        for agent in sunsetting:
            agent.advance(AgentPhase.SUNSETTING)
            self._sunset_agent(agent)

        # 5. Survivors breed
        children_count = 0
        for agent in survivors:
            agent.advance(AgentPhase.BREEDING)
            children_count += self._breed(agent, gen)

        # 6. Build report
        scores = [a.trinity_score for a in agents]
        report = GenerationReport(
            generation=gen,
            agents_spawned=len(agents),
            agents_survived=len(survivors),
            agents_sunset=len(sunsetting),
            peak_score=max(scores) if scores else 0.0,
            mean_score=sum(scores) / len(scores) if scores else 0.0,
            children_spawned=children_count,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
        )

        self._current_generation = gen + 1
        return report

    def _spawn_agents(
        self, generation: int, count: int, budget: ResourceBudget
    ) -> List[Agent]:
        """Spawn new agents for this generation."""
        agents: List[Agent] = []
        for _ in range(count):
            agent = Agent(
                generation=generation,
                resource_budget=budget,
            )
            agents.append(agent)
        return agents

    def _partition(
        self, agents: List[Agent], threshold: float
    ) -> Tuple[List[Agent], List[Agent]]:
        """Split agents into survivors and those to sunset."""
        survivors = [a for a in agents if a.trinity_score >= threshold]
        sunsetting = [a for a in agents if a.trinity_score < threshold]
        return survivors, sunsetting

    def _sunset_agent(self, agent: Agent) -> None:
        """Process a sunsetting agent: write docs, archive."""
        epilogue = Epilogue(
            agent_id=agent.id,
            what_i_tried=f"Agent {agent.id} competed in generation {agent.generation}",
            what_i_found=f"Trinity score: {agent.trinity_score:.4f}",
            why_not_relevant="Below survival threshold",
            peak_trinity_score=agent.trinity_score,
            generation=agent.generation,
        )
        summary = Summary(
            agent_id=agent.id,
            work_from_my_perspective="Competed but did not find sufficient relevance.",
        )

        entry = SunsetEntry(
            agent_id=agent.id,
            generation=agent.generation,
            parent_id=agent.parent_id,
            epilogue=epilogue,
            summary=summary,
            peak_trinity_score=agent.trinity_score,
            content_blob=f"gen={agent.generation} score={agent.trinity_score:.4f}",
        )
        self.tensor_archive.archive(entry)

        # Write onboarding for potential cross-pollination
        onboarding = Onboarding(
            agent_id=agent.id,
            letter_to_children=f"From {agent.id}: I scored {agent.trinity_score:.4f}.",
            variant="continuation",
            parent_id=agent.parent_id or agent.id,
            generation=agent.generation,
        )
        self.seed_bank.store(onboarding)

        agent.advance(AgentPhase.ASLEEP)

    def _breed(self, agent: Agent, generation: int) -> int:
        """Breed children from a surviving agent. Returns number of children."""
        for variant in ("continuation", "cross-pollination", "mutation"):
            onboarding = Onboarding(
                agent_id=agent.id,
                letter_to_children=f"From survivor {agent.id}: I thrived.",
                what_works="Maintained trinity connections above threshold.",
                variant=variant,
                parent_id=agent.id,
                generation=generation,
            )
            self.seed_bank.store(onboarding, relevance=agent.trinity_score, novelty=0.7)
        return 3


__all__ = ["GenerationRunner", "GenerationReport", "EthosProfile"]
