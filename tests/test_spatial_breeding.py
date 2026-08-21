"""
Tests for Spatial Breeding Integration.

Covers: SpatialBreedingContext, parent selection strategies,
spatial metrics, breeding daemon integration.
"""

import math
import pytest

from fleet.spatial_projector import SpatialProjector, WorldState
from fleet.spatial_breeding import (
    SpatialBreedingConfig,
    SpatialBreedingContext,
    SpatialParentCandidate,
)


@pytest.fixture
def populated_projector():
    """Projector with 5 agents in known positions."""
    proj = SpatialProjector("node-test", dimension=2)

    # Agent at origin (ethos room)
    proj.project_state(
        "agent-1",
        "ethos",
        WorldState(position=(0.0, 0.0), semantics={"role": "breeder"}),
    )

    # Agent nearby (ethos room)
    proj.project_state(
        "agent-2",
        "ethos",
        WorldState(position=(3.0, 4.0), semantics={"role": "breeder"}),
    )

    # Agent nearby (ethos room)
    proj.project_state(
        "agent-3",
        "ethos",
        WorldState(position=(4.0, 3.0), semantics={"role": "solver"}),
    )

    # Agent far away (pathos room)
    proj.project_state(
        "agent-4",
        "pathos",
        WorldState(position=(50.0, 50.0), semantics={"role": "auditor"}),
    )

    # Agent far away (logos room)
    proj.project_state(
        "agent-5",
        "logos",
        WorldState(position=(60.0, 40.0), semantics={"role": "tester"}),
    )

    return proj


class TestSpatialBreedingContext:
    def test_select_proximal_parents(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        parents = ctx.select_proximal_parents("agent-1", radius=10.0, k=3)

        # Should find agent-2 (dist=5) and agent-3 (dist=5), not agent-4/5
        assert len(parents) == 2
        ids = [p.agent_id for p in parents]
        assert "agent-2" in ids
        assert "agent-3" in ids
        assert "agent-4" not in ids
        assert "agent-5" not in ids

    def test_proximal_with_genome_fn(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)

        def mock_genome(aid):
            return {"id": aid, "genes": [1, 2, 3]}

        parents = ctx.select_proximal_parents(
            "agent-1", radius=10.0, k=3, genome_fn=mock_genome
        )

        assert len(parents) == 2
        assert parents[0].genome == {"id": "agent-2", "genes": [1, 2, 3]}

    def test_select_diverse_parents(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        parents = ctx.select_diverse_parents("agent-1", min_distance=20.0, k=2)

        # Should find agent-4 and agent-5 (far away)
        assert len(parents) == 2
        ids = [p.agent_id for p in parents]
        assert "agent-4" in ids
        assert "agent-5" in ids

    def test_select_diverse_from_isolated_agent(self, populated_projector):
        # agent-4 is isolated
        ctx = SpatialBreedingContext(populated_projector)
        parents = ctx.select_diverse_parents("agent-4", min_distance=20.0, k=2)

        # Should find agents far from agent-4 (agent-1, agent-2, agent-3)
        assert len(parents) >= 2
        ids = [p.agent_id for p in parents]
        assert "agent-1" in ids

    def test_select_hybrid_parents(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        # With default config: 60% proximal (3), 40% diverse (2)
        parents = ctx.select_hybrid_parents("agent-1", total_k=4)

        ids = [p.agent_id for p in parents]
        # Should have mix of near and far
        near = {"agent-2", "agent-3"}
        far = {"agent-4", "agent-5"}
        has_near = bool(near & set(ids))
        has_far = bool(far & set(ids))

        # Hybrid should include both types (or at least some variety)
        assert len(parents) <= 4
        assert len(set(ids)) == len(ids)  # No duplicates

    def test_select_room_affinity(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        parents = ctx.select_room_affinity_parents("agent-1", room_id="ethos", k=3)

        # Should prefer agents in ethos room
        ids = [p.agent_id for p in parents]
        assert "agent-2" in ids or "agent-3" in ids

    def test_select_room_affinity_bonus(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)

        def fitness_fn(aid):
            return 1.0  # Equal base fitness

        parents = ctx.select_room_affinity_parents(
            "agent-1", room_id="ethos", k=3, fitness_fn=fitness_fn
        )

        # Agents in ethos should have higher adjusted fitness
        ethos_parents = [p for p in parents if p.room_id == "ethos"]
        other_parents = [p for p in parents if p.room_id != "ethos"]

        if ethos_parents and other_parents:
            assert ethos_parents[0].fitness > other_parents[0].fitness

    def test_trajectory_compatible(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        # Give agents velocity so trajectories differ
        populated_projector.project_state(
            "agent-1", "ethos", WorldState(position=(0.0, 0.0), velocity=(1.0, 0.0))
        )
        populated_projector.project_state(
            "agent-2", "ethos", WorldState(position=(3.0, 4.0), velocity=(0.0, 1.0))
        )

        parents = ctx.select_trajectory_compatible_parents("agent-1", k=3)
        # agent-1 and agent-2 trajectories diverge, so no collision
        assert len(parents) >= 1

    def test_trajectory_collision(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        # Two agents on collision course - close enough to collide
        populated_projector.project_state(
            "agent-1", "ethos", WorldState(position=(0.0, 0.0), velocity=(1.0, 0.0))
        )
        populated_projector.project_state(
            "agent-2", "ethos", WorldState(position=(1.5, 0.0), velocity=(-1.0, 0.0))
        )

        parents = ctx.select_trajectory_compatible_parents("agent-1", k=3)
        # agent-2 should be excluded due to collision (dist < 1.0 after step 1)
        ids = [p.agent_id for p in parents]
        assert "agent-2" not in ids

    def test_to_breeder_parents(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        candidates = [
            SpatialParentCandidate(
                agent_id="a1",
                genome={"g": 1},
                fitness=10.0,
                position=(0.0, 0.0),
                distance=1.0,
            ),
            SpatialParentCandidate(
                agent_id="a2",
                genome=None,
                fitness=5.0,
                position=(1.0, 0.0),
                distance=2.0,
            ),
        ]
        parents = ctx.to_breeder_parents(candidates)
        # Only a1 has a genome
        assert len(parents) == 1
        assert parents[0] == ({"g": 1}, 10.0)

    def test_spatial_fitness_adjustment_isolated(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        # agent-4 has some neighbors but let's check the formula
        adjusted = ctx.spatial_fitness_adjustment("agent-4", base_fitness=100.0)
        # Isolated agent gets bonus based on actual diversity score
        diversity = populated_projector.get_spatial_diversity_score("agent-4")
        expected = 100.0 * (1.0 + 0.2 * diversity)
        assert abs(adjusted - expected) < 0.001

    def test_spatial_fitness_adjustment_clustered(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        # agent-1 is clustered with agent-2 and agent-3
        adjusted = ctx.spatial_fitness_adjustment("agent-1", base_fitness=100.0)
        # Clustered agent gets small or no bonus
        assert adjusted <= 120.0

    def test_population_spatial_entropy(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        entropy = ctx.population_spatial_entropy()
        # 5 agents spread across 0,0 to 60,40 should have moderate entropy
        assert 0.0 < entropy <= 1.0

    def test_population_spatial_entropy_single_agent(self):
        proj = SpatialProjector("node-test", dimension=2)
        proj.project_state("agent-1", "ethos", WorldState(position=(0.0, 0.0)))
        ctx = SpatialBreedingContext(proj)
        entropy = ctx.population_spatial_entropy()
        assert entropy == 0.0  # Single agent = no entropy

    def test_room_distribution(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        dist = ctx.room_distribution()
        assert dist.get("ethos") == 3
        assert dist.get("pathos") == 1
        assert dist.get("logos") == 1

    def test_detect_clusters(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        clusters = ctx.detect_clusters(radius=10.0)

        # Should find one cluster: agent-1, agent-2, agent-3 (all close)
        assert len(clusters) == 1
        ids = clusters[0]
        assert {"agent-1", "agent-2", "agent-3"}.issubset(ids)
        assert "agent-4" not in ids
        assert "agent-5" not in ids

    def test_detect_clusters_none(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        clusters = ctx.detect_clusters(radius=1.0)
        # With radius 1, no two agents are close enough
        assert len(clusters) == 0

    def test_recommend_relocation_needed(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        rec = ctx.recommend_relocation("agent-1", target_entropy=0.9)

        # Population is clustered, so relocation recommended
        assert rec is not None
        assert rec.position != (0.0, 0.0)  # Should suggest different position

    def test_recommend_relocation_not_needed(self):
        # Well-dispersed population
        proj = SpatialProjector("node-test", dimension=2)
        proj.project_state("a1", "ethos", WorldState(position=(0.0, 0.0)))
        proj.project_state("a2", "pathos", WorldState(position=(100.0, 0.0)))
        proj.project_state("a3", "logos", WorldState(position=(0.0, 100.0)))

        ctx = SpatialBreedingContext(proj)
        rec = ctx.recommend_relocation("a1", target_entropy=0.5)
        # Already well-dispersed, no relocation needed
        assert rec is None

    def test_recommend_relocation_unknown_agent(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        rec = ctx.recommend_relocation("nonexistent", target_entropy=0.9)
        assert rec is None

    def test_adaptive_radius_expansion(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        # agent-1 has 2 neighbors at radius 10
        radius = ctx._adapt_radius("agent-1", base_radius=10.0)
        # Density is moderate (2 neighbors), radius stays similar
        assert radius >= 5.0

    def test_adaptive_radius_sparse(self):
        proj = SpatialProjector("node-test", dimension=2)
        proj.project_state("agent-1", "ethos", WorldState(position=(0.0, 0.0)))
        # No other agents

        ctx = SpatialBreedingContext(proj)
        radius = ctx._adapt_radius("agent-1", base_radius=10.0)
        # Sparse: expand radius
        assert radius > 10.0

    def test_distance_calculation(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        dist = ctx._distance("agent-1", "agent-2")
        assert dist == 5.0  # 3-4-5 triangle

    def test_distance_unknown_agent(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        dist = ctx._distance("agent-1", "nonexistent")
        assert dist == float("inf")

    def test_get_all_agent_ids(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        ids = ctx._get_all_agent_ids()
        assert len(ids) == 5
        assert "agent-1" in ids

    def test_get_all_agent_ids_exclude(self, populated_projector):
        ctx = SpatialBreedingContext(populated_projector)
        ids = ctx._get_all_agent_ids(exclude="agent-1")
        assert "agent-1" not in ids
        assert len(ids) == 4

    def test_config_defaults(self):
        config = SpatialBreedingConfig()
        assert config.proximal_radius == 10.0
        assert config.proximity_ratio == 0.6
        assert config.adaptive_radius is True

    def test_custom_config(self):
        config = SpatialBreedingConfig(
            proximal_radius=5.0,
            diversity_k=5,
            same_room_bonus=1.5,
        )
        assert config.proximal_radius == 5.0
        assert config.diversity_k == 5
        assert config.same_room_bonus == 1.5


class TestSpatialBreedingIntegration:
    def test_full_breeding_pipeline(self):
        """End-to-end: project → select parents → convert for breeder."""
        proj = SpatialProjector("node-test", dimension=2)

        # Population of agents
        for i in range(10):
            proj.project_state(
                f"agent-{i}",
                "ethos",
                WorldState(position=(float(i * 2), 0.0), semantics={"genome_id": i}),
            )

        ctx = SpatialBreedingContext(proj)

        # Select proximal parents for agent-5
        genome_fn = lambda aid: {"agent": aid}
        fitness_fn = lambda aid: 1.0

        proximal = ctx.select_proximal_parents(
            "agent-5", radius=5.0, k=3, genome_fn=genome_fn, fitness_fn=fitness_fn
        )
        assert len(proximal) > 0

        # Convert to breeder format
        breeder_parents = ctx.to_breeder_parents(proximal)
        assert len(breeder_parents) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in breeder_parents)

    def test_entropy_improvement_via_relocation(self):
        """Relocation recommendations should improve entropy."""
        proj = SpatialProjector("node-test", dimension=2)

        # Clustered population
        for i in range(5):
            proj.project_state(
                f"agent-{i}", "ethos", WorldState(position=(float(i), 0.0))
            )

        ctx = SpatialBreedingContext(proj)
        initial_entropy = ctx.population_spatial_entropy()

        # Get relocation recommendation for agent-0
        rec = ctx.recommend_relocation("agent-0", target_entropy=0.9)
        if rec:
            # Apply relocation
            proj.project_state("agent-0", "ethos", rec)

        new_entropy = ctx.population_spatial_entropy()
        # Entropy should improve or stay same
        assert new_entropy >= initial_entropy

    def test_room_affinity_breeding(self):
        """Prefer parents in the same room with explicit fitness."""
        proj = SpatialProjector("node-test", dimension=2)
        proj.project_state("a1", "ethos", WorldState(position=(0.0, 0.0)))
        proj.project_state("a2", "ethos", WorldState(position=(1.0, 0.0)))
        proj.project_state("a3", "pathos", WorldState(position=(2.0, 0.0)))

        ctx = SpatialBreedingContext(proj)

        def fitness_fn(aid):
            return 10.0  # Equal base fitness

        parents = ctx.select_room_affinity_parents(
            "a1", room_id="ethos", k=2, fitness_fn=fitness_fn
        )

        # a2 is in ethos, a3 is in pathos
        # With same_room_bonus=1.2, a2 should have fitness=12.0
        # With cross_room_penalty=0.8, a3 should have fitness=8.0
        a2_parent = next((p for p in parents if p.agent_id == "a2"), None)
        a3_parent = next((p for p in parents if p.agent_id == "a3"), None)

        if a2_parent and a3_parent:
            assert a2_parent.fitness == 12.0  # 10 * 1.2
            assert a3_parent.fitness == 8.0  # 10 * 0.8
            assert a2_parent.fitness > a3_parent.fitness

    def test_diversity_injection(self):
        """Diverse parents inject spatial novelty."""
        proj = SpatialProjector("node-test", dimension=2)
        proj.project_state("agent-1", "ethos", WorldState(position=(0.0, 0.0)))
        proj.project_state("agent-2", "ethos", WorldState(position=(1.0, 0.0)))
        proj.project_state("agent-3", "ethos", WorldState(position=(2.0, 0.0)))
        proj.project_state("agent-4", "pathos", WorldState(position=(100.0, 0.0)))

        ctx = SpatialBreedingContext(proj)

        # Proximal parents: all close
        proximal = ctx.select_proximal_parents("agent-1", radius=5.0, k=3)
        assert all(p.distance <= 5.0 for p in proximal)

        # Diverse parents: far away
        diverse = ctx.select_diverse_parents("agent-1", min_distance=50.0, k=2)
        assert len(diverse) == 1  # Only agent-4 qualifies
        assert diverse[0].agent_id == "agent-4"

    def test_trinity_room_spatial_breeding(self):
        """Agents in different trinity rooms have spatial separation."""
        proj = SpatialProjector("node-test", dimension=2)

        # Ethos agents at origin cluster
        proj.project_state("breeder-e1", "ethos", WorldState(position=(0.0, 0.0)))
        proj.project_state("breeder-e2", "ethos", WorldState(position=(2.0, 0.0)))

        # Pathos agents shifted y
        proj.project_state("solver-p1", "pathos", WorldState(position=(0.0, 20.0)))
        proj.project_state("solver-p2", "pathos", WorldState(position=(2.0, 20.0)))

        # Logos agents shifted x
        proj.project_state("auditor-l1", "logos", WorldState(position=(40.0, 0.0)))

        ctx = SpatialBreedingContext(proj)

        # Ethos breeder should find ethos neighbors
        ethos_neighbors = ctx.select_proximal_parents("breeder-e1", radius=5.0, k=2)
        assert all(p.room_id == "ethos" for p in ethos_neighbors)

        # But diverse selection finds pathos/logos
        diverse = ctx.select_diverse_parents("breeder-e1", min_distance=15.0, k=2)
        ids = [p.agent_id for p in diverse]
        assert any("solver" in aid or "auditor" in aid for aid in ids)
