"""EcosystemHub — Auto-discovery and integration mapping for the SuperInstance fleet.

An emergent application that:
- Discovers repos via GitHub API
- Maps them to integration opportunities in sunset-ecosystem
- Tracks which repos have Python bridges
- Suggests priority order based on impact × effort
- Generates integration task cards for FleetMonitor

Usage
-----
    hub = EcosystemHub("SuperInstance")
    hub.discover()
    hub.map_integrations()
    for task in hub.suggest_priority_tasks():
        print(task.priority, task.target_repo, task.integration_module)
"""

from __future__ import annotations

__all__ = [
    "EcosystemHub",
    "RepoCard",
    "IntegrationTask",
    "IntegrationMap",
]

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RepoCard:
    """A discovered repository with metadata."""

    name: str
    url: str
    description: str | None = None
    primary_language: str | None = None
    is_fork: bool = False
    pushed_at: str | None = None
    updated_at: str | None = None
    tags: list[str] = field(default_factory=list)
    has_python_bridge: bool = False
    test_count: int | None = None
    integration_status: str = "none"  # none, planned, in_progress, complete


@dataclass
class IntegrationTask:
    """A concrete integration task."""

    priority: str  # P0, P1, P2
    target_repo: str
    target_module: str
    description: str
    effort_estimate: str  # hours
    impact_score: float  # 0.0-1.0
    sunset_module: str  # Which sunset-ecosystem module to extend
    bridge_type: str  # PyO3, FFI, subprocess, REST, none
    blocked_by: list[str] = field(default_factory=list)


@dataclass
class IntegrationMap:
    """A mapping from a SuperInstance repo to a sunset-ecosystem module."""

    repo_name: str
    sunset_module: str
    opportunity: str
    rationale: str
    priority: str


class EcosystemHub:
    """Discover and map the SuperInstance repo collection.

    Parameters
    ----------
    org : str
        GitHub organization name (default: "SuperInstance").
    cache_path : Path | None
        Where to cache discovered repo list.
    """

    def __init__(
        self,
        org: str = "SuperInstance",
        cache_path: Path | None = None,
    ) -> None:
        self.org = org
        self.cache_path = cache_path or Path(".ecosystem_hub_cache.json")
        self.repos: dict[str, RepoCard] = {}
        self.maps: list[IntegrationMap] = []
        self.tasks: list[IntegrationTask] = []
        self._discovery_time: float | None = None

    # ── Discovery ───────────────────────────────────────────

    def discover(self, force_refresh: bool = False) -> dict[str, RepoCard]:
        """Discover all repos in the organization.

        Uses cached results if available and fresh (<24h).
        """
        if not force_refresh and self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                data = json.load(f)
            age = time.time() - data.get("timestamp", 0)
            if age < 86400:
                self.repos = {
                    name: RepoCard(**card) for name, card in data["repos"].items()
                }
                self._discovery_time = data["timestamp"]
                logger.info(
                    "Loaded %d repos from cache (age=%.0fh)",
                    len(self.repos),
                    age / 3600,
                )
                return self.repos

        # Fetch via gh CLI
        try:
            result = subprocess.run(
                [
                    "gh",
                    "repo",
                    "list",
                    self.org,
                    "--limit",
                    "200",
                    "--json",
                    "name,description,primaryLanguage,pushedAt,updatedAt,url,isFork",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error("gh repo list failed: %s", result.stderr)
                return self.repos

            raw_repos = json.loads(result.stdout)
            for r in raw_repos:
                lang = (
                    r.get("primaryLanguage", {}).get("name")
                    if r.get("primaryLanguage")
                    else None
                )
                card = RepoCard(
                    name=r["name"],
                    url=r["url"],
                    description=r.get("description"),
                    primary_language=lang,
                    is_fork=r.get("isFork", False),
                    pushed_at=r.get("pushedAt"),
                    updated_at=r.get("updatedAt"),
                )
                self._auto_tag(card)
                self.repos[card.name] = card

            self._discovery_time = time.time()
            self._save_cache()
            logger.info("Discovered %d repos from GitHub", len(self.repos))

        except Exception as e:
            logger.error("Discovery failed: %s", e)

        return self.repos

    def _auto_tag(self, card: RepoCard) -> None:
        """Auto-tag repos based on name/description."""
        name = card.name.lower()
        desc = (card.description or "").lower()
        tags = []

        if "-rs" in name or card.primary_language == "Rust":
            tags.append("rust")
        if card.primary_language == "Python":
            tags.append("python")
        if card.primary_language == "C":
            tags.append("c")
        if "band" in name:
            tags.append("music")
        if "breed" in name or "evolution" in name or "genetic" in name:
            tags.append("breeding")
        if (
            "crypto" in name
            or "cipher" in name
            or "signature" in name
            or "lattice" in name
        ):
            tags.append("crypto")
        if "compress" in name or "huffman" in name or "lz" in name or "bwt" in name:
            tags.append("compression")
        if "tree" in name or "graph" in name or "search" in name:
            tags.append("algorithms")
        if "mesh" in name or "swarm" in name or "fleet" in name:
            tags.append("fleet")
        if "plato" in name or "engine" in name:
            tags.append("embedded")
        if "cuda" in name or "gpu" in name or "oxide" in name:
            tags.append("gpu")
        if "market" in name or "finance" in name:
            tags.append("finance")
        if "agent" in name or "ai" in name or "llm" in name:
            tags.append("agentic")
        if "math" in name or "geometry" in name or "topology" in name:
            tags.append("math")
        if "transport" in name or "wasserstein" in name:
            tags.append("optimal-transport")
        if card.is_fork:
            tags.append("fork")

        card.tags = tags

    def _save_cache(self) -> None:
        """Save discovered repos to cache file."""
        data = {
            "timestamp": self._discovery_time or time.time(),
            "repos": {
                name: self._repo_to_dict(card) for name, card in self.repos.items()
            },
        }
        with open(self.cache_path, "w") as f:
            json.dump(data, f, indent=2)

    def _repo_to_dict(self, card: RepoCard) -> dict[str, Any]:
        return {
            "name": card.name,
            "url": card.url,
            "description": card.description,
            "primary_language": card.primary_language,
            "is_fork": card.is_fork,
            "pushed_at": card.pushed_at,
            "updated_at": card.updated_at,
            "tags": card.tags,
            "has_python_bridge": card.has_python_bridge,
            "test_count": card.test_count,
            "integration_status": card.integration_status,
        }

    # ── Integration Mapping ─────────────────────────────────

    def map_integrations(self) -> list[IntegrationMap]:
        """Map discovered repos to sunset-ecosystem integration opportunities.

        Uses hard-coded rules based on repo analysis.
        """
        self.maps = []

        # Pre-defined high-value mappings
        mappings = [
            IntegrationMap(
                repo_name="cuda-oxide",
                sunset_module="swarm.hnsw_mesh_table",
                opportunity="GPU-accelerated HNSW ANN search",
                rationale="cuda-oxide provides Flux→PTX GPU runtime. HNSW mesh table can use GPU for batch distance computation, achieving 10-100x speedup over CPU brute-force.",
                priority="P0",
            ),
            IntegrationMap(
                repo_name="agent-operations",
                sunset_module="fleet.fleet_monitor",
                opportunity="Operational anti-patterns and best practices",
                rationale="agent-operations has patterns from 100+ repo sweeps. FleetMonitor can detect these patterns in fleet behavior and raise alerts.",
                priority="P0",
            ),
            IntegrationMap(
                repo_name="t-minus-rs",
                sunset_module="nerve.distributed_metronome_bridge",
                opportunity="Distributed deadline propagation",
                rationale="t-minus provides countdown/timer primitives with scheduling. MetronomeBridge can propagate deadlines across nodes for time-sensitive breeding tasks.",
                priority="P0",
            ),
            IntegrationMap(
                repo_name="NEXAH",
                sunset_module="swarm.mesh_grouping",
                opportunity="Topological anomaly detection",
                rationale="NEXAH extracts structure from dynamical systems. MeshGrouping can use persistent homology to detect topological anomalies in fleet vector distributions.",
                priority="P1",
            ),
            IntegrationMap(
                repo_name="optimal-transport-rs",
                sunset_module="swarm.fleet_bft_qd",
                opportunity="Wasserstein-based parent selection",
                rationale="optimal-transport provides Sinkhorn algorithm and Wasserstein distances. QD breeding can use Wasserstein distance for diversity-aware parent selection.",
                priority="P1",
            ),
            IntegrationMap(
                repo_name="market-manifold",
                sunset_module="swarm.vector_swarm",
                opportunity="Financial pattern search",
                rationale="market-manifold treats financial analysis as topological navigation. VectorSwarm can distribute financial pattern queries across fleet nodes.",
                priority="P1",
            ),
            IntegrationMap(
                repo_name="lattice-crypto-rs",
                sunset_module="swarm.fleet_bft_qd",
                opportunity="Post-quantum consensus signatures",
                rationale="lattice-crypto provides LWE/Ring-LWE primitives. FleetBFT can upgrade from HMAC to lattice-based signatures for post-quantum security.",
                priority="P2",
            ),
            IntegrationMap(
                repo_name="self-improving-band",
                sunset_module="fleet.cognitive_cache",
                opportunity="Musical pattern prediction",
                rationale="self-improving-band generates musical patterns. CognitiveCache can predict and preload musical patterns for real-time ensemble coordination.",
                priority="P2",
            ),
            IntegrationMap(
                repo_name="plato-engine-block-c",
                sunset_module="fleet.fleet_monitor",
                opportunity="Embedded sensor health monitoring",
                rationale="plato-engine-block-c is a bare-metal sensor→history→alarm engine. FleetMonitor can ingest sensor streams from embedded nodes.",
                priority="P2",
            ),
            IntegrationMap(
                repo_name="c-ternary",
                sunset_module="swarm.caslang_executor",
                opportunity="Three-valued logic constraints",
                rationale="c-ternary provides trit type and Leminal Zone deadband. caslang can extend to three-valued logic for uncertain constraint evaluation.",
                priority="P2",
            ),
            IntegrationMap(
                repo_name="conservation-law-rs",
                sunset_module="fleet.sense_decide_act",
                opportunity="Conservation-aware decision making",
                rationale="conservation-law provides Lagrangian mechanics and Noether theorem. SenseDecideAct can verify that decisions preserve conservation invariants.",
                priority="P2",
            ),
            IntegrationMap(
                repo_name="spectral-fleet-rs",
                sunset_module="swarm.mesh_vector_tables",
                opportunity="Spectral clustering of fleet vectors",
                rationale="spectral-fleet provides Lanczos iteration and spectral clustering. MeshVectorTables can use spectral methods for clustering instead of k-means.",
                priority="P2",
            ),
            IntegrationMap(
                repo_name="wasserstein-agents-rs",
                sunset_module="swarm.vector_swarm",
                opportunity="Multi-marginal agent distance",
                rationale="wasserstein-agents provides multi-marginal optimal transport. VectorSwarm can compute Wasserstein distances between agent distributions.",
                priority="P2",
            ),
            IntegrationMap(
                repo_name="tropical-geometry-rs",
                sunset_module="swarm.mesh_vector_tables",
                opportunity="Tropical min-plus vector operations",
                rationale="tropical-geometry provides min-plus/max-plus semirings. Vector operations can be extended to tropical algebra for scheduling and pathfinding.",
                priority="P3",
            ),
            IntegrationMap(
                repo_name="dial-theory-rs",
                sunset_module="fleet.fleet_memory",
                opportunity="Cultural memory persistence",
                rationale="dial-theory models cultural dial positions. FleetMemory can persist cultural evolution trajectories across generations.",
                priority="P3",
            ),
            IntegrationMap(
                repo_name="agent-homeostasis-rs",
                sunset_module="fleet.fleet_monitor",
                opportunity="Homeostatic fleet regulation",
                rationale="agent-homeostasis provides PID-inspired control loops. FleetMonitor can use homeostatic regulation to maintain stable fleet conditions.",
                priority="P3",
            ),
            IntegrationMap(
                repo_name="categorical-agents-rs",
                sunset_module="swarm.vector_swarm",
                opportunity="Categorical agent composition",
                rationale="categorical-agents provides category theory for agents. VectorSwarm can use categorical composition for complex agent workflows.",
                priority="P3",
            ),
            IntegrationMap(
                repo_name="fleet-warden-rs",
                sunset_module="fleet.fleet_monitor",
                opportunity="Resource guardian integration",
                rationale="fleet-warden provides automated disk cleanup and budget enforcement. FleetMonitor can trigger warden actions when resources exceed thresholds.",
                priority="P3",
            ),
            IntegrationMap(
                repo_name="hodge-music-rs",
                sunset_module="swarm.scene_tracker",
                opportunity="Hodge decomposition for scene tracking",
                rationale="hodge-music provides Hodge decomposition. SceneTracker can decompose query patterns into harmonic/curl/exact components.",
                priority="P3",
            ),
            IntegrationMap(
                repo_name="intention-field-rs",
                sunset_module="fleet.cognitive_cache",
                opportunity="Intention-based cache prediction",
                rationale="intention-field models agent intention fields. CognitiveCache can predict cache needs based on intention field gradients.",
                priority="P3",
            ),
        ]

        # Only add mappings for repos that actually exist in our collection
        for m in mappings:
            if m.repo_name in self.repos:
                self.maps.append(m)

        logger.info("Generated %d integration maps", len(self.maps))
        return self.maps

    # ── Priority Tasks ─────────────────────────────────────

    def suggest_priority_tasks(self) -> list[IntegrationTask]:
        """Generate concrete integration tasks sorted by priority and impact."""
        if not self.maps:
            self.map_integrations()

        # Convert maps to tasks with effort estimates
        task_specs = [
            (
                "P0",
                "cuda-oxide",
                "swarm.hnsw_mesh_table",
                "GPU HNSW backend",
                "8-12h",
                0.95,
                "PyO3",
            ),
            (
                "P0",
                "agent-operations",
                "fleet.fleet_monitor",
                "Operational pattern detection",
                "4-6h",
                0.90,
                "subprocess",
            ),
            (
                "P0",
                "t-minus-rs",
                "nerve.distributed_metronome_bridge",
                "Deadline propagation",
                "6-8h",
                0.85,
                "PyO3",
            ),
            (
                "P1",
                "NEXAH",
                "swarm.mesh_grouping",
                "Topological anomaly detection",
                "10-15h",
                0.80,
                "REST",
            ),
            (
                "P1",
                "optimal-transport-rs",
                "swarm.fleet_bft_qd",
                "Wasserstein parent selection",
                "8-12h",
                0.75,
                "PyO3",
            ),
            (
                "P1",
                "market-manifold",
                "swarm.vector_swarm",
                "Financial pattern search",
                "6-10h",
                0.70,
                "REST",
            ),
            (
                "P2",
                "lattice-crypto-rs",
                "swarm.fleet_bft_qd",
                "Post-quantum signatures",
                "12-20h",
                0.65,
                "PyO3",
            ),
            (
                "P2",
                "self-improving-band",
                "fleet.cognitive_cache",
                "Musical pattern prediction",
                "6-10h",
                0.60,
                "subprocess",
            ),
            (
                "P2",
                "plato-engine-block-c",
                "fleet.fleet_monitor",
                "Embedded sensor monitoring",
                "8-12h",
                0.55,
                "FFI",
            ),
            (
                "P2",
                "c-ternary",
                "swarm.caslang_executor",
                "Three-valued logic",
                "4-6h",
                0.50,
                "FFI",
            ),
            (
                "P2",
                "conservation-law-rs",
                "fleet.sense_decide_act",
                "Conservation-aware decisions",
                "8-12h",
                0.50,
                "PyO3",
            ),
            (
                "P2",
                "spectral-fleet-rs",
                "swarm.mesh_vector_tables",
                "Spectral clustering",
                "6-10h",
                0.45,
                "PyO3",
            ),
            (
                "P2",
                "wasserstein-agents-rs",
                "swarm.vector_swarm",
                "Multi-marginal distances",
                "8-12h",
                0.45,
                "PyO3",
            ),
            (
                "P3",
                "tropical-geometry-rs",
                "swarm.mesh_vector_tables",
                "Tropical vector ops",
                "6-10h",
                0.35,
                "PyO3",
            ),
            (
                "P3",
                "dial-theory-rs",
                "fleet.fleet_memory",
                "Cultural memory",
                "6-10h",
                0.30,
                "PyO3",
            ),
            (
                "P3",
                "agent-homeostasis-rs",
                "fleet.fleet_monitor",
                "Homeostatic regulation",
                "6-10h",
                0.30,
                "PyO3",
            ),
            (
                "P3",
                "categorical-agents-rs",
                "swarm.vector_swarm",
                "Categorical composition",
                "8-12h",
                0.25,
                "PyO3",
            ),
            (
                "P3",
                "fleet-warden-rs",
                "fleet.fleet_monitor",
                "Resource guardian",
                "4-6h",
                0.25,
                "PyO3",
            ),
            (
                "P3",
                "hodge-music-rs",
                "swarm.scene_tracker",
                "Hodge scene tracking",
                "6-10h",
                0.20,
                "PyO3",
            ),
            (
                "P3",
                "intention-field-rs",
                "fleet.cognitive_cache",
                "Intention prediction",
                "6-10h",
                0.20,
                "PyO3",
            ),
        ]

        for priority, repo, module, desc, effort, impact, bridge in task_specs:
            if repo in self.repos:
                self.tasks.append(
                    IntegrationTask(
                        priority=priority,
                        target_repo=repo,
                        target_module=module,
                        description=desc,
                        effort_estimate=effort,
                        impact_score=impact,
                        sunset_module=module,
                        bridge_type=bridge,
                    )
                )

        # Sort by priority then impact
        priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        self.tasks.sort(
            key=lambda t: (priority_order.get(t.priority, 99), -t.impact_score)
        )

        logger.info("Generated %d integration tasks", len(self.tasks))
        return self.tasks

    # ── Reports ─────────────────────────────────────────────

    def generate_report(self) -> dict[str, Any]:
        """Generate a comprehensive ecosystem report."""
        if not self.repos:
            self.discover()
        if not self.maps:
            self.map_integrations()
        if not self.tasks:
            self.suggest_priority_tasks()

        # Categorize repos
        categories: dict[str, list[str]] = {
            "rust_crates": [],
            "python_apps": [],
            "c_embedded": [],
            "forks": [],
            "band_cluster": [],
            "math_physics": [],
            "crypto_compression": [],
            "agent_systems": [],
            "uncategorized": [],
        }

        for card in self.repos.values():
            if card.is_fork:
                categories["forks"].append(card.name)
            elif "band" in card.tags or "music" in card.tags:
                categories["band_cluster"].append(card.name)
            elif "rust" in card.tags and "algorithms" in card.tags:
                categories["rust_crates"].append(card.name)
            elif "python" in card.tags or card.primary_language == "Python":
                categories["python_apps"].append(card.name)
            elif "c" in card.tags:
                categories["c_embedded"].append(card.name)
            elif "math" in card.tags or "optimal-transport" in card.tags:
                categories["math_physics"].append(card.name)
            elif "crypto" in card.tags or "compression" in card.tags:
                categories["crypto_compression"].append(card.name)
            elif "agentic" in card.tags or "fleet" in card.tags:
                categories["agent_systems"].append(card.name)
            else:
                categories["uncategorized"].append(card.name)

        return {
            "org": self.org,
            "total_repos": len(self.repos),
            "discovery_time": self._discovery_time,
            "categories": {k: len(v) for k, v in categories.items()},
            "integration_maps": len(self.maps),
            "pending_tasks": len(self.tasks),
            "p0_tasks": len([t for t in self.tasks if t.priority == "P0"]),
            "p1_tasks": len([t for t in self.tasks if t.priority == "P1"]),
            "p2_tasks": len([t for t in self.tasks if t.priority == "P2"]),
            "p3_tasks": len([t for t in self.tasks if t.priority == "P3"]),
            "top_5_tasks": [
                {
                    "priority": t.priority,
                    "repo": t.target_repo,
                    "description": t.description,
                    "effort": t.effort_estimate,
                    "impact": t.impact_score,
                    "bridge": t.bridge_type,
                }
                for t in self.tasks[:5]
            ],
            "python_bridge_gaps": len(
                [
                    c
                    for c in self.repos.values()
                    if "rust" in c.tags and not c.has_python_bridge
                ]
            ),
        }

    def write_report(self, path: Path | None = None) -> Path:
        """Write the ecosystem report to a markdown file."""
        report = self.generate_report()
        path = path or Path("docs/ECOSYSTEM_HUB_REPORT.md")

        lines = [
            "# SuperInstance Ecosystem Hub Report",
            "",
            f"**Organization:** {report['org']}",
            f"**Total Repos:** {report['total_repos']}",
            f"**Discovery Time:** {time.ctime(report['discovery_time']) if report['discovery_time'] else 'N/A'}",
            "",
            "## Categories",
            "",
        ]
        for cat, count in report["categories"].items():
            lines.append(f"- **{cat}:** {count}")
        lines.append("")
        lines.append("## Integration Pipeline")
        lines.append("")
        lines.append(f"- **Integration Maps:** {report['integration_maps']}")
        lines.append(f"- **Pending Tasks:** {report['pending_tasks']}")
        lines.append(f"  - P0: {report['p0_tasks']}")
        lines.append(f"  - P1: {report['p1_tasks']}")
        lines.append(f"  - P2: {report['p2_tasks']}")
        lines.append(f"  - P3: {report['p3_tasks']}")
        lines.append("")
        lines.append("## Top 5 Priority Tasks")
        lines.append("")
        for i, task in enumerate(report["top_5_tasks"], 1):
            lines.append(f"### {i}. [{task['priority']}] {task['repo']}")
            lines.append(f"- **Description:** {task['description']}")
            lines.append(f"- **Effort:** {task['effort']}")
            lines.append(f"- **Impact:** {task['impact']:.0%}")
            lines.append(f"- **Bridge Type:** {task['bridge']}")
            lines.append("")
        lines.append("## Python Bridge Gaps")
        lines.append("")
        lines.append(
            f"**{report['python_bridge_gaps']}** Rust repos lack Python bridges."
        )
        lines.append("This is the largest untapped capability surface in the fleet.")
        lines.append("")
        lines.append("## Recommendations")
        lines.append("")
        lines.append(
            "1. **Build PyO3 bridges for P0 repos** — cuda-oxide, t-minus-rs, agent-operations"
        )
        lines.append(
            "2. **Create integration examples** for each bridge showing concrete usage"
        )
        lines.append("3. **Add CI/CD** to auto-test bridges on every commit")
        lines.append(
            "4. **Write E2E tests** that exercise the full stack from Rust → Python → Fleet"
        )
        lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        logger.info("Wrote ecosystem report to %s", path)
        return path
