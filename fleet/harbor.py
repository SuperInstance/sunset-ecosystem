"""Harbor — Fleet module registry and health tracker.

An emergent application that tracks the sunset-ecosystem fleet's 20+ modules,
monitors their health, manages dependencies, and generates integration reports.

Think of Harbor as the fleet's dry dock: every module checks in, every test
is logged, every integration path is mapped. When something breaks, Harbor
knows where to look.

Usage
-----
    from fleet.harbor import Harbor

    harbor = Harbor()

    # Register a module
    harbor.register_module(
        name="VectorSwarm",
        path="swarm/vector_swarm.py",
        test_path="tests/test_vector_swarm.py",
        status="healthy",
        integrations=["FleetMemory", "CognitiveCache"],
    )

    # Get fleet health report
    report = harbor.generate_fleet_report()
    print(report["healthy_count"], "/", report["total_modules"])

    # Find integration gaps
    gaps = harbor.find_integration_gaps()
    for gap in gaps:
        print(f"{gap['source']} -> {gap['target']} is untested")
"""

from __future__ import annotations

__all__ = [
    "Harbor",
    "ModuleEntry",
    "IntegrationPath",
    "HealthReport",
    "DependencyGraph",
]

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleet.ternary_types import TernaryMap, TernaryValue

logger = logging.getLogger(__name__)


@dataclass
class ModuleEntry:
    """A registered fleet module."""

    name: str
    path: str
    test_path: str | None = None
    status: str = "unknown"  # unknown, healthy, degraded, critical
    test_count: int = 0
    test_passed: int = 0
    integrations: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    description: str = ""
    commit_hash: str = ""
    last_updated: str = ""

    @property
    def test_coverage(self) -> float:
        if self.test_count == 0:
            return 0.0
        return self.test_passed / self.test_count

    @property
    def health_ternary(self) -> int:
        """Health as ternary value: +1=healthy, 0=degraded, -1=critical."""
        if self.status == "healthy":
            return TernaryValue.POS
        if self.status == "degraded":
            return TernaryValue.ZERO
        return TernaryValue.NEG


@dataclass
class IntegrationPath:
    """A path between two modules."""

    source: str
    target: str
    status: str = "unmapped"  # unmapped, mapped, tested, broken
    tests: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class HealthReport:
    """Fleet-wide health report."""

    total_modules: int = 0
    healthy_count: int = 0
    degraded_count: int = 0
    critical_count: int = 0
    total_tests: int = 0
    tests_passed: int = 0
    integration_coverage: float = 0.0
    ternary_score: int = 0  # Fleet-wide consensus
    recommendations: list[str] = field(default_factory=list)


@dataclass
class DependencyGraph:
    """Dependency graph between modules."""

    nodes: dict[str, ModuleEntry] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def add_node(self, module: ModuleEntry) -> None:
        self.nodes[module.name] = module

    def add_edge(self, source: str, target: str) -> None:
        self.edges.append((source, target))

    def get_dependencies(self, name: str) -> list[str]:
        return [t for s, t in self.edges if s == name]

    def get_dependents(self, name: str) -> list[str]:
        return [s for s, t in self.edges if t == name]

    def topological_sort(self) -> list[str]:
        """Return modules in dependency order."""
        visited: set[str] = set()
        result: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for dep in self.get_dependencies(name):
                if dep in self.nodes:
                    visit(dep)
            result.append(name)

        for name in self.nodes:
            visit(name)

        return result

    def find_cycles(self) -> list[list[str]]:
        """Find circular dependencies."""
        cycles: list[list[str]] = []
        visited: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> None:
            if node in path:
                cycle = path[path.index(node) :]
                cycles.append(cycle)
                return
            if node in visited:
                return
            visited.add(node)
            path.append(node)
            for dep in self.get_dependencies(node):
                if dep in self.nodes:
                    dfs(dep)
            path.pop()

        for name in self.nodes:
            dfs(name)

        return cycles


class Harbor:
    """Fleet module registry and health tracker.

    Parameters
    ----------
    workspace : str
        Path to the sunset-ecosystem workspace.
    """

    def __init__(self, workspace: str = ".") -> None:
        self.workspace = Path(workspace)
        self.modules: dict[str, ModuleEntry] = {}
        self.integrations: list[IntegrationPath] = []
        self.graph = DependencyGraph()

    # ── Registration ────────────────────────────────────────

    def register_module(self, entry: ModuleEntry) -> None:
        """Register a module in Harbor."""
        self.modules[entry.name] = entry
        self.graph.add_node(entry)
        for dep in entry.dependencies:
            self.graph.add_edge(entry.name, dep)
        logger.info("Registered module: %s (%s tests)", entry.name, entry.test_count)

    def register_integration(self, path: IntegrationPath) -> None:
        """Register an integration path."""
        self.integrations.append(path)
        logger.info("Registered integration: %s -> %s", path.source, path.target)

    # ── Health Monitoring ───────────────────────────────────

    def update_module_health(self, name: str, status: str) -> None:
        """Update a module's health status."""
        if name not in self.modules:
            raise KeyError(f"Module not found: {name}")
        self.modules[name].status = status
        logger.info("Module %s health: %s", name, status)

    def update_test_results(self, name: str, passed: int, total: int) -> None:
        """Update test results for a module."""
        if name not in self.modules:
            raise KeyError(f"Module not found: {name}")
        self.modules[name].test_passed = passed
        self.modules[name].test_count = total

    def get_module_health(self, name: str) -> dict[str, Any]:
        """Get detailed health info for a module."""
        if name not in self.modules:
            return {"error": f"Module not found: {name}"}
        mod = self.modules[name]
        return {
            "name": mod.name,
            "status": mod.status,
            "test_coverage": mod.test_coverage,
            "health_ternary": mod.health_ternary,
            "dependencies": mod.dependencies,
            "integrations": mod.integrations,
        }

    # ── Fleet Reports ───────────────────────────────────────

    def generate_fleet_report(self) -> dict[str, Any]:
        """Generate comprehensive fleet health report."""
        total = len(self.modules)
        healthy = sum(1 for m in self.modules.values() if m.status == "healthy")
        degraded = sum(1 for m in self.modules.values() if m.status == "degraded")
        critical = sum(1 for m in self.modules.values() if m.status == "critical")
        total_tests = sum(m.test_count for m in self.modules.values())
        tests_passed = sum(m.test_passed for m in self.modules.values())

        # Integration coverage
        mapped = sum(1 for p in self.integrations if p.status != "unmapped")
        integration_coverage = (
            mapped / len(self.integrations) if self.integrations else 1.0
        )

        # Ternary consensus: fleet-wide health vote
        health_votes = [m.health_ternary for m in self.modules.values()]
        ternary_score = TernaryValue.consensus(health_votes, threshold=0.6)

        # Recommendations
        recommendations: list[str] = []
        if critical > 0:
            recommendations.append(
                f"{critical} critical modules need immediate attention"
            )
        if degraded > total * 0.3:
            recommendations.append(
                f"{degraded}/{total} modules degraded — fleet stability at risk"
            )
        if integration_coverage < 0.5:
            recommendations.append(
                f"Integration coverage only {integration_coverage * 100:.0f}% — more tests needed"
            )
        if total_tests == 0:
            recommendations.append("No tests registered — fleet is flying blind")

        report = HealthReport(
            total_modules=total,
            healthy_count=healthy,
            degraded_count=degraded,
            critical_count=critical,
            total_tests=total_tests,
            tests_passed=tests_passed,
            integration_coverage=integration_coverage,
            ternary_score=ternary_score,
            recommendations=recommendations,
        )

        return {
            "total_modules": report.total_modules,
            "healthy": report.healthy_count,
            "degraded": report.degraded_count,
            "critical": report.critical_count,
            "total_tests": report.total_tests,
            "tests_passed": report.tests_passed,
            "test_coverage": report.tests_passed / report.total_tests
            if report.total_tests > 0
            else 0.0,
            "integration_coverage": report.integration_coverage,
            "ternary_score": TernaryValue.to_string(report.ternary_score),
            "ternary_emoji": TernaryValue.to_emoji(report.ternary_score),
            "recommendations": report.recommendations,
            "module_details": [
                {
                    "name": m.name,
                    "status": m.status,
                    "tests": f"{m.test_passed}/{m.test_count}",
                    "health": TernaryValue.to_emoji(m.health_ternary),
                }
                for m in self.modules.values()
            ],
        }

    # ── Integration Analysis ────────────────────────────────

    def find_integration_gaps(self) -> list[dict[str, Any]]:
        """Find untested or unmapped integration paths."""
        gaps = []
        for path in self.integrations:
            if path.status == "unmapped":
                gaps.append(
                    {
                        "source": path.source,
                        "target": path.target,
                        "issue": "unmapped",
                        "description": path.description,
                    }
                )
            elif path.status == "broken":
                gaps.append(
                    {
                        "source": path.source,
                        "target": path.target,
                        "issue": "broken",
                        "description": path.description,
                    }
                )
        return gaps

    def get_integration_matrix(self) -> dict[str, dict[str, str]]:
        """Get integration status matrix."""
        matrix: dict[str, dict[str, str]] = {}
        for path in self.integrations:
            if path.source not in matrix:
                matrix[path.source] = {}
            matrix[path.source][path.target] = path.status
        return matrix

    def find_orphan_modules(self) -> list[str]:
        """Find modules with no integrations."""
        integrated = set()
        for path in self.integrations:
            integrated.add(path.source)
            integrated.add(path.target)
        return [name for name in self.modules if name not in integrated]

    def find_hub_modules(self, min_connections: int = 3) -> list[str]:
        """Find modules with many connections (hubs)."""
        connection_counts: dict[str, int] = {}
        for path in self.integrations:
            connection_counts[path.source] = connection_counts.get(path.source, 0) + 1
            connection_counts[path.target] = connection_counts.get(path.target, 0) + 1
        return [
            name
            for name, count in connection_counts.items()
            if count >= min_connections
        ]

    # ── Dependency Analysis ─────────────────────────────────

    def get_dependency_order(self) -> list[str]:
        """Get modules in dependency order."""
        return self.graph.topological_sort()

    def find_circular_dependencies(self) -> list[list[str]]:
        """Find circular dependencies."""
        return self.graph.find_cycles()

    def get_critical_path(self) -> list[str]:
        """Get the critical dependency path (deepest chain)."""
        order = self.graph.topological_sort()
        if not order:
            return []
        # Simple heuristic: longest path in DAG
        longest: list[str] = []
        for start in order:
            path = [start]
            current = start
            while True:
                deps = self.graph.get_dependencies(current)
                if not deps:
                    break
                next_dep = deps[0]
                if next_dep in path:
                    break
                path.append(next_dep)
                current = next_dep
            if len(path) > len(longest):
                longest = path
        return longest

    # ── Statistics ──────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get fleet statistics."""
        if not self.modules:
            return {"modules": 0, "integrations": 0, "tests": 0}

        statuses = [m.status for m in self.modules.values()]
        test_counts = [m.test_count for m in self.modules.values()]

        return {
            "modules": len(self.modules),
            "integrations": len(self.integrations),
            "tests": sum(test_counts),
            "mean_tests_per_module": sum(test_counts) / len(test_counts)
            if test_counts
            else 0,
            "status_distribution": {
                "healthy": statuses.count("healthy"),
                "degraded": statuses.count("degraded"),
                "critical": statuses.count("critical"),
                "unknown": statuses.count("unknown"),
            },
            "orphan_modules": len(self.find_orphan_modules()),
            "hub_modules": len(self.find_hub_modules(min_connections=3)),
            "circular_dependencies": len(self.find_circular_dependencies()),
        }

    # ── Bootstrapping ───────────────────────────────────────

    def bootstrap_fleet(self) -> None:
        """Bootstrap Harbor with the known fleet modules."""
        fleet_modules = [
            ModuleEntry(
                "HNSWMeshTable",
                "swarm/hnsw_mesh_table.py",
                "tests/test_hnsw_mesh_table.py",
                "healthy",
                11,
                11,
            ),
            ModuleEntry(
                "TieredMeshStorage",
                "swarm/tiered_mesh_storage.py",
                "tests/test_tiered_mesh_storage.py",
                "healthy",
                7,
                7,
            ),
            ModuleEntry(
                "FleetMemory",
                "fleet/fleet_memory.py",
                "tests/test_fleet_memory.py",
                "healthy",
                12,
                12,
            ),
            ModuleEntry(
                "MeshWAL",
                "swarm/mesh_wal.py",
                "tests/test_mesh_wal.py",
                "healthy",
                13,
                13,
            ),
            ModuleEntry(
                "MeshGrouping",
                "swarm/mesh_grouping.py",
                "tests/test_mesh_grouping.py",
                "healthy",
                10,
                10,
            ),
            ModuleEntry(
                "SceneTracker",
                "swarm/scene_tracker.py",
                "tests/test_scene_tracker.py",
                "healthy",
                10,
                10,
            ),
            ModuleEntry(
                "VectorSwarm",
                "swarm/vector_swarm.py",
                "tests/test_vector_swarm.py",
                "healthy",
                12,
                12,
            ),
            ModuleEntry(
                "CognitiveCache",
                "fleet/cognitive_cache.py",
                "tests/test_cognitive_cache.py",
                "healthy",
                15,
                15,
            ),
            ModuleEntry(
                "FleetAPI",
                "fleet/fleet_api.py",
                "tests/test_fleet_api.py",
                "healthy",
                8,
                8,
            ),
            ModuleEntry(
                "FleetMonitor",
                "fleet/fleet_monitor.py",
                "tests/test_fleet_monitor.py",
                "healthy",
                10,
                10,
            ),
            ModuleEntry(
                "QuantaVDBBridge",
                "fleet/quanta_vdb_bridge.py",
                "tests/test_quanta_vdb_bridge.py",
                "healthy",
                16,
                16,
            ),
            ModuleEntry(
                "CASLangExecutor",
                "fleet/caslang_executor.py",
                "tests/test_caslang_executor.py",
                "healthy",
                18,
                18,
            ),
            ModuleEntry(
                "LevelRunner",
                "fleet/level_runner.py",
                "tests/test_level_runner.py",
                "healthy",
                18,
                18,
            ),
            ModuleEntry(
                "Pincher",
                "fleet/pincher.py",
                "tests/test_pincher.py",
                "healthy",
                14,
                14,
            ),
            ModuleEntry(
                "xLangAgentBridge",
                "fleet/xlang_agent_bridge.py",
                "tests/test_xlang_agent_bridge.py",
                "healthy",
                16,
                16,
            ),
            ModuleEntry(
                "EcosystemHub",
                "fleet/ecosystem_hub.py",
                "tests/test_ecosystem_hub.py",
                "healthy",
                14,
                14,
            ),
            ModuleEntry(
                "PatternMine",
                "fleet/pattern_mine.py",
                "tests/test_pattern_mine.py",
                "healthy",
                23,
                23,
            ),
            ModuleEntry(
                "TMinusBridge",
                "fleet/t_minus_bridge.py",
                "tests/test_t_minus_bridge.py",
                "healthy",
                30,
                30,
            ),
            ModuleEntry(
                "BreedOptimizer",
                "fleet/breed_optimizer.py",
                "tests/test_breed_optimizer.py",
                "healthy",
                39,
                39,
            ),
            ModuleEntry(
                "TernaryTypes",
                "fleet/ternary_types.py",
                "tests/test_ternary_types.py",
                "healthy",
                60,
                60,
            ),
        ]

        for mod in fleet_modules:
            self.register_module(mod)

        # Register known integration paths
        integration_paths = [
            IntegrationPath(
                "VectorSwarm", "FleetMemory", "tested", ["test_vector_swarm"]
            ),
            IntegrationPath(
                "CognitiveCache", "FleetMemory", "tested", ["test_cognitive_cache"]
            ),
            IntegrationPath(
                "BreedOptimizer", "VectorSwarm", "mapped", ["test_breed_optimizer"]
            ),
            IntegrationPath(
                "BreedOptimizer", "CognitiveCache", "mapped", ["test_breed_optimizer"]
            ),
            IntegrationPath(
                "TMinusBridge", "FleetMonitor", "mapped", ["test_t_minus_bridge"]
            ),
            IntegrationPath(
                "TernaryTypes", "FleetMonitor", "mapped", ["test_ternary_types"]
            ),
            IntegrationPath(
                "PatternMine", "EcosystemHub", "mapped", ["test_pattern_mine"]
            ),
            IntegrationPath(
                "QuantaVDBBridge", "FleetMemory", "tested", ["test_quanta_vdb_bridge"]
            ),
            IntegrationPath(
                "CASLangExecutor", "LevelRunner", "tested", ["test_level_runner"]
            ),
            IntegrationPath("Pincher", "QuantaVDBBridge", "tested", ["test_pincher"]),
            IntegrationPath(
                "xLangAgentBridge", "LevelRunner", "tested", ["test_level_runner"]
            ),
            IntegrationPath(
                "EcosystemHub", "PatternMine", "tested", ["test_ecosystem_hub"]
            ),
        ]

        for path in integration_paths:
            self.register_integration(path)

        logger.info(
            "Bootstrapped fleet with %d modules and %d integrations",
            len(fleet_modules),
            len(integration_paths),
        )
