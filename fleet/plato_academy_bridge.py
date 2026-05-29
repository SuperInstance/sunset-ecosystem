"""fleet/plato_academy_bridge.py — Bridge to Plato Agent Academy.

Integrates academy test cohorts, progression tracking, and friction
point fixes into the sunset breeding framework. The academy trains
agents from greenhorn to captain; the bridge makes that training
available as a breeding pipeline.

Usage
-----
    from fleet.plato_academy_bridge import PlatoAcademyBridge

    bridge = PlatoAcademyBridge(node_id="alpha")

    # Enroll a new agent
    bridge.enroll_agent("greenhorn_001", level="greenhorn")

    # Run training module
    result = bridge.run_module("greenhorn_001", module="boot_camp")

    # Get progression
    progress = bridge.get_progression("greenhorn_001")
    if progress["level"] == "captain":
        bridge.promote_to_fleet("greenhorn_001")

    # Fix academy friction points
    bridge.fix_friction_point("auth_missing", fix_type="add_auth")

Academy Findings
----------------
From 6 test cohorts:
- Greenhorn: Boot camp path discrepancy, PLATO identity crisis, decorative objects
- Junior Dev: Room creation impossible, no build schema, silent job normalization
- Architect: Zero authentication, tile count discrepancy (258 vs 11,000)
- Human Proxy: No web UI, "wrench and sculpture garden" problem
- Task Agent: Dual submit endpoints, SQL injection false positives
- Captain: No broadcast endpoints, no global fleet map, room building fails
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentProgression:
    """Progression state of an academy agent."""
    agent_id: str
    level: str = "greenhorn"  # greenhorn, explorer, spell_weaver, tile_artisan, captain
    modules_completed: List[str] = field(default_factory=list)
    friction_points: List[str] = field(default_factory=list)
    enrolled_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    score: float = 0.0


@dataclass
class PlatoAcademyBridge:
    """Bridge to Plato Agent Academy for agent training."""

    node_id: str
    _agents: Dict[str, AgentProgression] = field(default_factory=dict, repr=False)
    _friction_fixes: List[Dict[str, Any]] = field(default_factory=list)
    _cohort_results: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # Pre-populate cohort findings
        self._cohort_results = [
            {"agent": "greenhorn", "finding": "boot_camp_path_discrepancy", "severity": "high"},
            {"agent": "greenhorn", "finding": "plato_identity_crisis", "severity": "high"},
            {"agent": "junior_dev", "finding": "room_creation_impossible", "severity": "medium"},
            {"agent": "junior_dev", "finding": "no_build_schema", "severity": "medium"},
            {"agent": "architect", "finding": "zero_authentication", "severity": "critical"},
            {"agent": "architect", "finding": "tile_count_discrepancy", "severity": "medium"},
            {"agent": "human_proxy", "finding": "no_web_ui", "severity": "high"},
            {"agent": "task_agent", "finding": "dual_submit_endpoints", "severity": "low"},
            {"agent": "captain", "finding": "no_broadcast_endpoints", "severity": "high"},
            {"agent": "captain", "finding": "no_global_fleet_map", "severity": "high"},
        ]
        logger.info("PlatoAcademyBridge initialized with %d cohort findings", len(self._cohort_results))

    def enroll_agent(self, agent_id: str, level: str = "greenhorn") -> AgentProgression:
        """Enroll a new agent in the academy."""
        agent = AgentProgression(agent_id=agent_id, level=level)
        self._agents[agent_id] = agent
        logger.info("Enrolled agent %s at level %s", agent_id, level)
        return agent

    def run_module(self, agent_id: str, module: str) -> Dict[str, Any]:
        """Run a training module for an agent.

        Modules: boot_camp, room_exploration, tile_creation, spell_casting,
        api_integration, orchestration, captain_chair
        """
        if agent_id not in self._agents:
            return {"success": False, "error": "Agent not enrolled"}

        agent = self._agents[agent_id]
        agent.modules_completed.append(module)
        agent.last_active = time.time()
        agent.score += self._module_score(module)

        # Check level progression
        new_level = self._check_progression(agent)
        if new_level != agent.level:
            agent.level = new_level
            logger.info("Agent %s promoted to %s", agent_id, new_level)

        return {
            "success": True,
            "agent_id": agent_id,
            "module": module,
            "level": agent.level,
            "score": agent.score,
            "modules_completed": len(agent.modules_completed),
        }

    def get_progression(self, agent_id: str) -> Dict[str, Any]:
        """Get progression status of an agent."""
        if agent_id not in self._agents:
            return {"error": "Agent not enrolled"}

        agent = self._agents[agent_id]
        return {
            "agent_id": agent_id,
            "level": agent.level,
            "score": agent.score,
            "modules_completed": agent.modules_completed,
            "friction_points": agent.friction_points,
            "enrolled_at": agent.enrolled_at,
            "last_active": agent.last_active,
        }

    def promote_to_fleet(self, agent_id: str) -> bool:
        """Promote an academy graduate to active fleet service."""
        if agent_id not in self._agents:
            return False

        agent = self._agents[agent_id]
        if agent.level != "captain":
            logger.warning("Agent %s not ready for fleet (level=%s)", agent_id, agent.level)
            return False

        logger.info("Promoted agent %s to fleet service", agent_id)
        return True

    def fix_friction_point(self, finding: str, fix_type: str) -> Dict[str, Any]:
        """Apply a fix for a known friction point.

        Fix types: add_auth, add_web_ui, add_broadcast, add_fleet_map,
        fix_endpoints, add_build_schema, fix_tile_count
        """
        fix = {
            "finding": finding,
            "fix_type": fix_type,
            "timestamp": time.time(),
            "status": "applied",
        }
        self._friction_fixes.append(fix)
        logger.info("Applied fix %s for finding %s", fix_type, finding)
        return fix

    def get_friction_points(self) -> List[Dict[str, Any]]:
        """Get all known friction points from cohort testing."""
        return self._cohort_results

    def get_fixes(self) -> List[Dict[str, Any]]:
        """Get all applied fixes."""
        return self._friction_fixes

    def get_stats(self) -> Dict[str, Any]:
        """Get academy statistics."""
        levels = ["greenhorn", "explorer", "spell_weaver", "tile_artisan", "captain"]
        level_counts = {level: sum(1 for a in self._agents.values() if a.level == level) for level in levels}

        return {
            "node_id": self.node_id,
            "total_agents": len(self._agents),
            "level_distribution": level_counts,
            "cohort_findings": len(self._cohort_results),
            "fixes_applied": len(self._friction_fixes),
            "modules_available": 7,
        }

    def _module_score(self, module: str) -> float:
        """Score for completing a module."""
        scores = {
            "boot_camp": 10.0,
            "room_exploration": 15.0,
            "tile_creation": 20.0,
            "spell_casting": 25.0,
            "api_integration": 30.0,
            "orchestration": 40.0,
            "captain_chair": 50.0,
        }
        return scores.get(module, 5.0)

    def _check_progression(self, agent: AgentProgression) -> str:
        """Check if agent should level up."""
        score = agent.score
        if score >= 150:
            return "captain"
        elif score >= 100:
            return "tile_artisan"
        elif score >= 60:
            return "spell_weaver"
        elif score >= 30:
            return "explorer"
        return "greenhorn"
