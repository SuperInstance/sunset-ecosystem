"""agent_migration.py — Agent state migration between fleet nodes.

Provides:
1. Serialize agent state (checkpoints)
2. Transfer state to target node
3. Resume agent on destination
4. Migration validation (state hash verification)
5. Rollback on failure

Usage:
    migrator = AgentMigrator()
    result = migrator.migrate(
        agent_id="agent-42",
        from_node="node-a",
        to_node="node-b",
        state={"memory": ..., "weights": ...},
    )
    assert result.success is True
"""

from __future__ import annotations

__all__ = [
    "AgentMigrator",
    "MigrationResult",
    "AgentState",
]

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Serialized agent state."""

    agent_id: str
    node_id: str
    checkpoint_data: dict[str, Any]
    timestamp: float
    version: str = "1.0"

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of checkpoint data."""
        payload = json.dumps(self.checkpoint_data, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass
class MigrationResult:
    """Result of a migration attempt."""

    success: bool
    agent_id: str
    from_node: str
    to_node: str
    state_hash: str = ""
    duration_ms: float = 0.0
    message: str = ""


class AgentMigrator:
    """Migrate agents between fleet nodes."""

    def __init__(self, max_history: int = 100) -> None:
        self._history: list[MigrationResult] = []
        self._max_history = max_history

    def migrate(
        self,
        agent_id: str,
        from_node: str,
        to_node: str,
        state: dict[str, Any],
    ) -> MigrationResult:
        """Migrate an agent from one node to another."""
        start = time.time()

        # Create state snapshot
        agent_state = AgentState(
            agent_id=agent_id,
            node_id=from_node,
            checkpoint_data=state,
            timestamp=start,
        )
        state_hash = agent_state.compute_hash()

        # Simulate transfer (in production: HTTP POST to target node)
        try:
            transferred = self._transfer_state(agent_state, to_node)
            if not transferred:
                result = MigrationResult(
                    success=False,
                    agent_id=agent_id,
                    from_node=from_node,
                    to_node=to_node,
                    state_hash=state_hash,
                    duration_ms=(time.time() - start) * 1000,
                    message="transfer failed",
                )
                self._log_result(result)
                return result

            # Verify state integrity on destination
            verified = self._verify_state(agent_state, to_node)
            if not verified:
                # Rollback
                self._rollback(agent_state, to_node)
                result = MigrationResult(
                    success=False,
                    agent_id=agent_id,
                    from_node=from_node,
                    to_node=to_node,
                    state_hash=state_hash,
                    duration_ms=(time.time() - start) * 1000,
                    message="state verification failed, rolled back",
                )
                self._log_result(result)
                return result

            result = MigrationResult(
                success=True,
                agent_id=agent_id,
                from_node=from_node,
                to_node=to_node,
                state_hash=state_hash,
                duration_ms=(time.time() - start) * 1000,
                message="migration successful",
            )
            self._log_result(result)
            return result

        except Exception as e:
            result = MigrationResult(
                success=False,
                agent_id=agent_id,
                from_node=from_node,
                to_node=to_node,
                state_hash=state_hash,
                duration_ms=(time.time() - start) * 1000,
                message=f"exception: {e}",
            )
            self._log_result(result)
            return result

    def _transfer_state(self, state: AgentState, to_node: str) -> bool:
        """Transfer state to target node. Placeholder for network call."""
        logger.info(f"Transferring agent {state.agent_id} to {to_node}")
        # In production: POST /api/v1/agents/{agent_id}/restore with payload
        return True

    def _verify_state(self, state: AgentState, node: str) -> bool:
        """Verify state integrity on destination. Placeholder."""
        logger.info(f"Verifying state for {state.agent_id} on {node}")
        return True

    def _rollback(self, state: AgentState, node: str) -> None:
        """Rollback migration on failure. Placeholder."""
        logger.warning(f"Rolling back migration of {state.agent_id} from {node}")

    def _log_result(self, result: MigrationResult) -> None:
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

    def history(self) -> list[MigrationResult]:
        """Get migration history."""
        return list(self._history)

    def success_rate(self) -> float:
        """Overall migration success rate."""
        if not self._history:
            return 0.0
        return sum(1 for r in self._history if r.success) / len(self._history)

    def avg_duration_ms(self) -> float:
        """Average migration duration."""
        if not self._history:
            return 0.0
        return sum(r.duration_ms for r in self._history) / len(self._history)

    def report(self) -> dict[str, Any]:
        """Migration statistics report."""
        total = len(self._history)
        successful = sum(1 for r in self._history if r.success)
        return {
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": self.success_rate(),
            "avg_duration_ms": self.avg_duration_ms(),
        }

    def __repr__(self) -> str:
        return f"AgentMigrator(history={len(self._history)})"
