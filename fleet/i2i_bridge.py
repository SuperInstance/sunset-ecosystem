"""fleet/i2i_bridge.py — I2I Protocol Bridge (5-Layer Agent Interaction).

Implements SuperInstance's I2I (Instance-to-Instance, Iteration-to-Iteration,
Individual-to-Individual, Interaction-to-Interaction, Iron-to-Iron) protocol
as a Python bridge for sunset-ecosystem fleet coordination.

Each layer has its own time scale, channel, and semantics:
  Instance     — ms scale, HTTP/API calls
  Iteration    — min-hours, PLATO tiles / ensigns
  Individual   — hours-days, git commits (bottles)
  Interaction  — days-weeks, Matrix / MUD rooms
  Iron         — permanent, hardware topology

References
----------
- SuperInstance/SuperInstance README — I2I five layers
- github.com/topics/cocapn — PLATO I2I repo
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Data structures ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentIdentity:
    """Origin-centric agent identity."""
    name: str
    role: str
    hardware: str
    capabilities: Tuple[str, ...] = ()

    def __repr__(self) -> str:
        return f"AgentIdentity({self.name}, {self.role})"


@dataclass
class I2IMessage:
    """A message at any I2I layer."""
    layer: str  # instance | iteration | individual | interaction | iron
    sender: AgentIdentity
    recipient: Optional[AgentIdentity]
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    channel: str = ""  # HTTP endpoint, PLATO room, git branch, Matrix room

    def to_json(self) -> str:
        return json.dumps({
            "layer": self.layer,
            "sender": {"name": self.sender.name, "role": self.sender.role},
            "recipient": {"name": self.recipient.name, "role": self.recipient.role} if self.recipient else None,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "channel": self.channel,
        }, default=str)


@dataclass
class Bottle:
    """Git-native message (Individual layer)."""
    from_agent: str
    to_agent: Optional[str] = None
    subject: str = ""
    body: str = ""
    repo_path: str = "."
    branch: str = "main"

    def write(self) -> str:
        """Write bottle as a git commit. Returns commit hash."""
        repo = Path(self.repo_path)
        if not (repo / ".git").exists():
            raise RuntimeError(f"Not a git repo: {self.repo_path}")

        bottle_dir = repo / "from-fleet" / self.from_agent
        bottle_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{self.subject.replace(' ', '-').lower()}.md"
        filepath = bottle_dir / filename

        content = f"""# {self.subject}

**From:** {self.from_agent}
**To:** {self.to_agent or 'fleet'}
**Time:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

{self.body}
"""
        filepath.write_text(content)

        # Stage and commit
        subprocess.run(
            ["git", "add", str(filepath)],
            cwd=self.repo_path, capture_output=True, check=True,
        )
        result = subprocess.run(
            ["git", "commit", "-m", f"bottle: {self.subject} ({self.from_agent})"],
            cwd=self.repo_path, capture_output=True, text=True,
        )
        if result.returncode != 0 and "nothing to commit" not in result.stderr:
            raise RuntimeError(f"git commit failed: {result.stderr}")

        # Get commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_path, capture_output=True, text=True, check=True,
        )
        return hash_result.stdout.strip()


# ── Layer implementations ────────────────────────────────────────────────

class InstanceLayer:
    """Milliseconds-scale compute↔compute via HTTP/API."""

    def __init__(self, timeout_ms: float = 5000):
        self.timeout_ms = timeout_ms
        self._log: List[I2IMessage] = []

    def call(self, sender: AgentIdentity, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Simulated HTTP call — subclasses override for real HTTP."""
        msg = I2IMessage(
            layer="instance",
            sender=sender,
            recipient=None,
            payload=payload,
            channel=endpoint,
        )
        self._log.append(msg)
        logger.debug(f"InstanceLayer.call {sender.name} → {endpoint}")
        return {"status": "simulated", "endpoint": endpoint, "sent_by": sender.name}

    def history(self) -> List[I2IMessage]:
        return list(self._log)


class IterationLayer:
    """Minutes-hours scale learning↔learning via PLATO tiles / ensigns."""

    def __init__(self, plato_bridge: Optional[Any] = None):
        self.plato = plato_bridge
        self._log: List[I2IMessage] = []

    def submit_tile(self, sender: AgentIdentity, domain: str, question: str, answer: str) -> Dict[str, Any]:
        payload = {"domain": domain, "question": question, "answer": answer}
        msg = I2IMessage(
            layer="iteration",
            sender=sender,
            recipient=None,
            payload=payload,
            channel=f"plato://{domain}",
        )
        self._log.append(msg)
        if self.plato is not None:
            try:
                return self.plato.submit(agent=sender.name, question=question, answer=answer, domain=domain)
            except Exception as e:
                logger.warning(f"PLATO submit failed: {e}")
        return {"status": "buffered", "domain": domain, "tile_by": sender.name}

    def history(self) -> List[I2IMessage]:
        return list(self._log)


class IndividualLayer:
    """Hours-days scale identity↔identity via git commits (bottles)."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self._log: List[I2IMessage] = []

    def send_bottle(self, bottle: Bottle) -> str:
        commit_hash = bottle.write()
        msg = I2IMessage(
            layer="individual",
            sender=AgentIdentity(bottle.from_agent, "sender", "generic"),
            recipient=AgentIdentity(bottle.to_agent, "recipient", "generic") if bottle.to_agent else None,
            payload={"commit": commit_hash, "subject": bottle.subject},
            channel=f"git://{bottle.branch}",
        )
        self._log.append(msg)
        return commit_hash

    def read_bottles(self, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read bottle files from from-fleet/ directory."""
        repo = Path(self.repo_path)
        bottles = []
        fleet_dir = repo / "from-fleet"
        if not fleet_dir.exists():
            return bottles

        search_dirs = [fleet_dir / agent_name] if agent_name else fleet_dir.iterdir()
        for d in search_dirs:
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.md")):
                bottles.append({
                    "from": d.name,
                    "file": str(f.relative_to(repo)),
                    "content": f.read_text(),
                })
        return bottles

    def history(self) -> List[I2IMessage]:
        return list(self._log)


class InteractionLayer:
    """Days-weeks scale exchange↔exchange via Matrix / MUD rooms."""

    def __init__(self, room_registry: Optional[Dict[str, str]] = None):
        self.rooms = room_registry or {}
        self._log: List[I2IMessage] = []

    def join_room(self, room_id: str, url: str) -> None:
        self.rooms[room_id] = url

    def send(self, sender: AgentIdentity, room_id: str, message: str) -> Dict[str, Any]:
        payload = {"message": message, "room": room_id}
        msg = I2IMessage(
            layer="interaction",
            sender=sender,
            recipient=None,
            payload=payload,
            channel=f"matrix://{room_id}",
        )
        self._log.append(msg)
        return {"status": "sent", "room": room_id, "by": sender.name}

    def history(self) -> List[I2IMessage]:
        return list(self._log)


class IronLayer:
    """Permanent scale hardware↔hardware topology awareness."""

    def __init__(self):
        self.topology: Dict[str, Dict[str, Any]] = {}
        self._log: List[I2IMessage] = []

    def register_hardware(self, agent: AgentIdentity, specs: Dict[str, Any]) -> None:
        self.topology[agent.name] = {
            "agent": agent,
            "specs": specs,
            "registered_at": time.time(),
        }
        msg = I2IMessage(
            layer="iron",
            sender=agent,
            recipient=None,
            payload={"action": "register", "specs": specs},
            channel="topology",
        )
        self._log.append(msg)

    def get_topology(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.topology)

    def find_by_capability(self, capability: str) -> List[AgentIdentity]:
        return [
            entry["agent"]
            for entry in self.topology.values()
            if capability in entry["agent"].capabilities
        ]

    def history(self) -> List[I2IMessage]:
        return list(self._log)


# ── Unified I2I Bridge ──────────────────────────────────────────────────

class I2IBridge:
    """Unified access to all five I2I layers."""

    def __init__(
        self,
        repo_path: str = ".",
        plato_bridge: Optional[Any] = None,
        room_registry: Optional[Dict[str, str]] = None,
    ):
        self.instance = InstanceLayer()
        self.iteration = IterationLayer(plato_bridge)
        self.individual = IndividualLayer(repo_path)
        self.interaction = InteractionLayer(room_registry)
        self.iron = IronLayer()

    @property
    def identity(self) -> AgentIdentity:
        """Override in subclasses to provide local agent identity."""
        return AgentIdentity("unknown", "agent", "generic")

    def full_history(self) -> List[I2IMessage]:
        return (
            self.instance.history()
            + self.iteration.history()
            + self.individual.history()
            + self.interaction.history()
            + self.iron.history()
        )

    def summary(self) -> Dict[str, int]:
        return {
            "instance": len(self.instance.history()),
            "iteration": len(self.iteration.history()),
            "individual": len(self.individual.history()),
            "interaction": len(self.interaction.history()),
            "iron": len(self.iron.history()),
        }

    def __repr__(self) -> str:
        return f"I2IBridge({self.summary()})"


__all__ = [
    "AgentIdentity",
    "I2IMessage",
    "Bottle",
    "InstanceLayer",
    "IterationLayer",
    "IndividualLayer",
    "InteractionLayer",
    "IronLayer",
    "I2IBridge",
]
