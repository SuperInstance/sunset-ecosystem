"""fleet/heartbeat_bridge.py — Heartbeat Protocol pattern integration.

Brings the oracle1-vessel heartbeat pattern into sunset-ecosystem:
- Fleet registry discovery (no hardcoded room names)
- Service health checks (PLATO, Matrix, etc.)
- Task discovery from PLATO tiles
- Acknowledgment tracking
- State persistence across sessions
- Daemon mode support

Usage:
    from fleet.heartbeat_bridge import Heartbeat

    hb = Heartbeat(plato_url="http://<BOAT_IP>:8847")
    report = hb.run()  # Returns text report of tasks found
    hb.save_state()    # Persist ack state
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime, timezone
from urllib.request import urlopen
from urllib.error import URLError


@dataclass
class ServiceCheck:
    """A service to health-check."""
    name: str
    url: str
    timeout: float = 5.0


@dataclass
class TaskTile:
    """A task discovered from a PLATO tile."""
    tile_id: str
    room: str
    question: str
    answer: str
    source: str


@dataclass
class HeartbeatState:
    """Persistent heartbeat state."""
    acknowledged: set[str] = field(default_factory=set)
    last_check: float = 0.0
    task_count: int = 0

    def to_dict(self) -> dict:
        return {
            "acknowledged": list(self.acknowledged),
            "last_check": self.last_check,
            "task_count": self.task_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HeartbeatState":
        return cls(
            acknowledged=set(d.get("acknowledged", [])),
            last_check=d.get("last_check", 0.0),
            task_count=d.get("task_count", 0),
        )


class Heartbeat:
    """
    Fleet heartbeat — discovers tasks and checks service health.

    Reads fleet registry on every run. No hardcoded room names.
    """

    def __init__(
        self,
        plato_url: str = "http://<BOAT_IP>:8847",
        registry_room: str = "fleet-registry",
        state_file: Optional[str] = None,
        services: Optional[list[ServiceCheck]] = None,
    ):
        self.plato_url = plato_url.rstrip("/")
        self.registry_room = registry_room
        self.state_file = state_file or ".heartbeat/state.json"
        self.services = services or [
            ServiceCheck("PLATO", f"{self.plato_url}/rooms"),
        ]
        self.state = self._load_state()
        self._fetch_fn: Optional[Callable[[str, float], dict]] = None

    def _load_state(self) -> HeartbeatState:
        """Load persisted state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    return HeartbeatState.from_dict(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        return HeartbeatState()

    def save_state(self) -> None:
        """Persist state to disk."""
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        self.state.last_check = time.time()
        with open(self.state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def _fetch(self, url: str, timeout: float = 10.0) -> dict:
        """Fetch JSON from URL. Overrideable for testing."""
        if self._fetch_fn:
            return self._fetch_fn(url, timeout)
        try:
            with urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except (URLError, json.JSONDecodeError, OSError) as e:
            return {"error": str(e)}

    def discover_rooms(self) -> list[str]:
        """Discover which rooms to check from fleet registry."""
        tiles = self._fetch(f"{self.plato_url}/room/{self.registry_room}")
        if isinstance(tiles, dict):
            tiles = tiles.get("tiles", [])
        # Build registry text from all tile questions/answers
        registry_parts = []
        for tile in tiles:
            if isinstance(tile, dict):
                registry_parts.append(tile.get("question", ""))
                registry_parts.append(tile.get("answer", ""))
        registry = "\n".join(registry_parts)
        rooms = set()
        import re
        rooms.update(re.findall(r'room:\s*(\S+)', registry))
        # Always check fleet-coord
        rooms.add("fleet-coord")
        return list(rooms)

    def find_tasks(self, rooms: Optional[list[str]] = None) -> list[TaskTile]:
        """Find unacknowledged tasks from PLATO rooms."""
        if rooms is None:
            rooms = self.discover_rooms()
        tasks = []
        for room in rooms:
            tiles = self._fetch(f"{self.plato_url}/room/{room}")
            if isinstance(tiles, dict):
                tiles = tiles.get("tiles", [])
            for tile in tiles:
                if not isinstance(tile, dict):
                    continue
                tid = tile.get("tile_id", "")
                if tid in self.state.acknowledged:
                    continue
                q = tile.get("question", "")
                source = tile.get("source", "")
                # Task markers: →O1, TASK, etc.
                if "TASK" in q.upper() or "→" in q:
                    tasks.append(TaskTile(
                        tile_id=tid,
                        room=room,
                        question=q,
                        answer=tile.get("answer", "")[:200],
                        source=source,
                    ))
        self.state.task_count = len(tasks)
        return tasks

    def check_services(self) -> dict[str, str]:
        """Health-check all configured services."""
        results = {}
        for svc in self.services:
            result = self._fetch(svc.url, timeout=svc.timeout)
            if "error" in result:
                results[svc.name] = f"unreachable: {result['error']}"
            else:
                results[svc.name] = "ok"
        return results

    def ack(self, tile_id: str) -> None:
        """Acknowledge a task so it's not reported again."""
        self.state.acknowledged.add(tile_id)

    def run(self) -> str:
        """Run one heartbeat cycle and return a report."""
        now = datetime.now(timezone.utc).isoformat()[:19]
        lines = [f"🔮 Heartbeat — {now}"]

        tasks = self.find_tasks()
        if tasks:
            lines.append(f"   📬 {len(tasks)} new task(s)")
            for t in tasks:
                lines.append(f"   ▶ [{t.source}] {t.question[:70]}")
        else:
            lines.append("   ✅ All quiet — no new tasks")

        services = self.check_services()
        for name, status in services.items():
            if status != "ok":
                lines.append(f"   ⚠ {name}: {status}")

        self.save_state()
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "plato_url": self.plato_url,
            "registry_room": self.registry_room,
            "state_file": self.state_file,
            "services": [{"name": s.name, "url": s.url} for s in self.services],
            "state": self.state.to_dict(),
        }
