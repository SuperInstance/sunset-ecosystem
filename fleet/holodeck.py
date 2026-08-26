"""Holodeck — 3D Room Grid Visualizer for the Cocapn Fleet's Sunset Ecosystem.

Generates a spatial model of PLATO rooms and agent avatars, then exports
interactive Three.js visualizations.  Works with real PLATO room state or the
bundled MockPlatoSource for offline demos.

Design reference
----------------
- Dieter Rams: minimal, functional, no decorative noise.
- Moebius: deep abyssal blues, bioluminescent greens, warm amber accents.
- Fleet metaphor: rooms are hermit-crab shells; agents are luminescent motes.

Usage
-----
    hd = Holodeck()
    hd.add_room("alpha", (0, 0, 0), capacity=10)
    hd.add_agent("a1", "alpha", {"ethos": 0.7, "pathos": 0.5, "logos": 0.9})
    hd.export_html("holodeck.html")
"""

from __future__ import annotations

__all__ = [
    "Holodeck",
    "RoomNode",
    "AgentAvatar",
    "MockPlatoSource",
    "room_color_for_diversity",
    "agent_color_for_phase",
]

import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── color palette ─────────────────────────────────────────
# Moebius-inspired abyssal palette
_PALETTE = {
    "abyss_blue": 0x0A1628,
    "deep_teal": 0x0D3B3B,
    "biolum_green": 0x39FF14,
    "amber": 0xFFB347,
    "crimson": 0xDC143C,
    "shell_white": 0xF0F0E6,
    "ash_gray": 0x8A8A8A,
}


def room_color_for_diversity(diversity: float) -> int:
    """Map 0.0→1.0 diversity to a room cube colour.

    blue (low)  → green (mid) → red (high)
    """
    if diversity < 0.33:
        return _PALETTE["abyss_blue"]  # low diversity
    if diversity < 0.66:
        return _PALETTE["biolum_green"]  # medium
    return _PALETTE["crimson"]  # high


def agent_color_for_phase(phase: str) -> int:
    """Map agent lifecycle phase to sphere colour."""
    mapping = {
        "incubating": _PALETTE["shell_white"],
        "competing": _PALETTE["amber"],
        "breeding": _PALETTE["biolum_green"],
        "sunsetting": _PALETTE["ash_gray"],
    }
    return mapping.get(phase, _PALETTE["shell_white"])


# ── data structures ─────────────────────────────────────


@dataclass
class AgentAvatar:
    """A single agent inside the 3D model."""

    agent_id: str
    room_id: Optional[str] = None
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    phase: str = "incubating"  # incubating | competing | breeding | sunsetting
    trinity_scores: Dict[str, float] = field(default_factory=dict)
    born_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if isinstance(self.position, (list, tuple)):
            self.position = np.array(self.position, dtype=np.float32)
        if isinstance(self.velocity, (list, tuple)):
            self.velocity = np.array(self.velocity, dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "room_id": self.room_id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "phase": self.phase,
            "trinity_scores": dict(self.trinity_scores),
            "born_at": self.born_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentAvatar":
        return cls(
            agent_id=d["agent_id"],
            room_id=d.get("room_id"),
            position=np.array(d.get("position", [0, 0, 0]), dtype=np.float32),
            velocity=np.array(d.get("velocity", [0, 0, 0]), dtype=np.float32),
            phase=d.get("phase", "incubating"),
            trinity_scores=d.get("trinity_scores", {}),
            born_at=d.get("born_at", 0.0),
        )


@dataclass
class RoomNode:
    """A room shell in 3D space."""

    room_id: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    capacity: int = 10
    occupancy: int = 0
    diversity_score: float = 0.0
    thermal_state: float = 0.0  # 0=cool … 1=hot
    agents: List[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.position, (list, tuple)):
            self.position = np.array(self.position, dtype=np.float32)

    @property
    def is_overcapacity(self) -> bool:
        return self.occupancy > self.capacity

    def to_dict(self) -> dict:
        return {
            "room_id": self.room_id,
            "position": self.position.tolist(),
            "capacity": self.capacity,
            "occupancy": self.occupancy,
            "diversity_score": self.diversity_score,
            "thermal_state": self.thermal_state,
            "agents": list(self.agents),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RoomNode":
        return cls(
            room_id=d["room_id"],
            position=np.array(d.get("position", [0, 0, 0]), dtype=np.float32),
            capacity=d.get("capacity", 10),
            occupancy=d.get("occupancy", 0),
            diversity_score=d.get("diversity_score", 0.0),
            thermal_state=d.get("thermal_state", 0.0),
            agents=list(d.get("agents", [])),
        )


# ── Holodeck engine ─────────────────────────────────────


class Holodeck:
    """3D spatial model of fleet rooms + agent avatars."""

    def __init__(self):
        self._rooms: Dict[str, RoomNode] = {}
        self._agents: Dict[str, AgentAvatar] = {}
        self._connections: set[Tuple[str, str]] = set()
        self._lock = threading.RLock()
        self._tick_count = 0

    # ── room ops ─────────────────────────────────────────

    def add_room(
        self,
        room_id: str,
        position: Tuple[float, float, float],
        capacity: int = 10,
    ) -> None:
        with self._lock:
            if room_id in self._rooms:
                logger.warning("Room %s already registered — overwriting", room_id)
            self._rooms[room_id] = RoomNode(
                room_id=room_id,
                position=np.array(position, dtype=np.float32),
                capacity=capacity,
            )

    def update_room(
        self,
        room_id: str,
        occupancy: Optional[int] = None,
        diversity: Optional[float] = None,
        thermal: Optional[float] = None,
    ) -> None:
        with self._lock:
            room = self._rooms.get(room_id)
            if room is None:
                raise KeyError(f"Room '{room_id}' not registered")
            if occupancy is not None:
                room.occupancy = occupancy
            if diversity is not None:
                room.diversity_score = max(0.0, min(1.0, diversity))
            if thermal is not None:
                room.thermal_state = max(0.0, min(1.0, thermal))

    def remove_room(self, room_id: str) -> None:
        with self._lock:
            room = self._rooms.pop(room_id, None)
            if room:
                for aid in list(room.agents):
                    self._agents.pop(aid, None)
            self._connections = {c for c in self._connections if room_id not in c}

    # ── agent ops ────────────────────────────────────────

    def add_agent(
        self,
        agent_id: str,
        room_id: str,
        trinity_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        with self._lock:
            if room_id not in self._rooms:
                raise KeyError(f"Room '{room_id}' not registered")
            avatar = AgentAvatar(
                agent_id=agent_id,
                room_id=room_id,
                trinity_scores=trinity_scores or {},
            )
            # Jitter position inside room volume
            avatar.position = self._jitter_inside(room_id)
            self._agents[agent_id] = avatar
            self._rooms[room_id].agents.append(agent_id)
            self._rooms[room_id].occupancy = len(self._rooms[room_id].agents)

    def move_agent(self, agent_id: str, from_room: str, to_room: str) -> None:
        with self._lock:
            avatar = self._agents.get(agent_id)
            if avatar is None:
                raise KeyError(f"Agent '{agent_id}' not found")
            if from_room not in self._rooms or to_room not in self._rooms:
                raise KeyError(f"Room not found in transition {from_room} → {to_room}")

            # Remove from old
            if agent_id in self._rooms[from_room].agents:
                self._rooms[from_room].agents.remove(agent_id)
            self._rooms[from_room].occupancy = len(self._rooms[from_room].agents)

            # Add to new
            self._rooms[to_room].agents.append(agent_id)
            self._rooms[to_room].occupancy = len(self._rooms[to_room].agents)

            avatar.room_id = to_room
            avatar.position = self._jitter_inside(to_room)
            self._connections.add(tuple(sorted((from_room, to_room))))

    def remove_agent(self, agent_id: str) -> None:
        with self._lock:
            avatar = self._agents.pop(agent_id, None)
            if avatar and avatar.room_id and avatar.room_id in self._rooms:
                room = self._rooms[avatar.room_id]
                if agent_id in room.agents:
                    room.agents.remove(agent_id)
                room.occupancy = len(room.agents)

    def set_agent_phase(self, agent_id: str, phase: str) -> None:
        with self._lock:
            avatar = self._agents.get(agent_id)
            if avatar is None:
                raise KeyError(f"Agent '{agent_id}' not found")
            avatar.phase = phase

    # ── helpers ──────────────────────────────────────────

    def _jitter_inside(self, room_id: str) -> np.ndarray:
        """Return a random point inside the room cube."""
        room = self._rooms[room_id]
        scale = float(room.capacity) * 0.15  # shell size heuristic
        jitter = (np.random.rand(3).astype(np.float32) - 0.5) * 2 * scale
        return room.position + jitter

    # ── queries ──────────────────────────────────────────

    def get_scene(self) -> dict:
        """Complete scene graph: rooms, agents, connections."""
        with self._lock:
            return {
                "rooms": {rid: r.to_dict() for rid, r in self._rooms.items()},
                "agents": {aid: a.to_dict() for aid, a in self._agents.items()},
                "connections": [list(c) for c in self._connections],
                "tick": self._tick_count,
            }

    def snapshot(self) -> dict:
        """JSON-serializable scene dict (alias for get_scene)."""
        return self.get_scene()

    def room_count(self) -> int:
        with self._lock:
            return len(self._rooms)

    def agent_count(self) -> int:
        with self._lock:
            return len(self._agents)

    def get_room(self, room_id: str) -> Optional[RoomNode]:
        with self._lock:
            return self._rooms.get(room_id)

    def get_agent(self, agent_id: str) -> Optional[AgentAvatar]:
        with self._lock:
            return self._agents.get(agent_id)

    # ── HTML export ──────────────────────────────────────

    def export_html(self, output_path: str) -> None:
        """Write a self-contained Three.js HTML file."""
        scene = self.get_scene()
        html = self._generate_threejs_html(scene)
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info("Holodeck exported to %s", output_path)

    def _generate_threejs_html(self, scene: dict) -> str:
        rooms = scene["rooms"]
        agents = scene["agents"]
        connections = scene["connections"]

        rooms_js = json.dumps(rooms)
        agents_js = json.dumps(agents)
        connections_js = json.dumps(connections)

        # colour helpers (mirror Python logic in JS)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cocapn Fleet — Holodeck</title>
<style>
  body {{ margin: 0; overflow: hidden; background: #050a12; font-family: 'Segoe UI', sans-serif; }}
  #tooltip {{
    position: absolute; pointer-events: none; background: rgba(10,22,40,0.92);
    color: #e0e6ed; padding: 8px 12px; border-radius: 6px; font-size: 12px;
    border: 1px solid #1a3a4a; display: none; max-width: 280px; z-index: 10;
  }}
  #header {{
    position: absolute; top: 10px; left: 15px; color: #8ab4c7;
    font-size: 14px; z-index: 5; pointer-events: none;
  }}
</style>
</head>
<body>
<div id="header">🦀 Cocapn Fleet Holodeck — Rooms: <span id="roomCount">0</span> | Agents: <span id="agentCount">0</span></div>
<div id="tooltip"></div>
<script type="importmap">{{"imports":{{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}}}</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const roomsData = {rooms_js};
const agentsData = {agents_js};
const connectionsData = {connections_js};

function roomColor(d) {{
  if (d < 0.33) return 0x0a1628;
  if (d < 0.66) return 0x39ff14;
  return 0xdc143c;
}}

function agentColor(phase) {{
  const map = {{
    incubating: 0xf0f0e6,
    competing: 0xffb347,
    breeding: 0x39ff14,
    sunsetting: 0x8a8a8a,
  }};
  return map[phase] || 0xf0f0e6;
}}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050a12);
scene.fog = new THREE.FogExp2(0x050a12, 0.035);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 500);
camera.position.set(18, 14, 18);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.autoRotate = true;
controls.autoRotateSpeed = 0.8;
controls.enableDamping = true;

// lights
const ambient = new THREE.AmbientLight(0x404040, 1.5);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xaaccff, 1.2);
dirLight.position.set(10, 20, 10);
scene.add(dirLight);
const point = new THREE.PointLight(0x39ff14, 0.6, 50);
point.position.set(0, 10, 0);
scene.add(point);

const roomMeshes = {{}};
const agentMeshes = {{}};
const tooltip = document.getElementById('tooltip');

// draw rooms
for (const [rid, r] of Object.entries(roomsData)) {{
  const size = Math.max(1.5, r.capacity * 0.4);
  const geometry = new THREE.BoxGeometry(size, size, size);
  const material = new THREE.MeshStandardMaterial({{
    color: roomColor(r.diversity_score),
    transparent: true,
    opacity: 0.55,
    roughness: 0.3,
    metalness: 0.1,
  }});
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...r.position);
  mesh.userData = {{ type: 'room', ...r }};
  scene.add(mesh);
  roomMeshes[rid] = mesh;

  // wireframe shell
  const edges = new THREE.EdgesGeometry(geometry);
  const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({{ color: 0x1a4a5a, transparent: true, opacity: 0.4 }}));
  line.position.copy(mesh.position);
  scene.add(line);
}}

// draw agents
for (const [aid, a] of Object.entries(agentsData)) {{
  const geometry = new THREE.SphereGeometry(0.25, 16, 16);
  const material = new THREE.MeshStandardMaterial({{
    color: agentColor(a.phase),
    emissive: agentColor(a.phase),
    emissiveIntensity: 0.6,
    roughness: 0.2,
  }});
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...a.position);
  mesh.userData = {{ type: 'agent', ...a }};
  scene.add(mesh);
  agentMeshes[aid] = mesh;
}}

// connections
for (const [a, b] of connectionsData) {{
  if (roomMeshes[a] && roomMeshes[b]) {{
    const pts = [roomMeshes[a].position.clone(), roomMeshes[b].position.clone()];
    const geometry = new THREE.BufferGeometry().setFromPoints(pts);
    const material = new THREE.LineBasicMaterial({{ color: 0x2a5a6a, transparent: true, opacity: 0.25 }});
    scene.add(new THREE.Line(geometry, material));
  }}
}}

// tooltip raycaster
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

window.addEventListener('mousemove', (e) => {{
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(scene.children);
  let found = null;
  for (const h of hits) {{
    if (h.object.userData && (h.object.userData.type === 'room' || h.object.userData.type === 'agent')) {{
      found = h.object.userData; break;
    }}
  }}
  if (found) {{
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 12) + 'px';
    tooltip.style.top = (e.clientY + 12) + 'px';
    if (found.type === 'room') {{
      tooltip.innerHTML = `<strong>${{found.room_id}}</strong><br>capacity: ${{found.capacity}}<br>occupancy: ${{found.occupancy}}<br>diversity: ${{found.diversity_score.toFixed(2)}}<br>thermal: ${{found.thermal_state.toFixed(2)}}`;
    }} else {{
      const t = JSON.stringify(found.trinity_scores || {{}});
      tooltip.innerHTML = `<strong>${{found.agent_id}}</strong><br>phase: ${{found.phase}}<br>room: ${{found.room_id}}<br>trinity: ${{t}}`;
    }}
  }} else {{
    tooltip.style.display = 'none';
  }}
}});

// header stats
document.getElementById('roomCount').textContent = Object.keys(roomsData).length;
document.getElementById('agentCount').textContent = Object.keys(agentsData).length;

// animate
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  // gentle bob for agents
  const now = performance.now() * 0.001;
  for (const m of Object.values(agentMeshes)) {{
    m.position.y += Math.sin(now + m.id) * 0.002;
  }}
  renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>
"""

    # ── feed from external source ────────────────────────

    def ingest_mock_source(self, source: "MockPlatoSource") -> None:
        """Pull current state from a MockPlatoSource."""
        for rid, rstate in source.get_room_states().items():
            if rid not in self._rooms:
                self.add_room(rid, rstate["position"], rstate["capacity"])
            self.update_room(
                rid,
                occupancy=rstate["occupancy"],
                diversity=rstate["diversity"],
                thermal=rstate["thermal"],
            )
        # sync agents
        for aid, astate in source.get_agent_states().items():
            if aid not in self._agents:
                self.add_agent(aid, astate["room_id"], astate.get("trinity"))
            else:
                cur_room = self._agents[aid].room_id
                new_room = astate["room_id"]
                if cur_room != new_room and new_room is not None:
                    self.move_agent(aid, cur_room, new_room)
            self.set_agent_phase(aid, astate.get("phase", "incubating"))
        self._tick_count += 1


# ── Mock PLATO Source ───────────────────────────────────


class MockPlatoSource:
    """Simulates PLATO room state for offline demos / unit tests."""

    PHASES = ["incubating", "competing", "breeding", "sunsetting"]

    def __init__(
        self,
        room_count: int = 10,
        agent_count: int = 50,
        movement_prob: float = 0.15,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.room_count = room_count
        self.agent_count = agent_count
        self.movement_prob = movement_prob
        self._tick = 0

        self._rooms: Dict[str, dict] = {}
        self._agents: Dict[str, dict] = {}
        self._init_topology()

    def _init_topology(self):
        # 3D grid positions: 2×2×2 plus extras
        positions = []
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    positions.append((x * 5.0, y * 5.0, z * 5.0))
        # extras fill outward
        while len(positions) < self.room_count:
            positions.append(
                (
                    self.rng.uniform(-2, 12),
                    self.rng.uniform(-2, 12),
                    self.rng.uniform(-2, 12),
                )
            )

        for i in range(self.room_count):
            rid = f"room_{i:02d}"
            self._rooms[rid] = {
                "position": positions[i],
                "capacity": self.rng.randint(5, 15),
                "occupancy": 0,
                "diversity": self.rng.random(),
                "thermal": self.rng.random() * 0.3,
            }

        room_ids = list(self._rooms.keys())
        for j in range(self.agent_count):
            aid = f"agent_{j:03d}"
            room = self.rng.choice(room_ids)
            self._agents[aid] = {
                "room_id": room,
                "phase": self.rng.choice(self.PHASES),
                "trinity": {
                    "ethos": round(self.rng.random(), 2),
                    "pathos": round(self.rng.random(), 2),
                    "logos": round(self.rng.random(), 2),
                },
            }
            self._rooms[room]["occupancy"] += 1

    def tick(self) -> None:
        """Advance simulation by one step."""
        self._tick += 1
        room_ids = list(self._rooms.keys())

        # thermal drift
        for r in self._rooms.values():
            r["thermal"] = max(
                0.0, min(1.0, r["thermal"] + self.np_rng.normal(0, 0.02))
            )
            # diversity oscillates slowly
            r["diversity"] = max(
                0.0, min(1.0, r["diversity"] + self.np_rng.normal(0, 0.01))
            )

        # agent movement
        for aid, astate in self._agents.items():
            if self.rng.random() < self.movement_prob:
                old = astate["room_id"]
                new = self.rng.choice(room_ids)
                if new != old:
                    self._rooms[old]["occupancy"] -= 1
                    self._rooms[new]["occupancy"] += 1
                    astate["room_id"] = new
            # phase transitions
            if self.rng.random() < 0.05:
                astate["phase"] = self.rng.choice(self.PHASES)

    def get_room_states(self) -> Dict[str, dict]:
        return {k: dict(v) for k, v in self._rooms.items()}

    def get_agent_states(self) -> Dict[str, dict]:
        return {k: dict(v) for k, v in self._agents.items()}

    def get_room_count(self) -> int:
        return len(self._rooms)

    def get_agent_count(self) -> int:
        return len(self._agents)
