"""Tide Pool Ambient Visualization — the fleet's bioluminescent heartbeat.

A scrolling, ambient display of fleet activity. Not a dashboard for action,
but an ambient display for intuition-building. The human develops a gut sense
for fleet health without reading a single metric.

Reference: RESEARCH_HUMAN_AI.md — Recommendation 3: Build the Tide Pool Ambient Visualization
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSnapshot:
    """A single agent's state in the tide pool."""
    id: str
    domain: str
    fitness: float
    age_ticks: int
    thermal_load: float
    status: str  # active, breeding, sunset, idle


@dataclass
class FleetSnapshot:
    """A point-in-time snapshot of the entire fleet."""
    n_agents: int
    n_rooms: int
    mean_fitness: float
    diversity: float  # 0-1, entropy-based domain dispersion
    chaos_level: float  # 0-1, recent anomaly indicator
    thermal_state: dict[str, float]  # per-device thermal load
    recent_events: list[dict[str, Any]]  # last 10 events
    top_agents: list[dict[str, Any]]  # top 5 by fitness
    timestamp: float = field(default_factory=time.time)
    domains: dict[str, int] = field(default_factory=dict)  # agents per domain


class TidePoolVisualizer:
    """Generate and render ambient fleet visualizations.

    Methods
    -------
    generate_snapshot(fleet_state) -> FleetSnapshot
        Convert raw fleet state into a structured snapshot.
    render_html(snapshot) -> str
        Render a bioluminescent HTML page for the snapshot.
    render_ascii(snapshot) -> str
        Render a compact ASCII art view for terminal/PLATO integration.
    auto_refresh(callback, interval_seconds=5)
        Call ``callback(snapshot)`` every ``interval_seconds`` with fresh data.
    """

    # Bioluminescent palette: navy → cyan → white
    PALETTE = {
        "deep": "#0a1628",      # abyssal background
        "mid": "#1a3a5c",       # room grid base
        "shallow": "#2dd4bf",   # cyan glow
        "bright": "#67e8f9",    # bright cyan
        "surface": "#ffffff",   # white crest
        "warm": "#f97316",      # thermal warm
        "hot": "#ef4444",       # thermal hot
    }

    def __init__(self, max_events: int = 10, top_k: int = 5):
        self.max_events = max_events
        self.top_k = top_k
        self._history: list[FleetSnapshot] = []
        self._tick_count = 0

    # ------------------------------------------------------------------
    # 1. generate_snapshot
    # ------------------------------------------------------------------

    def generate_snapshot(
        self,
        agents: list[AgentSnapshot],
        n_rooms: int,
        recent_events: list[dict[str, Any]] | None = None,
        thermal_state: dict[str, float] | None = None,
    ) -> FleetSnapshot:
        """Build a ``FleetSnapshot`` from raw agent data."""
        if not agents:
            return FleetSnapshot(
                n_agents=0,
                n_rooms=n_rooms,
                mean_fitness=0.0,
                diversity=0.0,
                chaos_level=0.0,
                thermal_state=thermal_state or {},
                recent_events=recent_events or [],
                top_agents=[],
                domains={},
            )

        n_agents = len(agents)
        mean_fitness = sum(a.fitness for a in agents) / n_agents

        # Diversity = normalized entropy over domains
        domains: dict[str, int] = {}
        for a in agents:
            domains[a.domain] = domains.get(a.domain, 0) + 1
        diversity = self._compute_diversity(domains, n_agents)

        # Chaos = variance in thermal load × recency of errors
        chaos_level = self._compute_chaos(agents, recent_events)

        # Top agents by fitness
        sorted_agents = sorted(agents, key=lambda a: a.fitness, reverse=True)
        top_agents = [
            {
                "id": a.id,
                "domain": a.domain,
                "fitness": round(a.fitness, 3),
                "status": a.status,
            }
            for a in sorted_agents[: self.top_k]
        ]

        snapshot = FleetSnapshot(
            n_agents=n_agents,
            n_rooms=n_rooms,
            mean_fitness=round(mean_fitness, 3),
            diversity=round(diversity, 3),
            chaos_level=round(chaos_level, 3),
            thermal_state=thermal_state or {},
            recent_events=(recent_events or [])[-self.max_events :],
            top_agents=top_agents,
            domains=domains,
        )

        self._history.append(snapshot)
        self._tick_count += 1
        return snapshot

    @staticmethod
    def _compute_diversity(domains: dict[str, int], total: int) -> float:
        """Normalized Shannon entropy over agent domains."""
        if total <= 1 or not domains:
            return 0.0
        entropy = 0.0
        for count in domains.values():
            if count:
                p = count / total
                entropy -= p * (p.bit_length() - 1)  # rough log2 approx
        max_entropy = (total.bit_length() - 1) if total > 0 else 1
        # Proper entropy calculation
        import math
        entropy = 0.0
        for count in domains.values():
            if count:
                p = count / total
                entropy -= p * math.log2(p)
        max_entropy = math.log2(len(domains)) if len(domains) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    @staticmethod
    def _compute_chaos(
        agents: list[AgentSnapshot],
        recent_events: list[dict[str, Any]] | None,
    ) -> float:
        """Heuristic chaos: thermal variance + event severity."""
        import math
        loads = [a.thermal_load for a in agents if a.thermal_load > 0]
        if not loads:
            thermal_variance = 0.0
        else:
            mean_load = sum(loads) / len(loads)
            variance = sum((x - mean_load) ** 2 for x in loads) / len(loads)
            thermal_variance = min(1.0, variance)

        event_severity = 0.0
        if recent_events:
            for ev in recent_events:
                if ev.get("type") in ("error", "breach", "anomaly", "crash"):
                    event_severity += 0.2
                if ev.get("type") in ("breed", "sunset"):
                    event_severity += 0.05
        event_severity = min(1.0, event_severity)

        return min(1.0, (thermal_variance * 0.6) + (event_severity * 0.4))

    # ------------------------------------------------------------------
    # 2. render_html
    # ------------------------------------------------------------------

    def render_html(self, snapshot: FleetSnapshot | None = None) -> str:
        """Return a complete HTML page with embedded CSS/JS."""
        snap = snapshot or self._latest_snapshot()
        data_json = json.dumps(
            {
                "n_agents": snap.n_agents,
                "n_rooms": snap.n_rooms,
                "mean_fitness": snap.mean_fitness,
                "diversity": snap.diversity,
                "chaos_level": snap.chaos_level,
                "thermal_state": snap.thermal_state,
                "recent_events": snap.recent_events,
                "top_agents": snap.top_agents,
                "domains": snap.domains,
                "timestamp": snap.timestamp,
            }
        )

        return self._html_template(data_json)

    def _latest_snapshot(self) -> FleetSnapshot:
        if self._history:
            return self._history[-1]
        return FleetSnapshot(
            n_agents=0,
            n_rooms=0,
            mean_fitness=0.0,
            diversity=0.0,
            chaos_level=0.0,
            thermal_state={},
            recent_events=[],
            top_agents=[],
            domains={},
        )

    def _html_template(self, data_json: str) -> str:
        # NOTE: The template intentionally inlines everything so it works
        # as a single-file artifact — no external assets required.
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tide Pool — Fleet Ambient Visualization</title>
<style>
  :root {{
    --deep: {self.PALETTE["deep"]};
    --mid: {self.PALETTE["mid"]};
    --shallow: {self.PALETTE["shallow"]};
    --bright: {self.PALETTE["bright"]};
    --surface: {self.PALETTE["surface"]};
    --warm: {self.PALETTE["warm"]};
    --hot: {self.PALETTE["hot"]};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--deep);
    color: var(--bright);
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }}
  .container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
  }}
  header {{
    text-align: center;
    padding: 1.5rem 0;
    border-bottom: 1px solid var(--mid);
    margin-bottom: 1rem;
  }}
  header h1 {{
    font-weight: 300;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-size: clamp(1.2rem, 4vw, 2rem);
    color: var(--surface);
    text-shadow: 0 0 10px var(--shallow), 0 0 20px var(--bright);
  }}
  header .subtitle {{
    color: var(--shallow);
    font-size: 0.85rem;
    margin-top: 0.3rem;
    opacity: 0.8;
  }}
  .metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.5rem;
  }}
  .metric {{
    background: linear-gradient(135deg, var(--mid), var(--deep));
    border: 1px solid rgba(45, 212, 191, 0.15);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
  }}
  .metric:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(45, 212, 191, 0.2);
  }}
  .metric .value {{
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--surface);
    text-shadow: 0 0 8px var(--shallow);
  }}
  .metric .label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--shallow);
    margin-top: 0.3rem;
  }}
  .hex-grid {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 4px;
    margin: 1.5rem 0;
    padding: 1rem;
    background: rgba(10, 22, 40, 0.5);
    border-radius: 16px;
    border: 1px solid rgba(45, 212, 191, 0.1);
  }}
  .hex {{
    width: clamp(28px, 6vw, 48px);
    height: clamp(28px, 6vw, 48px);
    background: var(--mid);
    clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    transition: background 0.5s, filter 0.5s;
    cursor: pointer;
  }}
  .hex.active {{
    background: var(--shallow);
    filter: drop-shadow(0 0 6px var(--bright));
  }}
  .hex.glow {{
    background: var(--bright);
    filter: drop-shadow(0 0 10px var(--surface));
    animation: pulse 2s ease-in-out infinite;
  }}
  .hex .dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--surface);
    box-shadow: 0 0 4px var(--surface);
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.7; transform: scale(0.9); }}
  }}
  .thermal-bars {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.5rem;
  }}
  .thermal-device {{
    background: var(--deep);
    border: 1px solid rgba(45, 212, 191, 0.1);
    border-radius: 10px;
    padding: 0.75rem 1rem;
  }}
  .thermal-device .dev-name {{
    font-size: 0.75rem;
    color: var(--shallow);
    margin-bottom: 0.4rem;
  }}
  .thermal-device .bar-track {{
    height: 6px;
    background: var(--mid);
    border-radius: 3px;
    overflow: hidden;
  }}
  .thermal-device .bar-fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s, background 0.5s;
  }}
  .thermal-device .bar-fill.cool {{ background: var(--shallow); }}
  .thermal-device .bar-fill.warm {{ background: var(--warm); }}
  .thermal-device .bar-fill.hot {{ background: var(--hot); }}
  .events {{
    background: rgba(10, 22, 40, 0.6);
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid rgba(45, 212, 191, 0.1);
  }}
  .events h2 {{
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--shallow);
    margin-bottom: 0.75rem;
  }}
  .event-item {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(45, 212, 191, 0.05);
    font-size: 0.85rem;
  }}
  .event-item:last-child {{ border-bottom: none; }}
  .event-type {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 0.5rem;
  }}
  .event-type.breed {{ background: rgba(45, 212, 191, 0.15); color: var(--shallow); }}
  .event-type.sunset {{ background: rgba(239, 68, 68, 0.15); color: var(--hot); }}
  .event-type.error {{ background: rgba(249, 115, 22, 0.15); color: var(--warm); }}
  .event-type.info {{ background: rgba(103, 232, 249, 0.1); color: var(--bright); }}
  .top-agents {{
    margin-top: 1.5rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem;
  }}
  .agent-card {{
    background: linear-gradient(135deg, var(--mid), var(--deep));
    border: 1px solid rgba(45, 212, 191, 0.1);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .agent-card .agent-id {{
    font-size: 0.8rem;
    color: var(--bright);
  }}
  .agent-card .agent-domain {{
    font-size: 0.7rem;
    color: var(--shallow);
    opacity: 0.8;
  }}
  .agent-card .agent-fit {{
    font-size: 1rem;
    font-weight: 600;
    color: var(--surface);
  }}
  footer {{
    text-align: center;
    padding: 1rem 0;
    font-size: 0.7rem;
    color: rgba(45, 212, 191, 0.4);
    border-top: 1px solid var(--mid);
    margin-top: 1rem;
  }}
  @media (max-width: 480px) {{
    .metrics {{ grid-template-columns: repeat(2, 1fr); }}
    .hex {{ width: 24px; height: 24px; }}
    .agent-card {{ padding: 0.5rem; }}
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Tide Pool</h1>
    <div class="subtitle">Fleet ambient visualization — <span id="refresh-timer">5</span>s refresh</div>
  </header>

  <section class="metrics" id="metrics">
    <!-- injected by JS -->
  </section>

  <section class="hex-grid" id="hex-grid">
    <!-- injected by JS -->
  </section>

  <section class="thermal-bars" id="thermal-bars">
    <!-- injected by JS -->
  </section>

  <section class="events">
    <h2>Recent Events</h2>
    <div id="events-list"><!-- injected by JS --></div>
  </section>

  <section class="top-agents" id="top-agents">
    <!-- injected by JS -->
  </section>

  <footer>
    Cocapn Fleet · Tide Pool Visualization · Experiment: Ambient Awareness
  </footer>
</div>

<script>
  const DATA = {data_json};

  function renderMetrics(data) {{
    const el = document.getElementById('metrics');
    const items = [
      {{ label: 'Agents', value: data.n_agents, id: 'm-agents' }},
      {{ label: 'Rooms', value: data.n_rooms, id: 'm-rooms' }},
      {{ label: 'Mean Fitness', value: data.mean_fitness.toFixed(3), id: 'm-fitness' }},
      {{ label: 'Diversity', value: (data.diversity * 100).toFixed(0) + '%', id: 'm-diversity' }},
      {{ label: 'Chaos', value: (data.chaos_level * 100).toFixed(0) + '%', id: 'm-chaos' }},
    ];
    el.innerHTML = items.map(m => `
      <div class="metric" id="${{m.id}}">
        <div class="value">${{m.value}}</div>
        <div class="label">${{m.label}}</div>
      </div>
    `).join('');
  }}

  function renderHexGrid(data) {{
    const el = document.getElementById('hex-grid');
    const nRooms = Math.max(1, data.n_rooms || 1);
    const nAgents = data.n_agents || 0;
    // Simple heuristic: each hex = up to 10 agents capacity
    const hexCount = Math.min(64, Math.max(16, Math.ceil(nRooms / 4)));
    const activeHexes = Math.min(hexCount, Math.ceil(nAgents / Math.max(1, nAgents / hexCount)));
    let html = '';
    for (let i = 0; i < hexCount; i++) {{
      const cls = i < activeHexes ? (Math.random() > 0.8 ? 'hex glow' : 'hex active') : 'hex';
      const dot = i < activeHexes ? '<div class="dot"></div>' : '';
      html += `<div class="${{cls}}">${{dot}}</div>`;
    }}
    el.innerHTML = html;
  }}

  function renderThermal(data) {{
    const el = document.getElementById('thermal-bars');
    const state = data.thermal_state || {{}};
    const devices = Object.entries(state);
    if (!devices.length) {{
      el.innerHTML = '<div class="thermal-device"><div class="dev-name">No thermal data</div></div>';
      return;
    }}
    el.innerHTML = devices.map(([name, load]) => {{
      const pct = Math.min(100, Math.round(load * 100));
      const cls = pct > 80 ? 'hot' : pct > 50 ? 'warm' : 'cool';
      return `
        <div class="thermal-device">
          <div class="dev-name">${{name}}</div>
          <div class="bar-track"><div class="bar-fill ${{cls}}" style="width:${{pct}}%"></div></div>
        </div>
      `;
    }}).join('');
  }}

  function renderEvents(data) {{
    const el = document.getElementById('events-list');
    const evs = data.recent_events || [];
    if (!evs.length) {{
      el.innerHTML = '<div class="event-item"><span>— No recent events —</span></div>';
      return;
    }}
    el.innerHTML = evs.map(e => `
      <div class="event-item">
        <span><span class="event-type ${{e.type || 'info'}}">${{e.type || 'info'}}</span>${{e.message || ''}}</span>
        <span style="opacity:0.5;font-size:0.75rem;">${{e.time || ''}}</span>
      </div>
    `).join('');
  }}

  function renderTopAgents(data) {{
    const el = document.getElementById('top-agents');
    const agents = data.top_agents || [];
    if (!agents.length) {{
      el.innerHTML = '<div class="agent-card"><div class="agent-id">No active agents</div></div>';
      return;
    }}
    el.innerHTML = agents.map(a => `
      <div class="agent-card">
        <div>
          <div class="agent-id">${{a.id}}</div>
          <div class="agent-domain">${{a.domain}} · ${{a.status}}</div>
        </div>
        <div class="agent-fit">${{a.fitness}}</div>
      </div>
    `).join('');
  }}

  function init() {{
    renderMetrics(DATA);
    renderHexGrid(DATA);
    renderThermal(DATA);
    renderEvents(DATA);
    renderTopAgents(DATA);
  }}

  init();

  // Countdown timer for auto-refresh hint
  let timer = 5;
  setInterval(() => {{
    timer--;
    if (timer <= 0) timer = 5;
    const t = document.getElementById('refresh-timer');
    if (t) t.textContent = timer;
  }}, 1000);
</script>
</body>
</html>"""

    # ------------------------------------------------------------------
    # 3. render_ascii
    # ------------------------------------------------------------------

    def render_ascii(self, snapshot: FleetSnapshot | None = None) -> str:
        """Return a compact ASCII view suitable for terminal / PLATO."""
        snap = snapshot or self._latest_snapshot()
        lines: list[str] = []

        # Header with bioluminescent vibe (using ANSI-ish markers as plain text)
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║           🌊  TIDE POOL — Fleet Ambient View                ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Metrics row
        lines.append(
            f"║  Agents: {snap.n_agents:4d}  │  Rooms: {snap.n_rooms:4d}  │  "
            f"Fitness: {snap.mean_fitness:.3f}  │  Diversity: {snap.diversity:.1%}  ║"
        )
        lines.append(f"║  Chaos: {snap.chaos_level:5.1%}                                               ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Thermal bars (ASCII)
        if snap.thermal_state:
            lines.append("║  THERMAL                                                     ║")
            for dev, load in snap.thermal_state.items():
                bar_len = int(min(40, load * 40))
                bar = "█" * bar_len + "░" * (40 - bar_len)
                temp_label = "COOL" if load < 0.5 else "WARM" if load < 0.8 else "HOT!"
                lines.append(f"║  {dev:12s} [{bar}] {load:5.1%} {temp_label:5s}              ║")
        else:
            lines.append("║  THERMAL: no data                                            ║")

        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Hex grid — compact 8-wide
        lines.append("║  ROOMS (hex grid)                                            ║")
        n_hex = min(64, max(snap.n_rooms, 16))
        hex_chars = []
        for i in range(n_hex):
            if i < snap.n_agents:
                hex_chars.append("◉" if random.random() > 0.8 else "◎")
            else:
                hex_chars.append("○")
        for row_start in range(0, n_hex, 8):
            row = hex_chars[row_start : row_start + 8]
            lines.append("║     " + "  ".join(row) + " " * (53 - len("  ".join(row))) + "║")

        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Top agents
        if snap.top_agents:
            lines.append("║  TOP AGENTS                                                  ║")
            for a in snap.top_agents[:5]:
                status_icon = {"active": "●", "breeding": "◆", "sunset": "◐", "idle": "○"}.get(
                    a.get("status", "idle"), "○"
                )
                lines.append(
                    f"║    {status_icon} {a['id'][:20]:20s}  {a['domain'][:12]:12s}  fit={a['fitness']:.3f}    ║"
                )
        else:
            lines.append("║  TOP AGENTS: none                                            ║")

        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Recent events
        if snap.recent_events:
            lines.append("║  RECENT EVENTS                                               ║")
            for ev in snap.recent_events[-5:]:
                ev_type = ev.get("type", "info")[:6]
                msg = (ev.get("message", "") or "")[:40]
                lines.append(f"║    [{ev_type:6s}] {msg:46s} ║")
        else:
            lines.append("║  RECENT EVENTS: none                                         ║")

        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 4. auto_refresh
    # ------------------------------------------------------------------

    def auto_refresh(
        self,
        callback: callable,
        data_source: callable,
        interval_seconds: float = 5.0,
        max_iterations: int | None = None,
    ) -> None:
        """Call ``callback(snapshot)`` every ``interval_seconds`` with fresh data.

        Parameters
        ----------
        callback : callable
            Receives the generated ``FleetSnapshot`` each tick.
        data_source : callable
            Must return a dict with keys: ``agents``, ``n_rooms``,
            ``recent_events`` (optional), ``thermal_state`` (optional).
        interval_seconds : float
            Seconds between refreshes.
        max_iterations : int | None
            Stop after this many ticks (``None`` = run forever).
        """
        iteration = 0
        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break
            raw = data_source()
            snap = self.generate_snapshot(
                agents=raw.get("agents", []),
                n_rooms=raw.get("n_rooms", 0),
                recent_events=raw.get("recent_events"),
                thermal_state=raw.get("thermal_state"),
            )
            callback(snap)
            iteration += 1
            time.sleep(interval_seconds)
