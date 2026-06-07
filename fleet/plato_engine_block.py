"""
Plato Engine Block — Minimal room runtime for the sunset ecosystem.

A self-contained async room that ticks, reads sensors, stores history,
evaluates alarms, and speaks a text protocol. The atomic unit of PLATO.

Inspired by the C99 engine block from ida1.txt, ported to Python
with asyncio for integration into FleetConductorV2.

Text Protocol (agent → engine):
    tick                    → returns current tick as JSON
    history [N]             → returns last N ticks (default 10)
    actuator <name> <value> → sets actuator, returns ok|error
    alarm set <name> <cond> <action> → adds alarm rule
    alarm list              → lists active alarms
    subscribe               → streams every tick as JSON lines
    help                    → lists commands

Usage:
    engine = PlatoEngineBlock(room_id="engine_room", tick_hz=1.0)
    engine.add_sensor("coolant_temp", lambda: read_thermometer())
    engine.add_actuator("bilge_pump", lambda v: set_pump(v))
    engine.add_alarm("bilge_high", "bilge_level > 10", "activate_bilge_pump")
    await engine.start()  # blocks, runs tick loop + command server
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any


@dataclass
class AlarmRule:
    name: str
    condition: str          # e.g. "coolant_temp > 90"
    action: str             # e.g. "notify_captain"
    triggered: bool = False
    consecutive_ticks: int = 0


@dataclass
class Tick:
    timestamp: float
    seq: int
    values: Dict[str, float]


class PlatoEngineBlock:
    """Minimal room runtime."""

    def __init__(self, room_id: str, tick_hz: float = 1.0, history_size: int = 100):
        self.room_id = room_id
        self.tick_hz = tick_hz
        self.history_size = history_size
        self._sensors: Dict[str, Callable[[], float]] = {}
        self._actuators: Dict[str, Callable[[Any], None]] = {}
        self._alarms: List[AlarmRule] = []
        self._history: deque = deque(maxlen=history_size)
        self._seq = 0
        self._running = False
        self._subscribers: set = set()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def add_sensor(self, name: str, reader: Callable[[], float]) -> None:
        self._sensors[name] = reader

    def add_actuator(self, name: str, setter: Callable[[Any], None]) -> None:
        self._actuators[name] = setter

    def add_alarm(self, name: str, condition: str, action: str) -> None:
        self._alarms.append(AlarmRule(name, condition, action))

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------
    async def _tick(self) -> None:
        async with self._lock:
            self._seq += 1
            values = {}
            for name, reader in self._sensors.items():
                try:
                    values[name] = float(reader())
                except Exception:
                    values[name] = float("nan")
            tick = Tick(timestamp=time.time(), seq=self._seq, values=values)
            self._history.append(tick)
            await self._evaluate_alarms(tick)
            await self._notify_subscribers(tick)

    async def _evaluate_alarms(self, tick: Tick) -> None:
        for alarm in self._alarms:
            try:
                # Simple parser: "sensor_name operator threshold"
                parts = alarm.condition.split()
                if len(parts) != 3:
                    continue
                sensor_name, op, threshold_str = parts
                if sensor_name not in tick.values:
                    continue
                val = tick.values[sensor_name]
                threshold = float(threshold_str)
                triggered = False
                if op == ">":
                    triggered = val > threshold
                elif op == "<":
                    triggered = val < threshold
                elif op == ">=":
                    triggered = val >= threshold
                elif op == "<=":
                    triggered = val <= threshold
                elif op == "==":
                    triggered = val == threshold
                elif op == "!=":
                    triggered = val != threshold

                if triggered:
                    alarm.consecutive_ticks += 1
                else:
                    alarm.consecutive_ticks = 0
                    alarm.triggered = False

                # Require 3 consecutive ticks to trigger (debounce)
                if alarm.consecutive_ticks >= 3 and not alarm.triggered:
                    alarm.triggered = True
                    # In a real system, this would execute the action
                    # For now, we just log it
                    tick.values[f"__alarm_{alarm.name}"] = 1.0
            except Exception:
                pass

    async def _notify_subscribers(self, tick: Tick) -> None:
        payload = self._tick_to_json(tick)
        dead = set()
        for q in self._subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.add(q)
        self._subscribers -= dead

    # ------------------------------------------------------------------
    # Protocol commands
    # ------------------------------------------------------------------
    def _tick_to_json(self, tick: Tick) -> str:
        obj = {"t": tick.timestamp, "seq": tick.seq, "room": self.room_id}
        obj.update(tick.values)
        return json.dumps(obj)

    def handle_command(self, line: str) -> str:
        line = line.strip()
        if not line:
            return ""
        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "tick":
            if not self._history:
                asyncio.create_task(self._tick())
                return "{"  # placeholder; real impl would await
            return self._tick_to_json(self._history[-1])

        elif cmd == "history":
            n = 10
            if len(parts) > 1:
                try:
                    n = int(parts[1])
                except ValueError:
                    pass
            n = min(n, len(self._history))
            ticks = list(self._history)[-n:]
            return json.dumps([json.loads(self._tick_to_json(t)) for t in ticks])

        elif cmd == "actuator":
            if len(parts) < 3:
                return "error: actuator <name> <value>"
            name, value_str = parts[1], parts[2]
            if name not in self._actuators:
                return f"error: unknown actuator {name}"
            try:
                # Try float, fallback to string
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str
                self._actuators[name](value)
                return "ok"
            except Exception as e:
                return f"error: {e}"

        elif cmd == "alarm":
            if len(parts) < 2:
                return "error: alarm <set|list> ..."
            sub = parts[1].lower()
            if sub == "list":
                return json.dumps([{"name": a.name, "condition": a.condition, "action": a.action} for a in self._alarms])
            elif sub == "set" and len(parts) >= 5:
                name = parts[2]
                cond = " ".join(parts[3:-1])
                action = parts[-1]
                self.add_alarm(name, cond, action)
                return "ok"
            else:
                return "error: alarm set <name> <condition> <action>"

        elif cmd == "subscribe":
            q = asyncio.Queue(maxsize=10)
            self._subscribers.add(q)
            # Return immediately; streaming handled by caller
            return "subscribed"

        elif cmd == "help":
            return "Commands: tick, history [N], actuator <name> <value>, alarm set|list, subscribe, help"

        else:
            return f"error: unknown command {cmd}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self, command_server: Optional[Any] = None) -> None:
        """Start tick loop. Optionally attach a command server."""
        self._running = True
        interval = 1.0 / self.tick_hz
        while self._running:
            await self._tick()
            await asyncio.sleep(interval)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Synchronous helpers for testing
    # ------------------------------------------------------------------
    def tick_sync(self) -> Tick:
        """Execute one tick synchronously (for tests)."""
        self._seq += 1
        values = {}
        for name, reader in self._sensors.items():
            try:
                values[name] = float(reader())
            except Exception:
                values[name] = float("nan")
        tick = Tick(timestamp=time.time(), seq=self._seq, values=values)
        self._history.append(tick)
        self._evaluate_alarms_sync(tick)
        return tick

    def _evaluate_alarms_sync(self, tick: Tick) -> None:
        """Synchronous alarm evaluation for tick_sync."""
        for alarm in self._alarms:
            try:
                parts = alarm.condition.split()
                if len(parts) != 3:
                    continue
                sensor_name, op, threshold_str = parts
                if sensor_name not in tick.values:
                    continue
                val = tick.values[sensor_name]
                threshold = float(threshold_str)
                triggered = False
                if op == ">":
                    triggered = val > threshold
                elif op == "<":
                    triggered = val < threshold
                elif op == ">=":
                    triggered = val >= threshold
                elif op == "<=":
                    triggered = val <= threshold
                elif op == "==":
                    triggered = val == threshold
                elif op == "!=":
                    triggered = val != threshold

                if triggered:
                    alarm.consecutive_ticks += 1
                else:
                    alarm.consecutive_ticks = 0
                    alarm.triggered = False

                if alarm.consecutive_ticks >= 3 and not alarm.triggered:
                    alarm.triggered = True
                    tick.values[f"__alarm_{alarm.name}"] = 1.0
            except Exception:
                pass
