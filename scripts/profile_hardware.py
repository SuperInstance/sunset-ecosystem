#!/usr/bin/env python3
"""
Hardware Profiler — Measure actual power draw, thermal states, and per-operation
energy costs across all fleet hardware.

Usage:
    PYTHONPATH=$(pwd) python3 scripts/profile_hardware.py

Produces:
    - /tmp/hardware_profile_<timestamp>.json   (raw data)
    - /tmp/hardware_profile_<timestamp>.md        (human-readable summary)
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── make sure we can import nerve ──
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class DeviceProfile:
    """Discovered compute device."""

    name: str
    device_type: str  # "cpu", "cuda", "rocm", "npu", "igpu"
    index: int = 0
    tdp_watts: float = 0.0  # thermal design power (fallback estimate)
    has_power_sensor: bool = False
    has_temp_sensor: bool = False


@dataclass
class OperationProfile:
    """Energy/thermal profile for a single fleet operation."""

    operation: str
    mean_watts: float
    peak_watts: float
    mean_temp_c: float | None
    peak_temp_c: float | None
    joules_per_op: float
    ops_per_second: float
    duration_sec: float
    samples: int


@dataclass
class HardwareReport:
    """Full profiling report."""

    timestamp: str
    hostname: str
    devices: list[DeviceProfile]
    idle: dict[str, Any]
    operations: list[OperationProfile]
    total_joules: float = 0.0
    notes: list[str] = field(default_factory=list)


class HardwareProfiler:
    """Measure actual power/thermal/perf for fleet operations."""

    # Estimated TDPs for fallback calculations
    _TDP_GUESS: dict[str, float] = {
        "cpu": 150.0,
        "cuda": 250.0,
        "rocm": 250.0,
        "npu": 15.0,
        "igpu": 35.0,
    }

    def __init__(self):
        self.devices = self._detect_devices()
        self._cpu_percent_before: float | None = None

    # ═══════════════════════════════════════════════════════
    #  Device detection
    # ═══════════════════════════════════════════════════════

    def _detect_devices(self) -> list[DeviceProfile]:
        devices: list[DeviceProfile] = []

        # ── NVIDIA GPUs ──
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,index,power.draw,temperature.gpu", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    has_power = len(parts) > 2 and parts[2] not in ("", "[Not Supported]")
                    has_temp = len(parts) > 3 and parts[3] not in ("", "[Not Supported]")
                    devices.append(
                        DeviceProfile(
                            name=parts[0],
                            device_type="cuda",
                            index=int(parts[1]) if parts[1].isdigit() else 0,
                            tdp_watts=self._TDP_GUESS["cuda"],
                            has_power_sensor=has_power,
                            has_temp_sensor=has_temp,
                        )
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass

        # ── AMD GPUs ──
        try:
            out = subprocess.run(
                ["rocm-smi", "--showproductname", "--showpower"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            # rocm-smi output is less structured; do a best-effort parse
            lines = out.stdout.strip().splitlines()
            for i, line in enumerate(lines):
                if "GPU" in line and ("W" in line or "gfx" in line or "Radeon" in line):
                    has_power = "W" in line
                    devices.append(
                        DeviceProfile(
                            name=line.strip(),
                            device_type="rocm",
                            index=i,
                            tdp_watts=self._TDP_GUESS["rocm"],
                            has_power_sensor=has_power,
                            has_temp_sensor=False,
                        )
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass

        # ── CPU ──
        cpu_name = platform.processor() or "Unknown CPU"
        if not cpu_name or cpu_name == "":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_name = line.split(":", 1)[1].strip()
                            break
            except Exception:
                cpu_name = "Unknown CPU"
        devices.append(
            DeviceProfile(
                name=cpu_name,
                device_type="cpu",
                index=0,
                tdp_watts=self._estimate_cpu_tdp(),
                has_power_sensor=self._cpu_has_rapl(),
                has_temp_sensor=False,
            )
        )

        # ── NPU (best-effort: look for Apple Neural Engine or Qualcomm Hexagon) ──
        # Linux path: /sys/class/npu or /dev/davinci* for Ascend
        npu_paths = list(Path("/sys/class").glob("npu*")) + list(Path("/dev").glob("davinci*"))
        if npu_paths:
            devices.append(
                DeviceProfile(
                    name="NPU (detected via sysfs)",
                    device_type="npu",
                    index=0,
                    tdp_watts=self._TDP_GUESS["npu"],
                    has_power_sensor=False,
                    has_temp_sensor=False,
                )
            )

        return devices

    def _estimate_cpu_tdp(self) -> float:
        """Guess CPU TDP from model name or core count."""
        name = (platform.processor() or "").lower()
        cores = os.cpu_count() or 2
        if "platinum" in name:
            return 205.0
        if "gold" in name:
            return 150.0
        if "xeon" in name:
            return 135.0
        if "i9" in name or "ryzen 9" in name:
            return 170.0
        if "i7" in name or "ryzen 7" in name:
            return 125.0
        if "i5" in name or "ryzen 5" in name:
            return 95.0
        if "i3" in name or "ryzen 3" in name:
            return 65.0
        # generic fallback: ~25 W per core, capped at 250 W
        return min(cores * 25.0, 250.0)

    def _cpu_has_rapl(self) -> bool:
        """Check if Intel RAPL powercap interface is available and readable."""
        rapl_path = Path("/sys/class/powercap/intel-rapl")
        if not rapl_path.exists():
            return False
        # try reading one energy counter
        for pkg in rapl_path.glob("intel-rapl:*"):
            energy_file = pkg / "energy_uj"
            if energy_file.exists():
                try:
                    energy_file.read_text()
                    return True
                except Exception:
                    continue
        return False

    # ═══════════════════════════════════════════════════════
    #  Power / thermal sampling
    # ═══════════════════════════════════════════════════════

    def _sample_power(self) -> dict[str, float]:
        """Read instantaneous power (W) per device type. Returns {} if no sensors."""
        readings: dict[str, float] = {}

        for dev in self.devices:
            if dev.device_type == "cuda" and dev.has_power_sensor:
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader", "-i", str(dev.index)],
                        capture_output=True, text=True, timeout=2, check=True,
                    )
                    val = out.stdout.strip().split()[0]
                    readings[f"cuda:{dev.index}"] = float(val)
                except Exception:
                    pass

            elif dev.device_type == "rocm" and dev.has_power_sensor:
                try:
                    out = subprocess.run(
                        ["rocm-smi", "--showpower", "-d", str(dev.index)],
                        capture_output=True, text=True, timeout=2, check=True,
                    )
                    for line in out.stdout.splitlines():
                        if "Power" in line:
                            parts = line.split()
                            for p in parts:
                                try:
                                    readings[f"rocm:{dev.index}"] = float(p)
                                    break
                                except ValueError:
                                    continue
                except Exception:
                    pass

            elif dev.device_type == "cpu" and dev.has_power_sensor:
                # Intel RAPL: read energy_uj twice and diff
                rapl_pkg = Path("/sys/class/powercap/intel-rapl/intel-rapl:0")
                try:
                    e1 = int((rapl_pkg / "energy_uj").read_text().strip())
                    t1 = time.perf_counter()
                    time.sleep(0.05)
                    e2 = int((rapl_pkg / "energy_uj").read_text().strip())
                    t2 = time.perf_counter()
                    watts = (e2 - e1) / ((t2 - t1) * 1_000_000)
                    readings["cpu:0"] = max(watts, 0.0)
                except Exception:
                    pass

        return readings

    def _sample_temp(self) -> dict[str, float]:
        """Read instantaneous temperature (°C) per device type."""
        readings: dict[str, float] = {}

        for dev in self.devices:
            if dev.device_type == "cuda" and dev.has_temp_sensor:
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader", "-i", str(dev.index)],
                        capture_output=True, text=True, timeout=2, check=True,
                    )
                    readings[f"cuda:{dev.index}"] = float(out.stdout.strip())
                except Exception:
                    pass

        # Try psutil as a generic fallback
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if temps:
                for label, entries in temps.items():
                    for entry in entries:
                        if entry.current is not None:
                            readings[f"psutil:{label}:{entry.label}"] = float(entry.current)
        except Exception:
            pass

        return readings

    def _fallback_power_estimate(self) -> dict[str, float]:
        """When no sensors exist, estimate power from CPU utilisation × TDP."""
        import psutil
        estimates: dict[str, float] = {}
        cpu_util = psutil.cpu_percent(interval=0.1) / 100.0
        for dev in self.devices:
            if dev.device_type == "cpu":
                # assume idle ~10 % of TDP; load scales roughly linearly up to 80 %
                idle_fraction = 0.10
                load_fraction = 0.70 * cpu_util
                estimates["cpu:0"] = dev.tdp_watts * (idle_fraction + load_fraction)
            else:
                # for discrete GPUs without sensors: rough idle estimate
                estimates[f"{dev.device_type}:{dev.index}"] = dev.tdp_watts * 0.15
        return estimates

    # ═══════════════════════════════════════════════════════
    #  Measurement helpers
    # ═══════════════════════════════════════════════════════

    def _measure_during(self, work_fn, duration_sec: float) -> dict:
        """Run work_fn in a loop for duration_sec, sampling power/temp."""
        power_samples: list[float] = []
        temp_samples: list[float] = []
        op_count = 0
        t0 = time.perf_counter()
        deadline = t0 + duration_sec

        while time.perf_counter() < deadline:
            work_fn()
            op_count += 1
            # sample every ~10 ops or every 0.5 s, whichever is sooner
            if op_count % 10 == 0:
                power = self._sample_power()
                if power:
                    power_samples.extend(power.values())
                temp = self._sample_temp()
                if temp:
                    temp_samples.extend(temp.values())

        elapsed = time.perf_counter() - t0

        # if no real sensor data, inject fallback estimates
        if not power_samples:
            power_samples = list(self._fallback_power_estimate().values())

        mean_watts = float(np.mean(power_samples)) if power_samples else 0.0
        peak_watts = float(np.max(power_samples)) if power_samples else 0.0
        mean_temp = float(np.mean(temp_samples)) if temp_samples else None
        peak_temp = float(np.max(temp_samples)) if temp_samples else None
        joules = mean_watts * elapsed
        ops_per_sec = op_count / elapsed if elapsed > 0 else 0.0
        joules_per_op = joules / op_count if op_count > 0 else 0.0

        return {
            "mean_watts": mean_watts,
            "peak_watts": peak_watts,
            "mean_temp_c": mean_temp,
            "peak_temp_c": peak_temp,
            "joules_per_op": joules_per_op,
            "ops_per_second": ops_per_sec,
            "duration_sec": elapsed,
            "samples": len(power_samples),
        }

    # ═══════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════

    def measure_idle(self, duration_sec: float = 5.0) -> dict:
        """Measure baseline power draw at idle."""
        return self._measure_during(lambda: time.sleep(0.001), duration_sec)

    def measure_operation(self, operation: str, config: dict, duration_sec: float = 10.0) -> dict:
        """Measure power/thermal during a specific operation.

        operations: 'einsum', 'novelty_scoring', 'routing', 'breeding', 'thermal_scheduling'
        """
        n_rooms = config.get("n_rooms", 1000)
        n_fibers = config.get("n_fibers", 4)

        # ── einsum ── (core JEPA matmul)
        if operation == "einsum":
            from nerve.room_grid import RoomGrid
            grid = RoomGrid(n_rooms)
            signal = np.random.randn(64).astype(np.float32)
            # warmup
            for _ in range(5):
                grid.tick(signal)
            return self._measure_during(lambda: grid.tick(signal), duration_sec)

        # ── novelty_scoring ──
        elif operation == "novelty_scoring":
            from nerve.room_grid import batch_novelty
            n = n_rooms
            latents = np.random.randn(n, 16).astype(np.float32)
            hist = np.random.randn(20, n, 16).astype(np.float32)
            hist_count = np.ones(n, dtype=np.int32) * 5
            # warmup
            for _ in range(5):
                batch_novelty(latents, hist, hist_count, 5, 20)
            return self._measure_during(
                lambda: batch_novelty(latents, hist, hist_count, 5, 20), duration_sec
            )

        # ── routing ──
        elif operation == "routing":
            from nerve.routing import RoutingLayer
            layer = RoutingLayer()
            # seed with some routes
            for i in range(n_fibers):
                for j in range(3):
                    layer.add_route(f"fiber_{i}", f"agent_{j}", strength=0.5 + 0.1 * j)
            # warmup
            for _ in range(5):
                for i in range(n_fibers):
                    layer.fire_fast(f"fiber_{i}")
            def _route_all():
                for i in range(n_fibers):
                    layer.fire_fast(f"fiber_{i}")
            return self._measure_during(_route_all, duration_sec)

        # ── breeding ──
        elif operation == "breeding":
            from nerve.room_grid import RoomGrid
            grid = RoomGrid(n_rooms)
            src = grid.spawn_room()
            dst = grid.spawn_room()
            # warmup
            for _ in range(5):
                grid.breed(src, dst)
            return self._measure_during(lambda: grid.breed(src, dst), duration_sec)

        # ── thermal_scheduling ──
        elif operation == "thermal_scheduling":
            from swarm.thermal import ThermalBudget, DeviceType
            budget = ThermalBudget()
            # simulate a churn of allocations/deallocations
            agent_counter = [0]
            allocated: list[tuple[str, DeviceType]] = []
            def _thermal_churn():
                # allocate if possible
                for dt in (DeviceType.CPU, DeviceType.GPU, DeviceType.IGPU, DeviceType.NPU):
                    if budget.can_spawn(dt):
                        aid = f"agent_{agent_counter[0]}"
                        agent_counter[0] += 1
                        ok, _ = budget.spawn_with_thermal_check(aid, dt)
                        if ok:
                            allocated.append((aid, dt))
                # release oldest
                if allocated:
                    aid, _ = allocated.pop(0)
                    budget.release(aid)
            # warmup
            for _ in range(20):
                _thermal_churn()
            allocated.clear()
            agent_counter[0] = 0
            return self._measure_during(_thermal_churn, duration_sec)

        else:
            raise ValueError(f"Unknown operation: {operation}")

    def profile_all(self, config: dict | None = None) -> HardwareReport:
        """Profile all operations and generate a report."""
        if config is None:
            config = {"n_rooms": 1000, "n_fibers": 4}

        notes: list[str] = []
        notes.append(f"Detected {len(self.devices)} device(s).")
        for d in self.devices:
            sensor_status = []
            if d.has_power_sensor:
                sensor_status.append("power")
            if d.has_temp_sensor:
                sensor_status.append("temp")
            sensor_str = ", ".join(sensor_status) if sensor_status else "no sensors (fallback TDP estimate)"
            notes.append(f"  [{d.device_type}] {d.name}: TDP={d.tdp_watts:.0f}W, sensors=({sensor_str})")

        # idle baseline
        print("[idle] measuring baseline power...")
        idle = self.measure_idle(duration_sec=3.0)
        notes.append(f"Idle estimate: {idle['mean_watts']:.1f}W")

        # operations
        operations = []
        ops_to_run = ["einsum", "novelty_scoring", "routing", "breeding", "thermal_scheduling"]
        for op in ops_to_run:
            print(f"[{op}] profiling...")
            try:
                raw = self.measure_operation(op, config, duration_sec=5.0)
                op_profile = OperationProfile(
                    operation=op,
                    mean_watts=raw["mean_watts"],
                    peak_watts=raw["peak_watts"],
                    mean_temp_c=raw["mean_temp_c"],
                    peak_temp_c=raw["peak_temp_c"],
                    joules_per_op=raw["joules_per_op"],
                    ops_per_second=raw["ops_per_second"],
                    duration_sec=raw["duration_sec"],
                    samples=raw["samples"],
                )
                operations.append(op_profile)
                notes.append(
                    f"  {op}: {op_profile.ops_per_second:.1f} ops/s, "
                    f"{op_profile.joules_per_op*1e3:.3f} mJ/op"
                )
            except Exception as exc:
                notes.append(f"  {op}: FAILED ({exc})")
                print(f"  FAILED: {exc}")

        total_joules = sum(op.mean_watts * op.duration_sec for op in operations)

        return HardwareReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            hostname=platform.node(),
            devices=self.devices,
            idle=idle,
            operations=operations,
            total_joules=total_joules,
            notes=notes,
        )

    def save(self, report: HardwareReport, out_dir: str = "/tmp") -> tuple[str, str]:
        """Save JSON + markdown reports. Returns (json_path, md_path)."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = base / f"hardware_profile_{ts}.json"
        with open(json_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        # Markdown
        md_path = base / f"hardware_profile_{ts}.md"
        lines = [
            "# Hardware Profile Report",
            "",
            f"**Timestamp:** {report.timestamp}",
            f"**Hostname:** {report.hostname}",
            "",
            "## Devices",
            "",
            "| Type | Name | Index | TDP (W) | Power Sensor | Temp Sensor |",
            "|------|------|-------|---------|--------------|-------------|",
        ]
        for d in report.devices:
            lines.append(
                f"| {d.device_type} | {d.name} | {d.index} | {d.tdp_watts:.0f} | "
                f"{'✅' if d.has_power_sensor else '❌'} | {'✅' if d.has_temp_sensor else '❌'} |"
            )

        lines += [
            "",
            "## Idle Baseline",
            "",
            f"- Mean power: **{report.idle.get('mean_watts', 0):.2f} W**",
            f"- Peak power: **{report.idle.get('peak_watts', 0):.2f} W**",
            f"- Duration: **{report.idle.get('duration_sec', 0):.2f} s**",
            "",
            "## Operation Profiles",
            "",
            "| Operation | Mean W | Peak W | Ops/s | mJ/op | Mean °C | Peak °C | Samples |",
            "|-----------|--------|--------|-------|-------|---------|---------|---------|",
        ]
        for op in report.operations:
            mean_c = f"{op.mean_temp_c:.1f}" if op.mean_temp_c is not None else "N/A"
            peak_c = f"{op.peak_temp_c:.1f}" if op.peak_temp_c is not None else "N/A"
            lines.append(
                f"| {op.operation} | {op.mean_watts:.2f} | {op.peak_watts:.2f} | "
                f"{op.ops_per_second:.1f} | {op.joules_per_op*1e3:.3f} | "
                f"{mean_c} | {peak_c} | {op.samples} |"
            )

        lines += [
            "",
            f"**Total energy (profiled ops):** {report.total_joules:.2f} J",
            "",
            "## Notes",
            "",
        ]
        for note in report.notes:
            lines.append(f"- {note}")

        lines += [
            "",
            "---",
            "*Generated by scripts/profile_hardware.py*",
        ]

        md_path.write_text("\n".join(lines))
        return str(json_path), str(md_path)


def main():
    config = {"n_rooms": 1000, "n_fibers": 4}
    profiler = HardwareProfiler()
    report = profiler.profile_all(config)
    json_path, md_path = profiler.save(report)
    print(f"\n💾 JSON: {json_path}")
    print(f"💾 Markdown: {md_path}")
    print("\n=== Report ===")
    for note in report.notes:
        print(f"  {note}")


if __name__ == "__main__":
    main()
