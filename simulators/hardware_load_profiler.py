#!/usr/bin/env python3
"""
Hardware Load Profiler — 6 load levels × 3 schedulers
500 timesteps, per-device utilization tracking.
"""

import random
import json
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path

random.seed(42)


# ─── Device Specs ───────────────────────────────────────────────
@dataclass
class Device:
    name: str
    sm_count: int
    capacity_per_step: float
    thermal_budget: float
    power_idle: float
    power_max: float
    thermal: float = 0.0
    throttled: bool = False
    work_done: float = 0.0
    time_throttled: int = 0
    migrations_in: int = 0
    migrations_out: int = 0


DEVICES = [
    Device("RTX4050", 20, 80.0, 85.0, 15.0, 95.0),
    Device("RyzenAI", 12, 45.0, 80.0, 8.0, 55.0),
    Device("Radeon890M", 16, 60.0, 82.0, 10.0, 70.0),
    Device("XDNA2", 50, 120.0, 75.0, 5.0, 30.0),
]

DEVICE_NAME_MAP = {
    "RTX4050": "RTX",
    "RyzenAI": "RyzenAI",
    "Radeon890M": "Radeon",
    "XDNA2": "XDNA2",
}


# ─── Workload Types ─────────────────────────────────────────────
@dataclass
class Job:
    wid: int
    wtype: str
    remaining: float
    device_idx: int = -1
    migrated: bool = False
    flight_time: int = 0


WORKLOAD_PROFILES = {
    "jepa": {"mean_cu": 40, "burst": 1.8, "sigma": 0.2},
    "flux": {"mean_cu": 120, "burst": 1.0, "sigma": 0.05},
    "tournament": {"mean_cu": 80, "burst": 1.4, "sigma": 0.3},
}

# ─── Simulation Parameters ──────────────────────────────────────
STEPS = 500
MIGRATION_COST = 8.0
FLIGHT_STEPS = 2
COOLING_RATE = 2.0
HEATING_PER_CU = 0.11


# ─── Job Generator ──────────────────────────────────────────────
def generate_jobs(load_fraction: float) -> List[Job]:
    total_capacity = sum(d.capacity_per_step for d in DEVICES) * STEPS
    target_cu = total_capacity * load_fraction
    jobs = []
    wid = 0
    while sum(j.remaining for j in jobs) < target_cu:
        wtype = random.choice(list(WORKLOAD_PROFILES.keys()))
        prof = WORKLOAD_PROFILES[wtype]
        cu = prof["mean_cu"] * random.gauss(1.0, prof["sigma"]) * prof["burst"]
        cu = max(5, cu)
        jobs.append(Job(wid=wid, wtype=wtype, remaining=cu))
        wid += 1
    return jobs


# ─── Schedulers ─────────────────────────────────────────────────


def schedule_naive(devices: List[Device], queue: List[Job], step: int) -> List[Job]:
    """Round-robin. Ignores thermal."""
    idx = 0
    active = []
    for job in queue:
        for _ in range(len(devices)):
            d = devices[idx % len(devices)]
            if d.capacity_per_step >= 1.0:
                job.device_idx = idx % len(devices)
                active.append(job)
                idx += 1
                break
            idx += 1
    return active


def schedule_thermal(devices: List[Device], queue: List[Job], step: int) -> List[Job]:
    """Avoid devices above global 55% threshold."""
    THERMAL_THRESHOLD = 0.55
    active = []
    for job in queue:
        if job.flight_time > 0:
            job.flight_time -= 1
            continue

        if job.device_idx >= 0:
            d = devices[job.device_idx]
            if d.thermal >= d.thermal_budget * THERMAL_THRESHOLD:
                candidates = [
                    (i, devices[i].thermal)
                    for i in range(len(devices))
                    if i != job.device_idx
                    and devices[i].thermal
                    < devices[i].thermal_budget * THERMAL_THRESHOLD
                ]
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    new_idx = candidates[0][0]
                    job.device_idx = new_idx
                    job.remaining -= MIGRATION_COST
                    if job.remaining < 0:
                        job.remaining = 0
                    job.migrated = True
                    job.flight_time = FLIGHT_STEPS
                    devices[new_idx].migrations_in += 1
                    d.migrations_out += 1
                    continue
        else:
            eligible = [
                (i, devices[i].thermal)
                for i in range(len(devices))
                if devices[i].thermal < devices[i].thermal_budget * THERMAL_THRESHOLD
            ]
            if eligible:
                eligible.sort(key=lambda x: x[1])
                job.device_idx = eligible[0][0]
            else:
                job.device_idx = min(
                    range(len(devices)), key=lambda i: devices[i].thermal
                )
        active.append(job)
    return active


def schedule_adaptive(devices: List[Device], queue: List[Job], step: int) -> List[Job]:
    """Per-device threshold = thermal_budget × 0.5."""
    active = []
    for job in queue:
        if job.flight_time > 0:
            job.flight_time -= 1
            continue

        if job.device_idx >= 0:
            d = devices[job.device_idx]
            device_threshold = d.thermal_budget * 0.5
            if d.thermal >= device_threshold:
                candidates = [
                    (i, devices[i].thermal)
                    for i in range(len(devices))
                    if i != job.device_idx
                    and devices[i].thermal < devices[i].thermal_budget * 0.5
                ]
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    new_idx = candidates[0][0]
                    job.device_idx = new_idx
                    job.remaining -= MIGRATION_COST
                    if job.remaining < 0:
                        job.remaining = 0
                    job.migrated = True
                    job.flight_time = FLIGHT_STEPS
                    devices[new_idx].migrations_in += 1
                    d.migrations_out += 1
                    continue
        else:
            eligible = [
                (i, devices[i].thermal)
                for i in range(len(devices))
                if devices[i].thermal < devices[i].thermal_budget * 0.5
            ]
            if eligible:
                eligible.sort(key=lambda x: x[1])
                job.device_idx = eligible[0][0]
            else:
                job.device_idx = min(
                    range(len(devices)), key=lambda i: devices[i].thermal
                )
        active.append(job)
    return active


# ─── Simulation Engine ──────────────────────────────────────────


def simulate(devices, jobs, scheduler_fn, label):
    devs = [Device(**{k: v for k, v in d.__dict__.items()}) for d in devices]
    queue = [Job(**{k: v for k, v in j.__dict__.items()}) for j in jobs]
    total_cu_generated = sum(j.remaining for j in jobs)
    total_possible = [d.capacity_per_step * STEPS for d in devs]

    for step in range(STEPS):
        for d in devs:
            d.thermal = max(0.0, d.thermal - COOLING_RATE)
            d.throttled = False

        active = scheduler_fn(devs, queue, step)
        active_set = {j.wid for j in active}
        queue = [j for j in queue if j.wid not in active_set]

        for job in active:
            d = devs[job.device_idx]
            if d.thermal >= d.thermal_budget:
                d.throttled = True
                d.time_throttled += 1
                continue

            processed = min(d.capacity_per_step, job.remaining)
            job.remaining -= processed
            d.work_done += processed
            d.thermal += processed * HEATING_PER_CU
            if d.thermal >= d.thermal_budget:
                d.throttled = True
                d.time_throttled += 1

        for job in active:
            if job.remaining > 0.1:
                queue.append(job)

    total_cu_remaining = sum(j.remaining for j in queue)
    throughput = (
        1.0 - (total_cu_remaining / total_cu_generated) if total_cu_generated else 0
    )

    dev_metrics = {}
    for d in devs:
        idx = devs.index(d)
        dev_metrics[d.name] = {
            "work_done": d.work_done,
            "utilization": d.work_done / total_possible[idx]
            if total_possible[idx] > 0
            else 0,
            "throttle_steps": d.time_throttled,
            "migrations_in": d.migrations_in,
            "migrations_out": d.migrations_out,
        }

    return {
        "label": label,
        "throughput": throughput,
        "throttle_steps": sum(d.time_throttled for d in devs),
        "migrations": sum(d.migrations_in for d in devs),
        "devices": dev_metrics,
    }


# ─── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    loads = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    schedulers = [
        ("Naive", schedule_naive),
        ("Thermal-55%", schedule_thermal),
        ("Adaptive-50%", schedule_adaptive),
    ]

    results = []
    for load in loads:
        jobs = generate_jobs(load)
        for sched_name, sched_fn in schedulers:
            r = simulate(DEVICES, jobs, sched_fn, f"load={load:.0%} {sched_name}")
            results.append(r)

    # Compute utilization % for key devices
    def fmt_pct(val):
        return f"{val:.1%}"

    rows = []
    for r in results:
        load_str = r["label"].split()[0].replace("load=", "")
        sched = r["label"].split()[1]
        rows.append(
            {
                "load": load_str,
                "scheduler": sched,
                "throughput": r["throughput"],
                "throttling": r["throttle_steps"],
                "XDNA2%": r["devices"]["XDNA2"]["utilization"],
                "RyzenAI%": r["devices"]["RyzenAI"]["utilization"],
                "RTX%": r["devices"]["RTX4050"]["utilization"],
            }
        )

    # Build markdown report
    lines = []
    lines.append("# Hardware Load Profile Report\n")
    lines.append("**Generated by:** Hardware Load Profiler  ")
    lines.append("**Timesteps:** 500 per condition  ")
    lines.append("**Load levels:** 50%, 60%, 70%, 80%, 90%, 95%  ")
    lines.append(
        "**Schedulers:** Naive (round-robin), Thermal-55% (global threshold), Adaptive-50% (per-device threshold = thermal_budget × 0.5)\n"
    )

    lines.append("## Results Table\n")
    lines.append(
        "| Load | Scheduler | Throughput | Throttling | XDNA2% | RyzenAI% | RTX% |"
    )
    lines.append(
        "|------|-----------|------------|------------|--------|----------|------|"
    )
    for row in rows:
        lines.append(
            f"| {row['load']} | {row['scheduler']} | {fmt_pct(row['throughput'])} | "
            f"{row['throttling']} | {fmt_pct(row['XDNA2%'])} | {fmt_pct(row['RyzenAI%'])} | {fmt_pct(row['RTX%'])} |"
        )
    lines.append("")

    # Find breakdown load: where does adaptive lose advantage?
    lines.append("## Breakdown Load Analysis\n")
    adaptive_rows = [r for r in rows if r["scheduler"] == "Adaptive-50%"]
    thermal_rows = [r for r in rows if r["scheduler"] == "Thermal-55%"]
    naive_rows = [r for r in rows if r["scheduler"] == "Naive"]

    for ar, tr, nr in zip(adaptive_rows, thermal_rows, naive_rows):
        load = ar["load"]
        if ar["throughput"] < tr["throughput"] - 0.005:
            lines.append(
                f"- **{load}**: Adaptive-50% underperforms Thermal-55% by {fmt_pct(tr['throughput'] - ar['throughput'])}. "
                f"Adaptive throttles {ar['throttling']}× vs thermal {tr['throttling']}×. "
                f"Per-device threshold is too aggressive — migrations spike, flight time burns capacity."
            )
        elif ar["throughput"] > tr["throughput"] + 0.005:
            lines.append(
                f"- **{load}**: Adaptive-50% outperforms Thermal-55% by {fmt_pct(ar['throughput'] - tr['throughput'])}. "
                f"Tighter per-device budgets prevent cascade throttling."
            )
        else:
            lines.append(
                f"- **{load}**: Adaptive-50% and Thermal-55% are roughly equal ({fmt_pct(ar['throughput'])})."
            )

    # Find exact breakdown
    breakdown_load = None
    for ar, tr in zip(adaptive_rows, thermal_rows):
        if ar["throughput"] < tr["throughput"] - 0.005:
            breakdown_load = ar["load"]
            break

    if breakdown_load:
        lines.append(f"\n**Breakdown load level:** {breakdown_load}")
        lines.append(
            f"\nAdaptive-50% begins losing its edge at {breakdown_load}. "
            f"Below this, per-device granularity avoids the 'hot spot' cascade. "
            f"At/above this, the cost of early migrations (flight time + {MIGRATION_COST} CU penalty) "
            f"outweighs the thermal savings.\n"
        )
    else:
        lines.append(f"\n**Breakdown load level:** None observed in tested range.")
        lines.append(
            f"\nAdaptive-50% maintained parity or advantage across all tested loads. "
            f"The aggressive threshold did not backfire within 50%–95%.\n"
        )

    lines.append("## Recommended Scheduler per Load Range\n")
    # Naive baseline for comparison
    for i, (ar, tr, nr) in enumerate(zip(adaptive_rows, thermal_rows, naive_rows)):
        load = ar["load"]
        best = max([ar, tr, nr], key=lambda x: x["throughput"])
        if best["scheduler"] == "Adaptive-50%":
            lines.append(
                f"- **{load}**: **Adaptive-50%** — best throughput ({fmt_pct(best['throughput'])}), per-device granularity pays off."
            )
        elif best["scheduler"] == "Thermal-55%":
            lines.append(
                f"- **{load}**: **Thermal-55%** — best throughput ({fmt_pct(best['throughput'])}), global threshold is the sweet spot."
            )
        else:
            lines.append(
                f"- **{load}**: **Naive** — best throughput ({fmt_pct(best['throughput'])}). Thermal awareness is pure overhead at this load."
            )

    lines.append("\n---")
    lines.append(f"\n*Raw data written to `simulators/hw_load_profile.json`*")

    report_path = Path("HARDWARE-LOAD-PROFILE.md")
    report_path.write_text("\n".join(lines))

    # Save raw JSON
    with open("hw_load_profile.json", "w") as f:
        json.dump(
            {
                "rows": rows,
                "breakdown_load": breakdown_load,
                "params": {
                    "steps": STEPS,
                    "migration_cost": MIGRATION_COST,
                    "flight_steps": FLIGHT_STEPS,
                    "cooling_rate": COOLING_RATE,
                    "heating_per_cu": HEATING_PER_CU,
                },
            },
            f,
            indent=2,
        )

    print("Report written to:", report_path)
    print("JSON written to: simulators/hw_load_profile.json")
    print("\nPreview:")
    print("\n".join(lines[:20]))
