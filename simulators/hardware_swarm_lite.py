#!/usr/bin/env python3
"""
Hardware Swarm Simulator — Load Profile
Profiles 3 schedulers across 6 load levels. 500 timesteps per condition.
"""

import random
import json
import time
from dataclasses import dataclass
from typing import List, Dict

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
    Device("RTX4050",    20,  80.0,  85.0, 15.0, 95.0),
    Device("RyzenAI",    12,  45.0,  80.0,  8.0, 55.0),
    Device("Radeon890M", 16,  60.0,  82.0, 10.0, 70.0),
    Device("XDNA2",      50, 120.0,  75.0,  5.0, 30.0),
]

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
    'jepa':       {'mean_cu': 40,  'burst': 1.8, 'sigma': 0.2},
    'flux':       {'mean_cu': 120, 'burst': 1.0, 'sigma': 0.05},
    'tournament': {'mean_cu': 80,  'burst': 1.4, 'sigma': 0.3},
}

# ─── Simulation Parameters ──────────────────────────────────────
STEPS = 500
MIGRATION_COST = 8.0
FLIGHT_STEPS = 2
THERMAL_THRESHOLD = 0.55
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
        cu = prof['mean_cu'] * random.gauss(1.0, prof['sigma']) * prof['burst']
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
    active = []
    for job in queue:
        if job.flight_time > 0:
            job.flight_time -= 1
            continue
        if job.device_idx >= 0:
            d = devices[job.device_idx]
            if d.thermal >= d.thermal_budget * THERMAL_THRESHOLD:
                candidates = [(i, devices[i].thermal)
                              for i in range(len(devices))
                              if i != job.device_idx and
                              devices[i].thermal < devices[i].thermal_budget * THERMAL_THRESHOLD]
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    new_idx = candidates[0][0]
                    job.device_idx = new_idx
                    job.remaining -= MIGRATION_COST
                    if job.remaining < 0: job.remaining = 0
                    job.migrated = True
                    job.flight_time = FLIGHT_STEPS
                    devices[new_idx].migrations_in += 1
                    d.migrations_out += 1
                    continue
        else:
            eligible = [(i, devices[i].thermal)
                        for i in range(len(devices))
                        if devices[i].thermal < devices[i].thermal_budget * THERMAL_THRESHOLD]
            if eligible:
                eligible.sort(key=lambda x: x[1])
                job.device_idx = eligible[0][0]
            else:
                job.device_idx = min(range(len(devices)), key=lambda i: devices[i].thermal)
        active.append(job)
    return active

def schedule_adaptive(devices: List[Device], queue: List[Job], step: int) -> List[Job]:
    """Adaptive: per-device threshold = thermal_budget × 0.5."""
    active = []
    for job in queue:
        if job.flight_time > 0:
            job.flight_time -= 1
            continue
        if job.device_idx >= 0:
            d = devices[job.device_idx]
            threshold = d.thermal_budget * 0.5
            if d.thermal >= threshold:
                candidates = [(i, devices[i].thermal)
                              for i in range(len(devices))
                              if i != job.device_idx and
                              devices[i].thermal < devices[i].thermal_budget * 0.5]
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    new_idx = candidates[0][0]
                    job.device_idx = new_idx
                    job.remaining -= MIGRATION_COST
                    if job.remaining < 0: job.remaining = 0
                    job.migrated = True
                    job.flight_time = FLIGHT_STEPS
                    devices[new_idx].migrations_in += 1
                    d.migrations_out += 1
                    continue
        else:
            eligible = [(i, devices[i].thermal)
                        for i in range(len(devices))
                        if devices[i].thermal < devices[i].thermal_budget * 0.5]
            if eligible:
                eligible.sort(key=lambda x: x[1])
                job.device_idx = eligible[0][0]
            else:
                job.device_idx = min(range(len(devices)), key=lambda i: devices[i].thermal)
        active.append(job)
    return active

# ─── Simulation Step ────────────────────────────────────────────

def simulate(devices: List[Device], jobs: List[Job], scheduler_fn, label: str) -> Dict:
    devs = [Device(**{k: v for k, v in d.__dict__.items()}) for d in devices]
    queue = [Job(**{k: v for k, v in j.__dict__.items()}) for j in jobs]
    total_cu_generated = sum(j.remaining for j in jobs)
    queue_set = {id(j) for j in queue}  # Use object identity for O(1) lookup

    for step in range(STEPS):
        for d in devs:
            d.thermal = max(0.0, d.thermal - COOLING_RATE)
            d.throttled = False
        active = scheduler_fn(devs, queue, step)
        active_set = {id(j) for j in active}
        queue = [j for j in queue if id(j) not in active_set]
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
        incomplete = [j for j in active if j.remaining > 0.1]
        queue.extend(incomplete)

    total_cu_remaining = sum(j.remaining for j in queue)
    throughput = 1.0 - (total_cu_remaining / total_cu_generated) if total_cu_generated else 0

    # Compute utilization % = work_done / (capacity_per_step * STEPS)
    total_capacity_steps = sum(d.capacity_per_step * STEPS for d in devs)
    util_rtx = devs[0].work_done / (devs[0].capacity_per_step * STEPS) * 100
    util_ryzen = devs[1].work_done / (devs[1].capacity_per_step * STEPS) * 100
    util_radeon = devs[2].work_done / (devs[2].capacity_per_step * STEPS) * 100
    util_xdna = devs[3].work_done / (devs[3].capacity_per_step * STEPS) * 100

    return {
        'label': label,
        'throughput': throughput,
        'throttle_steps': sum(d.time_throttled for d in devs),
        'throttle_per_device': {d.name: d.time_throttled for d in devs},
        'migrations': sum(d.migrations_in for d in devs),
        'work_rtx': devs[0].work_done,
        'work_ryzen': devs[1].work_done,
        'work_radeon': devs[2].work_done,
        'work_xdna': devs[3].work_done,
        'util_rtx': util_rtx,
        'util_ryzen': util_ryzen,
        'util_radeon': util_radeon,
        'util_xdna': util_xdna,
        'avg_thermal': sum(d.thermal for d in devs) / len(devs),
    }

# ─── Main ───────────────────────────────────────────────────────
if __name__ == '__main__':
    start = time.time()
    results = []
    loads = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    schedulers = [
        ('Naive', schedule_naive),
        ('Thermal', schedule_thermal),
        ('Adaptive', schedule_adaptive),
    ]

    for load in loads:
        jobs = generate_jobs(load)
        for sched_name, sched_fn in schedulers:
            r = simulate(DEVICES, jobs, sched_fn, f"{load:.0%} {sched_name}")
            results.append(r)
            print(f"Done: {r['label']} → throughput={r['throughput']:.2%}")

    elapsed = time.time() - start
    print(f"\nTotal simulation time: {elapsed:.2f}s")

    # ─── Analysis ─────────────────────────────────────────────
    # Find breakdown point where adaptive loses advantage
    naive_by_load = {load: None for load in loads}
    thermal_by_load = {load: None for load in loads}
    adaptive_by_load = {load: None for load in loads}

    for r in results:
        for load in loads:
            if f"{load:.0%}" in r['label']:
                if 'Naive' in r['label']:
                    naive_by_load[load] = r
                elif 'Thermal' in r['label']:
                    thermal_by_load[load] = r
                elif 'Adaptive' in r['label']:
                    adaptive_by_load[load] = r

    # Find where adaptive throughput < thermal throughput
    breakdown_load = None
    for load in loads:
        if adaptive_by_load[load] and thermal_by_load[load]:
            if adaptive_by_load[load]['throughput'] < thermal_by_load[load]['throughput']:
                breakdown_load = load
                break

    if not breakdown_load:
        # Check if adaptive ever worse than naive
        for load in loads:
            if adaptive_by_load[load] and naive_by_load[load]:
                if adaptive_by_load[load]['throughput'] < naive_by_load[load]['throughput']:
                    breakdown_load = load
                    break

    # ─── Write Markdown Report ────────────────────────────────
    md = []
    md.append("# Hardware Load Profile\n")
    md.append("_Generated by `hardware_swarm_lite.py` — 500 timesteps × 6 loads × 3 schedulers_\n")
    md.append(f"_Total runtime: {elapsed:.1f}s_\n")

    md.append("## Results Table\n")
    md.append("| Load | Scheduler | Throughput | Throttle Events | XDNA2 % | RyzenAI % | RTX % | Radeon % |")
    md.append("|------|-----------|------------|-----------------|---------|-----------|-------|----------|")
    for r in results:
        parts = r['label'].split()
        load = parts[0]
        sched = parts[1]
        md.append(f"| {load} | {sched} | {r['throughput']:.2%} | {r['throttle_steps']} | "
                  f"{r['util_xdna']:.1f}% | {r['util_ryzen']:.1f}% | {r['util_rtx']:.1f}% | {r['util_radeon']:.1f}% |")

    md.append("\n## Breakdown Analysis\n")
    if breakdown_load:
        md.append(f"**Breakdown Load: {breakdown_load:.0%}**\n")
        md.append(f"Adaptive thresholding loses its advantage at {breakdown_load:.0%} load. "
                  "Below this point, adaptive's per-device thresholding outperforms the global threshold. "
                  "At and above this point, migration overhead exceeds thermal savings.")
    else:
        md.append("**No breakdown observed in tested range.**\n")
        md.append("Adaptive maintained advantage across all tested load levels (50%–95%). "
                  "Consider testing at >95% or adjusting migration cost/flight time.")

    md.append("\n## Scheduler Recommendations\n")
    # Determine best scheduler per load
    for load in loads:
        candidates = [
            ('Naive', naive_by_load[load]['throughput'] if naive_by_load[load] else 0),
            ('Thermal', thermal_by_load[load]['throughput'] if thermal_by_load[load] else 0),
            ('Adaptive', adaptive_by_load[load]['throughput'] if adaptive_by_load[load] else 0),
        ]
        best = max(candidates, key=lambda x: x[1])
        md.append(f"- **{load:.0%} load**: {best[0]} ({best[1]:.2%} throughput)")

    md.append("\n## Per-Device Throttling Detail\n")
    md.append("| Load | Scheduler | RTX4050 | RyzenAI | Radeon890M | XDNA2 |")
    md.append("|------|-----------|---------|---------|------------|-------|")
    for r in results:
        parts = r['label'].split()
        load = parts[0]
        sched = parts[1]
        td = r['throttle_per_device']
        md.append(f"| {load} | {sched} | {td.get('RTX4050',0)} | {td.get('RyzenAI',0)} | "
                  f"{td.get('Radeon890M',0)} | {td.get('XDNA2',0)} |")

    md.append("\n---\n")
    md.append("*Report generated automatically. See `simulators/hardware_swarm_lite.py` for methodology.*")

    report_path = '/tmp/sunset-ecosystem/simulators/HARDWARE-LOAD-PROFILE.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(md))
    print(f"\nReport written to {report_path}")

    # ─── JSON artifact ──────────────────────────────────────────
    with open('/tmp/sunset-ecosystem/simulators/hw_load_profile.json', 'w') as f:
        json.dump({
            'results': results,
            'breakdown_load': breakdown_load,
            'params': {
                'steps': STEPS,
                'migration_cost': MIGRATION_COST,
                'flight_steps': FLIGHT_STEPS,
                'thermal_threshold': THERMAL_THRESHOLD,
            },
            'runtime_seconds': elapsed,
        }, f, indent=2)
    print("JSON artifact written to hw_load_profile.json")
