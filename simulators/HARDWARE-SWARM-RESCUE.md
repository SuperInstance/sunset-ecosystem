# Hardware Swarm — Rescue Report

## Simulation Setup

| Parameter | Value |
|-----------|-------|
| Time steps | 100 |
| Devices | 4 (RTX4050, RyzenAI, Radeon890M, XDNA2) |
| Workloads | 3 (jepa, flux, tournament) |
| Migration cost | 8 CU |
| Flight time | 2 steps |
| Thermal threshold | 55% of budget |
| Load levels tested | 30%, 60%, 90% |

## Results Table

| Scenario | Throughput | Throttle Events | Migrations | RTX Work | Ryzen Work | Radeon Work | XDNA Work |
|----------|-----------:|----------------:|-----------:|---------:|-----------:|------------:|----------:|
| 30% naive | 100.00% | 3,290 | 0 | 2,365 | 2,292 | 2,251 | 2,307 |
| 30% thermal | 100.00% | 1,133 | 460 | 1,680 | 1,286 | 1,313 | 1,308 |
| 60% naive | 56.05% | 11,824 | 0 | 2,631 | 2,533 | 2,605 | 2,491 |
| 60% thermal | 100.00% | 4,000 | 1,351 | 2,221 | 1,850 | 1,814 | 1,760 |
| 90% naive | 37.32% | 20,774 | 0 | 2,582 | 2,550 | 2,593 | 2,528 |
| 90% thermal | 100.00% | 7,769 | 2,432 | 2,284 | 2,116 | 2,203 | 1,759 |

## Counter-Intuitive Finding

> **The fastest device does the least work.**

Under thermal-aware scheduling at 90% load, **XDNA2** (120 CU/step capacity, 50 SMs — the highest in the fleet) completes **1,759 CU** of work, while the modest **RyzenAI** (45 CU/step, 12 SMs) completes **2,116 CU**.

This inversion happens because XDNA2 has the *lowest thermal budget* (75°C vs 85°C for RTX4050). The aggressive 55% threshold triggers migrations off XDNA2 before it can burn through its massive capacity. Jobs ping-pong onto it, heat it up, then flee — leaving it underutilized while cooler, slower devices pick up the slack.

**The scheduler's "protection" starves its best performer.**

## Scheduling Recommendation

**Adaptive thresholding by device class.** Instead of a single global threshold (55%), compute a per-device migration threshold based on `capacity / thermal_budget` ratio:

- High-capacity, low-budget devices (XDNA2) need a *higher* threshold (e.g., 70%) so they can do meaningful work before being abandoned.
- Low-capacity, high-budget devices (RyzenAI) can tolerate a *lower* threshold (e.g., 50%) since they don't generate as much heat per unit work.

A single threshold optimized for the "average" device will systematically mis-allocate work across heterogeneous hardware. The swarm is only as smart as its threshold logic.

---
*Rescued from previous agent timeout. Simulation ran in <2 seconds. All results reproducible with `python3 simulators/hardware_swarm_lite.py`.*
