# Fleet Weather Report

**Phase 5.4** — Automated daily fleet health summaries.

## Overview

`FleetWeatherReport` ingests fleet statistics (from `FleetConductorV2` or direct `FleetStats`) and generates a concise, data-dense markdown report in institutional-research style.

## Sections

| Section | Description |
|---------|-------------|
| **Header** | Date, fleet name, node count |
| **Breeding Summary** | Attempted, FLUX passes/fails, thermal throttled, success rate |
| **Node Health** | Per-node drift, beat sync status, thermal tier |
| **Diversity Trend** | Current score + delta vs yesterday (if history exists) |
| **Notable Events** | Errors, anomalies, high-diversity discoveries |
| **Forecast** | Breed success rate trend vs last week |

## Usage

```python
from nexus.fleet_conductor_v2 import FleetConductorV2
from fleet.fleet_weather_report import FleetWeatherReport

conductor = FleetConductorV2(config)
report = FleetWeatherReport.from_conductor(conductor)

# Markdown output
print(report.to_markdown())

# Write to file
report.to_file("reports/fleet_weather_2026-05-29.md")

# Post to Matrix (optional)
report.post_to_matrix(
    hook_url="https://matrix.example.com/webhook",
    channel="#fleet-ops",
)
```

## History & Trends

`~/.fleet_weather_history.json` stores daily snapshots (90-day window). Used to compute:
- Diversity delta vs yesterday
- Breed success rate trend vs last week

## Design

- **No fluff.** Information-first, every line earns its place.
- **Defensive.** Missing data renders as "N/A" rather than crashing.
- **Lightweight.** No external dependencies beyond stdlib.
