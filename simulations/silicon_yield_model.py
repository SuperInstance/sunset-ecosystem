"""Simulation A2: Silicon Yield Prediction (Murphy model).

Implements the manufacturing yield model for a 28nm HPM ASIC (25mm^2 die)
from Research Package v3, SIMULATION_REQUIREMENTS_MASTER.md:

    Gross dies per wafer  — De Vries (2005) approximation
    Yield                — Murphy model with defect clustering
    Cost per good die    — wafer cost / good dies per wafer

Reference scenario (A2):
    300mm wafer, 25mm^2 die, defect density 0.5 defects/cm^2,
    70% yield target, cost per die target < $15 at 10K volume.
"""

from __future__ import annotations

import math

WAFER_COST_USD = 5_500.0  # approximate 300mm 28nm HPM wafer cost
YIELD_TARGET = 0.70
COST_TARGET_USD = 15.0


def dies_per_wafer(diameter_mm: float = 300.0, die_area_mm2: float = 25.0) -> int:
    """Gross dies per wafer using the De Vries (2005) approximation.

    DPW = pi*d^2/(4*A) - pi*d/sqrt(2*A)

    Accounts for wasted dies at the wafer's circular edge.
    """
    if diameter_mm <= 0:
        raise ValueError("diameter_mm must be positive")
    if die_area_mm2 <= 0:
        raise ValueError("die_area_mm2 must be positive")
    area_term = math.pi * diameter_mm**2 / (4.0 * die_area_mm2)
    edge_term = math.pi * diameter_mm / math.sqrt(2.0 * die_area_mm2)
    return max(0, int(math.floor(area_term - edge_term)))


def murphy_yield(defect_density_per_cm2: float = 0.5, die_area_mm2: float = 25.0) -> float:
    """Yield from the Murphy model with defect clustering.

    Y = [(1 - e^(-D*A)) / (D*A)]^2

    D = defect density (defects/cm^2), A = die area (cm^2).
    """
    if defect_density_per_cm2 < 0:
        raise ValueError("defect_density_per_cm2 must be non-negative")
    if die_area_mm2 <= 0:
        raise ValueError("die_area_mm2 must be positive")
    area_cm2 = die_area_mm2 / 100.0
    da = defect_density_per_cm2 * area_cm2
    if da == 0.0:
        return 1.0
    return ((1.0 - math.exp(-da)) / da) ** 2


def good_dies_per_wafer(
    diameter_mm: float = 300.0,
    die_area_mm2: float = 25.0,
    defect_density_per_cm2: float = 0.5,
) -> int:
    """Gross dies per wafer scaled by Murphy yield (known-good dies)."""
    gross = dies_per_wafer(diameter_mm, die_area_mm2)
    return int(math.floor(gross * murphy_yield(defect_density_per_cm2, die_area_mm2)))


def cost_per_die(
    wafer_cost_usd: float = WAFER_COST_USD,
    good_dies: int | None = None,
    diameter_mm: float = 300.0,
    die_area_mm2: float = 25.0,
    defect_density_per_cm2: float = 0.5,
) -> float:
    """Cost per known-good die (wafer cost amortized over good dies)."""
    if wafer_cost_usd < 0:
        raise ValueError("wafer_cost_usd must be non-negative")
    if good_dies is None:
        good_dies = good_dies_per_wafer(
            diameter_mm, die_area_mm2, defect_density_per_cm2
        )
    if good_dies <= 0:
        raise ValueError("good_dies must be positive")
    return wafer_cost_usd / good_dies


def simulate(
    wafer_diameter_mm: float = 300.0,
    die_area_mm2: float = 25.0,
    defect_density_per_cm2: float = 0.5,
    wafer_cost_usd: float = WAFER_COST_USD,
) -> dict:
    """Run the full A2 scenario and return a result summary dict."""
    gross = dies_per_wafer(wafer_diameter_mm, die_area_mm2)
    yield_frac = murphy_yield(defect_density_per_cm2, die_area_mm2)
    good = good_dies_per_wafer(
        wafer_diameter_mm, die_area_mm2, defect_density_per_cm2
    )
    cpd = cost_per_die(
        wafer_cost_usd=wafer_cost_usd,
        good_dies=good,
    )
    return {
        "wafer_diameter_mm": wafer_diameter_mm,
        "die_area_mm2": die_area_mm2,
        "defect_density_per_cm2": defect_density_per_cm2,
        "wafer_cost_usd": wafer_cost_usd,
        "gross_dies_per_wafer": gross,
        "yield_fraction": yield_frac,
        "yield_percent": yield_frac * 100.0,
        "good_dies_per_wafer": good,
        "cost_per_die_usd": cpd,
        "yield_target_met": yield_frac >= YIELD_TARGET,
        "cost_target_met": cpd <= COST_TARGET_USD,
    }


def main() -> None:
    r = simulate()
    print("=== Simulation A2: Silicon Yield Prediction (Murphy model) ===")
    print(f"Wafer diameter            : {r['wafer_diameter_mm']:.0f} mm")
    print(f"Die area                  : {r['die_area_mm2']:.1f} mm^2")
    print(f"Defect density            : {r['defect_density_per_cm2']:.2f} defects/cm^2")
    print(f"Wafer cost                : ${r['wafer_cost_usd']:,.0f}")
    print("-" * 60)
    print(f"Gross dies per wafer      : {r['gross_dies_per_wafer']}")
    print(f"Yield (Murphy model)      : {r['yield_percent']:.1f}%")
    print(f"Good dies per wafer       : {r['good_dies_per_wafer']}")
    print(f"Cost per good die         : ${r['cost_per_die_usd']:.2f}")
    print("-" * 60)
    print(f"Yield target (>70%)       : {'MET' if r['yield_target_met'] else 'NOT MET'} ({r['yield_fraction'] * 100:.1f}%)")
    print(f"Cost target (<$15/die)    : {'MET' if r['cost_target_met'] else 'NOT MET'} (${r['cost_per_die_usd']:.2f})")


if __name__ == "__main__":
    main()
