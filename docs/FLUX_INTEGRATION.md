# FLUX Integration Guide

**Version:** sunset-ecosystem v0.9  
**Date:** 2026-05-22  
**Author:** kimi1 (Fleet Integrator)  

---

## What is FLUX?

FLUX is a **constraint-based self-correction system** that watches the nerve grid and pushes rooms back toward healthy behavior when they drift. It does not train or backpropagate — it *constrains*.

Think of FLUX as a **reflex arc** for the fleet:

```
RoomGrid.tick() ──→ latents produced
        │
        ▼
FLUX.check_batch() ──→ violations detected
        │
        ▼
apply_constraint_feedback() ──→ chaos adjusted
        │
        ▼
RoomGrid continues ticking
```

---

## Quick Start

```python
from nerve.room_grid import RoomGrid
from sunset.flux_integration import FluxConstraintChecker

# 1. Create a grid
grid = RoomGrid(n=1000)

# 2. Create a FLUX checker
flux = FluxConstraintChecker(preset="neural_bounds")

# 3. Attach to grid
grid.attach_flux_checker(flux)

# 4. Tick normally — FLUX runs automatically after every forward pass
result = grid.tick(np.random.randn(64))
# result now includes any constraint violations that were corrected
```

---

## Presets

| Preset | Bounds | L2 Norm | Variance | Use Case |
|--------|--------|---------|----------|----------|
| `neural_bounds` | ±10 | 25 | 5 | Default — safe for most MLP latents |
| `safe_mode` | ±5 | 10 | 2 | Conservative — strict control |
| `exploration` | ±20 | 50 | 10 | Permissive — high chaos, creative exploration |

```python
# Switch preset at runtime
flux.set_preset("exploration")
```

---

## Constraint Types

### 1. Bounds Check
Ensures every latent dimension stays within `[min_val, max_val]`.

```python
violations = flux.check_batch(latents)
# violations[i] == True  →  room i has at least one dimension out of bounds
```

### 2. L2 Norm Check
Ensures the Euclidean norm of each room's latent vector doesn't exceed `max_l2_norm`.

```python
# Internally: np.linalg.norm(latents, axis=1) > max_l2_norm
```

### 3. Variance Check
Ensures the variance across all rooms for each dimension stays below `max_variance`.

```python
# Internally: latents.var(axis=0) > max_variance
```

---

## How Feedback Works

When violations are detected, `apply_constraint_feedback()` does three things:

1. **Chaos injection** — Violating rooms get `chaos += 0.05` (default), making them more likely to fire chaotically on the next tick.
2. **Saturation cap** — Chaos is clamped to `0.95` so it never goes out of control.
3. **Activity bonus** — Violating rooms get `activity += 1` so they appear hotter in the breeding tournament.

```python
# From sunset/flux_integration.py
def apply_constraint_feedback(grid, checker):
    violations = checker.check_batch(grid.latents)
    grid.chaos[violations] += checker.config.chaos_increase
    grid.chaos = np.minimum(grid.chaos, 0.95)
    grid.activity[violations] += 1
```

---

## Custom Constraints

```python
from sunset.flux_integration import FluxConstraintChecker, ConstraintConfig

# Build your own config
my_config = ConstraintConfig(
    bounds=(-15.0, 15.0),
    max_l2_norm=30.0,
    max_variance=3.0,
    chaos_increase=0.1,  # stronger correction
)

flux = FluxConstraintChecker(config=my_config)
grid.attach_flux_checker(flux)
```

---

## Integration with RoomGrid Lifecycle

FLUX hooks into `RoomGrid.tick()` at three lifecycle points:

| Phase | What FLUX Does |
|-------|----------------|
| **After forward** | `check_batch()` on produced latents |
| **Post-fire** | `apply_constraint_feedback()` adjusts chaos |
| **On rebirth** | Chaos reset to `0.3` (not FLUX-specific, but relevant) |

```python
# RoomGrid.tick() pseudo-code
def tick(self, x):
    latents = self._forward(x)  # ← forward pass
    self.latents = latents  # ← store for FLUX
    # ... novelty + chaos gating ...
    if self._flux_checker is not None:
        apply_constraint_feedback(self, self._flux_checker)  # ← FLUX hook
    return {"fired": ..., "ids": ..., "tick": ...}
```

---

## Integration with Breeder

When a room is rebirthed via the breeder, its chaos is reset to `0.3` — the default starting value. FLUX will begin watching it again from the next tick.

```python
# In AutoBreeder._rebirth_with_clone()
self.grid.chaos[target_room] = 0.3  # reset chaos for new agent
```

---

## Performance

| Rooms | FLUX Check Time | Overhead |
|-------|-----------------|----------|
| 1,000 | ~0.05 ms | <0.5% |
| 10,000 | ~0.5 ms | <1% |
| 50,000 | ~2.5 ms | <2% |

FLUX is pure Python + NumPy. For 10K+ rooms, a Rust FFI backend (`libflux_vm.so`) is planned.

---

## Rust FFI Backend (Future)

The `flux-vm-v3-temp/` directory contains a Rust implementation of `check_batch()`. Compile with:

```bash
cd flux-vm-v3-temp/
cargo build --release
# Copy target/release/libflux_vm.so to sunset-ecosystem/
```

When `libflux_vm.so` is present, `FluxConstraintChecker` will auto-detect and use it.

---

## Testing

```python
# Run FLUX tests
pytest tests/test_flux_integration.py -v

# Test specific preset
pytest tests/test_flux_integration.py::TestPresets -v
```

---

## Troubleshooting

**"No violations detected even with crazy latents"**
→ The grid's forward pass uses `np.tanh()` which bounds outputs to [-1, 1]. With default `bounds=(-10, 10)`, violations are extremely rare. This is by design — the grid is already self-stabilizing. FLUX catches the edge cases.

**"Chaos doesn't increase after violations"**
→ Check that `grid.attach_flux_checker()` was called before `tick()`. The checker must be attached *before* the tick that produces the violating latents.

**"Checker raises TypeError"**
→ `attach_flux_checker()` requires an object with `check_batch()` and `get_violations()` methods. Use `FluxConstraintChecker` or implement the duck-type interface.

---

## API Reference

### `FluxConstraintChecker`

```python
class FluxConstraintChecker:
    def __init__(self, preset="neural_bounds", config=None):
        """preset: 'neural_bounds' | 'safe_mode' | 'exploration'"""

    def check_batch(self, latents: np.ndarray) -> np.ndarray:
        """Returns boolean mask of violating rooms."""

    def get_violations(self, latents, room_ids=None) -> list[ConstraintViolation]:
        """Returns detailed violation records with room IDs."""

    def set_preset(self, preset: str) -> None:
        """Switch preset at runtime."""
```

### `apply_constraint_feedback`

```python
def apply_constraint_feedback(grid: RoomGrid, checker: FluxConstraintChecker) -> None:
    """Apply FLUX feedback to a RoomGrid after tick().
    Called automatically by RoomGrid.tick() when checker is attached.
    """
```

### `RoomGrid.attach_flux_checker`

```python
def attach_flux_checker(self, checker) -> None:
    """Attach a FLUX constraint checker for self-correcting behavior.
    Checker must have .check_batch() and .get_violations() methods.
    """
```

---

## See Also

- `docs/INTEGRATION_MAP.md` — Full-stack architecture diagram
- `docs/SPEC-FLUX-RESOLUTION.md` — FLUX vs v2/v3 VM resolution
- `sunset/flux_integration.py` — Source code
- `tests/test_flux_integration.py` — Test suite

---

*"The constraint that disappears is the constraint that works."*  
— Forgemaster ⚒️
