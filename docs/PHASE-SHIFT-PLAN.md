# Phase Shift Plan — Components → Continuous Daemon

Three file changes to make the system actually run continuously:

## 1. nerve/room_grid.py — Add `breed(src, dst)`
Rebirth only resets a room to RANDOM weights. Tournament breeding requires copying winner weights from src into dst. That's the gene-transfer step that's missing.
```python
def breed(self, src_idx: int, dst_idx: int) -> None:
    """Rebirth(dst) with weights cloned from src instead of random."""
    for k in ("w1", "w2", "w3"):
        self.w[k][dst_idx] = self.w[k][src_idx].copy()
    # Mutate slightly
    rng = np.random.RandomState(dst_idx + 8888)
    for k in ("w1", "w2", "w3"):
        mutation = rng.randn(*self.w[k][dst_idx].shape).astype(np.float32) * 0.005
        self.w[k][dst_idx] += mutation
    self.activity[dst_idx] = 0
    self.chaos[dst_idx] = 0.3
    self.history[dst_idx] = []
```

## 2. swarm/swarm_runner.py — Add `run_forever()` loop
No continuous loop exists. This wires grid.tick() → cold() → tournament → breed → rebirth into while True.

## 3. swarm/thermal.py — Add `can_breed() -> bool`
`thermal_headroom()` returns float but nothing gates breeding. Need a boolean check (e.g. headroom < 0.8).

## 4. distill/distillation_signal.py — JEPA latent as ranking signal
The cosine similarity between a response's JEPA latent and the room's predicted latent IS the ranking score. This replaces or augments user ranking, giving the distillation loop a continuous online signal.

### Priority
1. breed() (enables gene transfer)
2. run_forever() (makes it run)
3. can_breed() (thermal gating)
4. JEPA-as-signal (latent-based distillation)
