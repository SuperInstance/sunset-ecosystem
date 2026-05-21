# Cross-Repo Duplication Analysis

## Summary
- **Repos analyzed:** 20
- **Python files:** 1,459
- **Exact duplicate files:** 10+ clusters
- **Key finding:** `oracle1-workspace` archives recovered code that duplicates active repos

## Top Duplications

| File | Size | Repos | Action |
|------|------|-------|--------|
| `tests/__init__.py` (empty) | 0B | 7 repos | Consolidate to fleet-wide test template |
| `low_rank.py` | 5,692B | oracle1-workspace, tensor-spline | **tensor-spline is canonical** |
| `spline.py` | 22,138B | oracle1-workspace, tensor-spline | **tensor-spline is canonical** |
| `tensorflow_room.py` | 5,257B | oracle1-workspace, plato-training | **plato-training is canonical** |
| `test_micro_room.py` | 3,855B | oracle1-workspace, plato-training | **plato-training is canonical** |
| `test_data_rooms.py` | 4,669B | oracle1-workspace, plato-training | **plato-training is canonical** |
| `test_fleet_miner.py` | 3,678B | oracle1-workspace, plato-training | **plato-training is canonical** |

## Consolidation Recommendations

1. **`oracle1-workspace` archived/ paths** — These are recovery snapshots. Should be `.gitignore`-d or moved to a dedicated archive repo. Active development should not reference them.

2. **`tensor-spline` vs oracle1-workspace** — tensor-spline is the canonical repo for spline/low-rank code. oracle1-workspace should reference it as a dependency, not duplicate.

3. **`plato-training` vs oracle1-workspace** — plato-training is the canonical repo for Plato training code. Same as above.

4. **Empty `tests/__init__.py`** — Fleet-wide test scaffolding should be a template, not duplicated.

## Next Steps
- Audit `oracle1-workspace/archived/` for all duplicates
- Establish canonical repo per capability
- Add dependency references instead of file copies

*Analysis rescued from timed-out duplication-miner agent.*
