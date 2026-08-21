# 🔗 Integration Guide

How to integrate new modules into the Sunset Ecosystem fleet.

## Prerequisites

Before adding a new module, ensure:

1. **Tests**: Every module must have a comprehensive pytest suite
2. **Docstrings**: All public classes and functions must have docstrings
3. **Integration**: Identify at least one existing module to connect with
4. **Documentation**: Add a description of what the module does

## Step-by-Step Integration

### 1. Register the Module

```python
from fleet.harbor import Harbor

harbor = Harbor()
harbor.register_module("MyModule", "swarm/my_module.py", ["VectorSwarm"])
```

### 2. Add Integration Paths

```python
harbor.add_integration(
    "MyModule", "VectorSwarm", "MyModule uses VectorSwarm for search"
)
```

### 3. Verify Health

```python
health = harbor.check_fleet_health()
print(health["healthy"], "/", health["total"])
```

### 4. Run Tests

```bash
python3 -m pytest tests/test_my_module.py -v
```

## Existing Integration Patterns

| Source | Target | Pattern |
|--------|--------|---------|
| VectorSwarm | FleetMemory | tested |
| CognitiveCache | FleetMemory | tested |
| BreedOptimizer | VectorSwarm | mapped |
| BreedOptimizer | CognitiveCache | mapped |
| TMinusBridge | FleetMonitor | mapped |
| TernaryTypes | FleetMonitor | mapped |
| PatternMine | EcosystemHub | mapped |
| QuantaVDBBridge | FleetMemory | tested |
| CASLangExecutor | LevelRunner | tested |
| Pincher | QuantaVDBBridge | tested |
| xLangAgentBridge | LevelRunner | tested |
| EcosystemHub | PatternMine | tested |

## Dependency Order

Modules should be initialized in this order:

1. HNSWMeshTable
2. TieredMeshStorage
3. FleetMemory
4. MeshWAL
5. MeshGrouping
6. SceneTracker
7. VectorSwarm
8. CognitiveCache
9. FleetAPI
10. FleetMonitor
11. QuantaVDBBridge
12. CASLangExecutor
13. LevelRunner
14. Pincher
15. xLangAgentBridge
16. EcosystemHub
17. PatternMine
18. TMinusBridge
19. BreedOptimizer
20. TernaryTypes
