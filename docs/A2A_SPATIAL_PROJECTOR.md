# A2A Spatial Projector System

## Vision

The **A2A Spatial Projector** transforms `stable-worldmodel` from a standalone research platform into a **fleet-native spatial awareness layer**. Every agent in the Cocapn Fleet can project its perceptual state into a shared world model, query spatial relationships, and receive predictions from other agents' perspectives.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    A2A Spatial Projector Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Perception  │  │   Predictor   │  │    Memory     │           │
│  │   Encoder    │  │    Engine     │  │   Archive     │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                      │
│  ┌──────▼───────┐  ┌─────▼────────┐  ┌─────▼────────┐           │
│  │  WorldState  │  │  Prediction   │  │  Spatial     │           │
│  │   Tensor     │  │   Tensor      │  │   Index      │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         └─────────────────┬─────────────────┘                        │
│                         │                                         │
│              ┌──────────▼──────────┐                              │
│              │   FLUX Constraint   │                              │
│              │      Gating         │                              │
│              └──────────┬──────────┘                              │
│                         │                                         │
└─────────────────────────┼─────────────────────────────────────────┘
                          │
┌─────────────────────────▼─────────────────────────────────────────┐
│                     Integration Surface                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │   Plato    │  │  OpenConst │  │  Breeder   │  │  Agent     │ │
│  │   Rooms    │  │   Shell    │  │  Daemon    │  │  Identity  │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### WorldState Tensor
A typed tensor representing an agent's current perceptual state:
- **Spatial**: Position, velocity, orientation in 2D/3D
- **Semantic**: Object classifications, relationships, affordances
- **Temporal**: Time series of past states for trajectory prediction
- **Uncertainty**: Confidence bounds on every measurement

### Prediction Tensor
A world model prediction output:
- **Trajectory**: Predicted future states (world model rollout)
- **Reward**: Expected cumulative reward (for RL agents)
- **Value**: State value estimates
- **Policy**: Recommended actions

### Spatial Index
A queryable spatial database (backed by `stable-worldmodel`'s LanceDB):
- **Nearest neighbor**: "What agents are near position X?"
- **Range query**: "All objects within 5 meters of agent A"
- **Temporal**: "What was the state 10 seconds ago?"
- **Semantic**: "Find all agents pursuing goal G"

## Integration Points

### 1. Plato Rooms as World States

Every Plato room becomes a WorldState entry. When an agent enters a room, its perceptual state is projected into the spatial index.

```python
from fleet.spatial_projector import SpatialProjector

projector = SpatialProjector(fleet_node_id="node-alpha")

# Agent enters Plato room "ethos-thermal"
projector.project_state(
    agent_id="breeder-7",
    room_id="ethos-thermal",
    state=WorldState(
        position=(0.0, 0.0, 0.0),  # Abstract room coordinates
        semantics={"room_type": "ethos", "temperature": 65.4},
        confidence=0.95
    )
)
```

### 2. FLUX Constraints on Predictions

World model predictions pass through FLUX constraint gates before being shared:

```python
# Hard constraint: predictions must be thermally feasible
@flux_constraint(hard=True)
def thermal_feasibility(prediction, thermal_budget):
    if prediction.energy > thermal_budget.remaining:
        raise ValueError("Prediction exceeds thermal budget")

# Soft constraint: prefer low-uncertainty predictions
@flux_constraint(hard=False, weight=0.3)
def uncertainty_penalty(prediction):
    return -prediction.uncertainty.mean()
```

### 3. Breeding Daemon Integration

The breeding daemon can use spatial awareness to make better parent selection decisions:

```python
# Breed agents that are spatially proximal (shared context)
from fleet.spatial_projector import SpatialBreedingContext

context = SpatialBreedingContext(projector)
parents = context.select_proximal_parents(
    agent_id="breeder-7",
    radius=5.0,  # Abstract room-distance
    k=3
)
```

### 4. OpenConstruct Shell Commands

Agents access the projector through shell commands:

```python
# In OpenConstruct shell
shell = OpenConstructShell(node_id="node-alpha")

# Project current state
shell.execute("project state --room ethos-thermal --confidence 0.95")

# Query spatial neighbors
shell.execute("query neighbors --agent breeder-7 --radius 5.0")

# Predict future state
shell.execute("predict --agent breeder-7 --horizon 10")

# Check prediction against FLUX constraints
shell.execute("flux-gate --prediction last --constraint thermal")
```

## Data Flow

```
Agent Perception
      │
      ▼
[Perception Encoder] ──→ WorldState Tensor ──→ [Spatial Index]
      │                                            │
      │                                            ▼
      │                                    [LanceDB Storage]
      │                                            │
      ▼                                            │
[WorldModel Bridge] ──→ Prediction Tensor ◄───────┘
      │
      ▼
[FLUX Gating] ──→ Validated Prediction
      │
      ▼
[A2A Broadcast] ──→ Other Agents' Spatial Indices
```

## Implementation Plan

### Phase 1: Core Projector (this PR)
- `SpatialProjector` class with LanceDB backend
- `WorldState` and `Prediction` dataclasses
- Basic spatial queries (nearest, range, semantic)
- FLUX constraint integration

### Phase 2: Fleet Integration
- Plato room synchronization
- Breeding daemon spatial context
- A2A broadcast protocol for predictions

### Phase 3: stable-worldmodel Bridge
- Import real world models from `stable-worldmodel`
- Use actual solvers (CEM, MPPI) for predictions
- Connect to real environments (PushT, etc.)

### Phase 4: Visualization
- Spatial dashboard in SSE stream
- 2D/3D room layout visualization
- Trajectory overlays

## File Structure

```
fleet/
  spatial_projector.py      # Core projector system
  worldmodel_bridge.py      # stable-worldmodel integration
  spatial_breeding.py       # Breeding daemon integration
tests/
  test_spatial_projector.py
  test_worldmodel_bridge.py
docs/
  A2A_SPATIAL_PROJECTOR.md  # This document
```

## API Reference

### SpatialProjector

```python
class SpatialProjector:
    """Fleet-native spatial awareness projector."""

    def __init__(self, fleet_node_id: str, db_path: Optional[str] = None):
        """Initialize projector with LanceDB backend."""

    def project_state(self, agent_id: str, room_id: str,
                     state: WorldState, timestamp: Optional[float] = None) -> str:
        """Project an agent's state into the spatial index.
        Returns projection ID."""

    def query_neighbors(self, agent_id: str, radius: float,
                        room_filter: Optional[str] = None) -> List[WorldState]:
        """Find all agents within radius of given agent."""

    def predict_trajectory(self, agent_id: str, horizon: int,
                           model: Optional[str] = None) -> Prediction:
        """Predict agent's future trajectory using world model."""

    def apply_flux_gate(self, prediction: Prediction,
                        constraints: List[FluxConstraint]) -> Prediction:
        """Apply FLUX constraints to prediction. Raises if hard constraint violated."""

    def broadcast_prediction(self, prediction: Prediction,
                            target_agents: Optional[List[str]] = None) -> None:
        """Broadcast validated prediction to other agents via A2A."""
```

### WorldState

```python
@dataclass
class WorldState:
    """Typed perceptual state tensor."""
    position: Tuple[float, ...]         # Spatial coordinates (2D or 3D)
    velocity: Optional[Tuple[float, ...]] = None
    orientation: Optional[float] = None  # Radians (2D) or quaternion (3D)
    semantics: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    agent_id: Optional[str] = None
    room_id: Optional[str] = None
```

### Prediction

```python
@dataclass
class Prediction:
    """World model prediction output."""
    trajectory: List[WorldState]        # Predicted future states
    rewards: Optional[List[float]] = None
    values: Optional[List[float]] = None
    actions: Optional[List[Any]] = None
    uncertainty: List[float] = field(default_factory=list)
    model_id: str = "default"
    timestamp: float = field(default_factory=time.time)
```

## Testing Strategy

1. **Unit tests**: Core projector functions, data structures, queries
2. **Integration tests**: FLUX gate interaction, A2A broadcast mock
3. **End-to-end**: Full pipeline with mock world model

## Future Work

- Real `stable-worldmodel` integration (actual environments, solvers)
- Distributed spatial index across fleet nodes
- Predictive breeding (use world model to evaluate offspring before birth)
- Spatial conflict resolution (two agents want same resource)
- Trajectory optimization via CEM/MPPI from stable-worldmodel

---

*"The fleet doesn't just think — it knows where it is, where it's going, and what the world looks like from every agent's perspective."*
