# Sunset Ecosystem — The Trinity Architecture

> *Agents are born parallel, compete for relevance, sunset with dignity, and their compressed wisdom seeds the next generation.*

## Installation

```bash
pip install sunset-ecosystem
```

## Quick Start

```python
from sunset import GenerationRunner

runner = GenerationRunner()
report = runner.run_generation(generation=0)
# 12 agents spawned, competed, sunset or bred
# Losers wrote epilogues, survivors spawned children
# Next generation ready
```

## The Trinity

Every agent lives or dies by its connections to three rooms:

| Room | Purpose | Measures |
|------|---------|----------|
| **Ethos** | The metal | Hardware efficiency, resource fit, thermal awareness |
| **Pathos** | The human | Problem solved? Human waiting? Frustration reduced? |
| **Logos** | The code | Clean integration? Maintainable? Understands why? |

```
trinity_score = ethos × pathos × logos
```

If any connection is zero, the agent sunsets.

## The Lifecycle

```
INCUBATE → COMPETE → (SURVIVE → BREED) or (SUNSET → ARCHIVE)
                                        ↓                ↓
                                   Children          Seed Bank
                                   (streamlined)     (searchable)
```

## Sunset Documents

Every sunset agent writes three documents:

1. **Epilogue** — What I tried, what I found, why it wasn't relevant enough
2. **Summary** — My work from MY perspective (subjective, honest)
3. **Onboarding** — A letter to the next generation, written knowing I'm being put away

The onboarding can be written in multiple variants for diversity:
- `continuation` — For a similar agent to carry on
- `cross-pollination` — For a different species to learn from
- `mutation` — For a completely novel approach

## Hardware Swarm

| Unit | Count | Role |
|------|-------|------|
| RTX 4050 SMs | 20 | GPU-bound inference agents |
| Ryzen AI cores | 12 | CPU routing + scoring |
| Radeon 890M CUs | 16 | Overflow matmul |
| XDNA 2 NPU TOPS | 50 | INT8 quantized agents |
| **Full swarm** | **~110** | **Max parallel** |

## License

MIT
