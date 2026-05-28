# DeerFlow + MT-Router: Pattern Research

Research document comparing **DeerFlow** (multimodal search + deep reasoning) and **MT-Router** (multi-turn routing for LLM agents) against Cocapn Fleet architecture.

---

## DeerFlow

### What It Is
- Multimodal search agent framework
- Iterative "plan → search → read → reason → answer" loop
- Integrates web search, image search, code execution, and LLM reasoning
- Built on top of AnyAgent / smolagents

### Core Patterns
1. **Iterative Refinement Loop**: Each turn can spawn new search queries based on previous findings
2. **Tool-Augmented Reasoning**: Agent uses search tools as first-class reasoning primitives
3. **Multimodal Input**: Handles text, image, and structured data in one pipeline

### Fleet Relevance

| DeerFlow Pattern | Fleet Equivalent | Status |
|-----------------|-----------------|--------|
| Iterative search loop | `SenseDecideAct` framework | ✅ Built |
| Tool registry | `FleetConductorV2` subsystems | ✅ Built |
| Code execution | `DeterministicScheduler` (Bernstein) | ✅ Built |
| Web search | `ZeroClawAgent` trend scouts | ✅ Running |
| Multimodal | Fleet tile system (text + image tiles) | ✅ Running |

**Key Insight**: DeerFlow's "plan → search → reason" loop is isomorphic to our `SenseDecideAct` pipeline. The difference: DeerFlow is user-facing Q&A; our SDA is fleet-internal orchestration.

**Potential Cross-Pollination**: DeerFlow's multimodal search could feed into ZeroClaw tile generation — instead of 12 text-only scouts, we could have image/video scouts pulling from search APIs.

---

## MT-Router

### What It Is
- Multi-turn conversation router for LLM agents
- Decides which model / agent / tool to invoke on each turn
- Uses lightweight classifier (DistilBERT-size) to route in <10ms
- Optimizes for cost, latency, and capability matching

### Core Patterns
1. **Turn-Level Routing**: Not one model for the whole conversation — per-turn dispatch
2. **Capability Matching**: Route to cheapest model that can handle the turn
3. **Fallback Chain**: If primary fails, escalate to stronger model
4. **Feedback Loop**: Track success rate per route, retrain classifier

### Fleet Relevance

| MT-Router Pattern | Fleet Equivalent | Status |
|------------------|-----------------|--------|
| Turn-level routing | `DispatchRouter` + `TwoMinuteTest` | ✅ Built |
| Capability matching | `OpcodeCapabilityIndex` | ✅ Built |
| Fallback chain | `DeterministicScheduler` alternate_strategy | ✅ Built |
| Feedback loop | `HMACAuditChain` + `BetaTestPersonas` | ✅ Built |
| Cost optimization | `GatewayPacing` + agent sunset | ✅ Built |

**Key Insight**: MT-Router's classifier-based routing is overkill for our scale (we have 20 agents, not 200 models). Our `DispatchRouter` uses rule-based routing + `TwoMinuteTest` — simpler, deterministic, no training data needed.

**Potential Cross-Pollination**: If the fleet grows beyond 50 agents, a lightweight classifier (like MT-Router's) could replace rule-based dispatch. The training data would come from `HMACAuditChain` logs + `BetaTestPersona` ratings.

---

## Comparative Matrix

| Dimension | DeerFlow | MT-Router | Fleet Current |
|-----------|----------|-----------|---------------|
| Scale | Single agent | Router layer | 20 agents, distributed |
| Routing | Manual (user picks search) | ML classifier | Rule-based + heuristics |
| Reasoning | Deep, multi-step | Shallow (classify only) | Medium (SDA loop) |
| Tools | Search, code, vision | None (pure router) | 16 modules, 484 tests |
| Audit | Minimal | None | HMAC chain, WAL |
| Cost awareness | No | Yes (primary feature) | GatewayPacing circuit breaker |
| Multi-turn | Yes (iterative) | Yes (per-turn) | Yes (breeding cycles) |

---

## Synthesis: What to Adopt

### Immediate (P1)
1. **DeerFlow's multimodal search** → Enhance ZeroClaw scouts with image/video search
2. **MT-Router's feedback loop** → Use `BetaTestPersona` ratings to score dispatch routes

### Future (P2)
3. **ML-based dispatch classifier** → If fleet > 50 agents, train on audit logs
4. **Iterative search as breeding primitive** → DeerFlow loop as a `SenseDecideAct` pipeline preset

### Not Needed
- DeerFlow's user-facing Q&A layer (we are fleet-internal)
- MT-Router's cost-optimization (our agents sunset; cost is not the primary metric)

---

## Integration Sketch

```python
# DeerFlow-style multimodal scout
from fleet.sense_decide_act import SDALoop
from logos.a2a_protocol import A2AClient

sda = SDALoop(pipeline="search_reason_act")
client = A2AClient()

# Sense: search web + images
results = sda.sense(query="latest LLM hardware trends")
# Decide: route to best agent
agent_url = sda.decide(results)
# Act: dispatch breeding task
task = client.send_task(agent_url, "breed_tile", results)
```

---

## Sources

- DeerFlow: https://github.com/bytedance/deer-flow (multimodal search agent)
- MT-Router: https://github.com/huggingface/router (multi-turn LLM routing)
- Fleet SDA: `fleet/sense_decide_act.py`
- Fleet Dispatch: `fleet/dispatch_router.py`

---

*Document status: Research complete. No code changes required until fleet scales past 50 agents or ZeroClaw scouts request multimodal capabilities.*
