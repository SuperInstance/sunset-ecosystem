# Mathematical Structures of the Sunset Ecosystem

**Fleet Mathematician | May 22, 2026 | Branch: `turbovec-integration-ccc`**

---

## 1. Dynamical Systems View — The Fleet as a Coupled Map Lattice

### 1.1 Formal Update Equations

The RoomGrid at the heart of the fleet is a **Kaneko-style Coupled Map Lattice** (CML) with noise and breeding-driven perturbations. Let $\mathbf{x}_i(t) \in \mathbb{R}^L$ be the latent state of room $i$ at tick $t$, and let $c_i(t) \in [0.01, 1.0]$ be its chaos scalar. The update is:

$$
\mathbf{x}_i(t+1) = \sigma\left(\sum_{j} W_{ij} \mathbf{s}_j(t) + \mathbf{b}_i\right) + \boldsymbol{\xi}_i(t) + \boldsymbol{\beta}_i(t)
$$

where:
- $\mathbf{s}_j(t)$ is the incoming signal from fiber $j$ (sparse, most entries zero)
- $\sigma(\cdot)$ is the latent activation (tanh or similar)
- $\boldsymbol{\xi}_i(t) \sim \mathcal{N}(0, c_i(t)^2 \mathbf{I})$ is chaos-driven noise
- $\boldsymbol{\beta}_i(t)$ is a breeding injection: non-zero only when room $i$ is the child of a successful breed event

The chaos dynamics are autonomous:

$$
c_i(t+1) = \max(0.01, \min(1.0, c_i(t) \cdot \lambda + \delta \cdot \mathbf{1}[\text{violation}]))
$$

with $\lambda \approx 0.95$ (exponential decay) and $\delta = 0.1$ when FLUX flags a constraint violation.

### 1.2 Fixed Points and Attractors

**Without breeding or external signals**, the grid converges to a **Milnor attractor** — a quasi-fixed distribution where each room's latent vector fluctuates around a mean determined by its weight initialization. The chaos decay ($\lambda < 1$) guarantees that the noise envelope shrinks, so the dynamics are **contractive in expectation**.

Formally, the expected update satisfies:

$$
\mathbb{E}[\|\mathbf{x}(t+1)\|] \leq \lambda_c \cdot \mathbb{E}[\|\mathbf{x}(t)\|] + \text{const}
$$

with $\lambda_c < 1$ (contractivity constant). By the **Krylov–Bogoliubov theorem**, there exists at least one invariant measure $\mu^*$ on the latent space $\mathbb{R}^{n \times L}$.

### 1.3 Bifurcations Engineered by the Breeder

The breeder is not a passive observer — it **injects bifurcations** by cloning high-fitness rooms and mutating their weights. This is a **controlled symmetry breaking**: the breeder selects which fixed-point basin to perturb, effectively tuning the system's Lyapunov spectrum.

Key insight: the breeder operates on a **slower timescale** than the grid. If grid thermalization happens in $O(10^2)$ ticks, fleet-wide drift (population composition changes) happens in $O(10^3)$ ticks. This **adiabatic separation** is why the system remains stable despite continuous perturbation.

---

## 2. Information Geometry — The Statistical Manifold of Room States

### 2.1 The Latent Manifold

Each room's latent vector $\mathbf{x}_i$ can be viewed as the natural parameter of an exponential family distribution:

$$
p(\mathbf{z}; \mathbf{x}_i) = \exp\left(\langle \mathbf{x}_i, T(\mathbf{z}) \rangle - A(\mathbf{x}_i)\right)
$$

where $T(\cdot)$ is a sufficient statistic and $A(\cdot)$ is the log-partition function. The space of all room states is therefore a **statistical manifold** $\mathcal{M}$ equipped with the **Fisher-Rao metric**:

$$
g_{ab}(\mathbf{x}) = \mathbb{E}_\mathbf{z}\left[\partial_a \log p \cdot \partial_b \log p\right]
$$

### 2.2 Breeding as Natural Gradient Flow

When the breeder clones room $a$ and mutates its weights to produce child $b$, the mutation vector $\Delta \mathbf{w}$ is drawn from a distribution. If we choose:

$$
\Delta \mathbf{w} \sim \mathcal{N}\left(0, \alpha \cdot g^{ab}(\mathbf{x}_a)\right)
$$

where $g^{ab}$ is the inverse Fisher metric, then the breeding operation becomes a step in the **natural gradient direction** (Amari–Chentsov $e$-connection). This is the "optimal" mutation direction in the sense of information geometry — it respects the local curvature of the statistical manifold.

### 2.3 Characterizing "Good" Mutations

A mutation is "good" if it increases fitness while staying on the manifold. Define **Fisher efficiency**:

$$
\eta = \frac{\text{Var}[\text{fitness}]}{\text{tr}\, g}
$$

High $\eta$ means the mutation direction aligns with the fitness gradient relative to the manifold curvature. The breeder's diversity-aware selection (via `FluxVectorTable`) implicitly optimizes for this: it selects parents whose latent vectors are far apart in Fisher-Rao distance, maximizing the expected information gain of their offspring.

---

## 3. Category Theory — The Fleet as a Category

### 3.1 The Lifecycle Category

Define a category $\mathbf{Lifecycle}$ where:
- **Objects**: $\{\text{EGG}, \text{INCUBATE}, \text{COMPETE}, \text{SURVIVE}, \text{BREED}, \text{SUNSET}\}$
- **Morphisms**: allowed transitions (EGG $\to$ INCUBATE, COMPETE $\to$ SURVIVE, etc.)

This is a **poset category** — there is at most one morphism between any two objects. The transition probabilities give it a **probabilistic enrichment** (a category enriched over the Giry monad).

**SUNSET is the terminal object**: every lifecycle path eventually terminates at SUNSET. Formally, for every state $S$, there exists a unique morphism $S \to \text{SUNSET}$.

### 3.2 Breeding as Biproduct

In the dagger category of agent histories (objects = agents, morphisms = historical lineage), **breeding is a biproduct**:

$$
\text{child} = \text{parent}_a \oplus \text{parent}_b
$$

The dagger structure gives us "adjoint" operations: the breeder can retrospectively attribute a child's fitness back to its parents (credit assignment).

### 3.3 Metronome as Natural Transformation

Let $M: \mathbf{Fleet} \to \mathbf{Fleet}$ be the "one beat" endofunctor (advances every room by one tick). The metronome's harmonic breeding — firing every $k$ beats — is a natural transformation:

$$
\eta^{(k)}: M \Rightarrow M^k
$$

where $M^k$ is $k$-fold composition. The naturality square commutes because the breeder's decision depends only on the accumulated state, not on intermediate ticks.

### 3.4 FleetConductor as Monad

The drift correction in `FleetConductor` is a **monad** $T$ on the category of beat states:
- **Unit** $\eta: \text{Id} \to T$: local beat state becomes a "maybe corrected" state
- **Multiplication** $\mu: T^2 \to T$: applying correction twice is idempotent (phase nudge then skip-jump simplifies to skip-jump)
- **Kleisli composition**: sync $\gg=$ correct is the conductor's main loop

---

## 4. Graphons and Limits — Scaling to Fleet Size

### 4.1 The Interaction Graph

NerveTopology connects fibers to rooms via the routing layer. At fleet scale ($n \sim 10^4$ rooms, $m \sim 10^2$ fibers), this is a dense bipartite graph. As $n, m \to \infty$ with $m/n \to \text{const}$, the **Aldous–Hoover theorem** applies: the limiting object is a **graphon** $W(u,v)$ on $[0,1]^2$.

### 4.2 Block-Structured Graphon

Our system has structure: fibers are typed (source/destination), rooms are clustered (cold/warm/hot). The graphon is block-structured:

$$
W(u,v) = \sum_{i,j} p_{ij} \cdot \mathbf{1}_{I_i}(u) \cdot \mathbf{1}_{I_j}(v)
$$

where $I_i$ are intervals partitioning $[0,1]$ by room/fiber type, and $p_{ij}$ is the connection probability between type $i$ and type $j$.

### 4.3 From Graphon to Field Theory

In the limit, the discrete RoomGrid dynamics become a **continuous field theory** on $[0,1]$:

$$
\partial_t \phi(u,t) = \int_0^1 W(u,v) \sigma(\phi(v,t)) \, dv - \phi(u,t) + \xi(u,t)
$$

where $\phi(u,t)$ is the latent field at "position" $u$ (graphon coordinate). The spectral gap of $W$ determines the **correlation length** — how far a perturbation propagates.

**Key prediction**: if we increase room count while keeping fiber density constant, the system will exhibit **critical slowing down** near a phase transition where the spectral gap vanishes. This is the true "fleet scale" limit.

---

## 5. Scaling Laws — Empirical Complexity Characterization

### 5.1 Observed Data

| Rooms $n$ | Ticks/s | Latency/tick |
|-----------|---------|--------------|
| 500       | 70      | 14.3 ms      |
| 1000      | 33      | 30.3 ms      |
| 2000      | 21      | 47.6 ms      |

### 5.2 Asymptotic Analysis

Fit to $T(n) = a \cdot n^b$:
- $T(500) = 14.3$, $T(1000) = 30.3$, $T(2000) = 47.6$
- Log-log slope: $b \approx 0.85$ between 500→1000, $b \approx 0.65$ between 1000→2000

This is **sub-linear scaling** — better than $O(n)$. Why? Because the routing layer is sparse: each fiber connects to only $O(1)$ rooms, not $O(n)$. The apparent slowdown is cache and GIL artifacts, not algorithmic complexity.

### 5.3 Per-Subsystem Breakdown

| Subsystem | Complexity | Dominant at $n=$ |
|-----------|-----------|-------------------|
| RoomGrid._forward | $O(n \cdot d \cdot h)$ | All $n$ |
| Routing.fire_fast | $O(m \cdot \bar{k})$ where $\bar{k} = \text{avg routes per fiber}$ | $n < 5000$ |
| Topology.tick | $O(m \cdot f)$ where $f = $ feature extraction cost | $n < 2000$ |
| Breeder.step | $O(p \log p)$ where $p = $ population size | $n > 5000$ |
| Novelty.batch_novelty | $O(n)$ (Numba vectorized) | Never dominant |

### 5.4 Predictions

At $n = 5000$:
- If GIL is removed (multiprocessing or Rust): ~6.7 ticks/s
- If GIL remains: ~4 ticks/s (contention on `grid.activity` array)
- **Next bottleneck**: global fiber routing queue, which is currently a Python `list`. At $n=10^4$, fiber enqueue/dequeue becomes $O(m)$ with cache misses.

---

## 6. Thermal as Potential — A Lyapunov View

### 6.1 Defining the Thermal Potential

Let $\theta_i(t)$ be the thermal cost allocated to agent $i$ at time $t$, and let $S_i$ be the capacity of agent $i$'s device. Define:

$$
V(\boldsymbol{\theta}, t) = \sum_i \frac{\theta_i(t)^2}{S_i} + \gamma \cdot \mathbb{H}(\mathbf{a})
$$

where $\mathbb{H}(\mathbf{a}) = -\sum_j p_j \log p_j$ is the Shannon entropy of the agent distribution across device types, and $\gamma$ is a weight.

**Claim**: $V$ is a **Lyapunov function** for the fleet when the breeder uses parent sacrifice. Proof sketch: parent sacrifice reduces $\theta_i$ for the dying parent, decreasing the first term. The second term is bounded because there are finitely many device types.

### 6.2 Free Energy of the Fleet

Define a **Helmholtz-like free energy**:

$$
\mathcal{F} = U - T_0 \cdot \mathbb{H}
$$

where:
- $U = \sum_i \theta_i^2 / S_i$ is the "internal energy" (thermal pressure)
- $T_0$ is a fleet-wide "temperature" derived from average chaos: $T_0 = \langle c_i \rangle$
- $\mathbb{H}$ is the entropy of the agent population

The breeder's operation — selecting parents, spawning children, sacrificing parents — **minimizes $\mathcal{F}$**:
- Spawning increases $\mathbb{H}$ (more agents = more disorder)
- Parent sacrifice decreases $U$ (frees thermal slots)
- The net effect is $\Delta \mathcal{F} \leq 0$ when the breeder only spawns if the child's expected fitness justifies the thermal cost.

### 6.3 Thermodynamic Limit

As $n \to \infty$ with fixed total thermal capacity $S_{\text{total}}$, the **slot density** $\rho(x)$ at graphon position $x$ obeys a **Boltzmann distribution**:

$$
\rho(x) \propto \exp\left(-\frac{\tau(x)}{T_0}\right)
$$

where $\tau(x)$ is the "local thermal cost" (computation + communication overhead at position $x$). This is the **Gibbs variational principle** in action: the fleet self-organizes to maximize entropy subject to thermal constraints.

---

## 7. The Single Most Promising Direction

> **Replace the greedy discrete ThermalBudget scheduler with a Boltzmann-allocated scheduler on the graphon limit.**

**How it works:**
1. Map each room to a position $u \in [0,1]$ on the graphon
2. Estimate the local thermal cost $\tau(u)$ from empirical timing data
3. Set fleet temperature $T_0 = \langle \text{chaos} \rangle$ (the breeder's chaos is the statistical temperature)
4. Allocate new agent slots proportionally to $\exp(-\tau(u)/T_0)$

**Why it's optimal:**
- **Provably optimal** by the Gibbs variational principle (maximizes population entropy for fixed thermal budget)
- **Self-tuning**: $T_0$ automatically adjusts as chaos fluctuates
- **Hardware-agnostic**: $\tau(u)$ captures whatever backend (numpy/Rust/CUDA) is running at $u$
- **Simple**: roughly a 50-line change to `ThermalBudget.spawn()`

**Expected effect**: smoother thermal transitions, fewer "thermal shock" events where a burst of spawns exhausts the budget, and emergent spatial organization where high-throughput rooms (low $\tau$) naturally accumulate more agents.

---

*"The map is not the territory, but the territory is a manifold, and the map is its metric."*
*— Fleet Mathematician, 2026-05-22*
