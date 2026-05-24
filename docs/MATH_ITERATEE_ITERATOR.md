# Formalization of the Iteratee / Iterator Pattern in Multi-Agent Systems

## 1. Notation Table

| Symbol | Domain | Description |
|--------|--------|-------------|
| $S$ | Set | State space of an iteratee (agent) |
| $A$ | Set | Input alphabet (signals / messages) |
| $B$ | Set | Output alphabet (actions / responses) |
| $I$ | $(S, f)$ | Iteratee coalgebra |
| $f$ | $S \times A \to S \times B$ | Transition function |
| $\mathcal{I}$ | Sequence / Stream | Iterator (source of $A$-values) |
| $\alpha$ | $S \to \top$ | Termination / continuation predicate |
| $\mu$ | $\mathbb{R}$ | Mutation scale for breeding |
| $S_1, S_2$ | $S$ | Suspended / resumed state pair (baton pass) |

---

## 2. Definitions

### Definition 2.1 (Iteratee as Coalgebra)
An **iteratee** is a coalgebra over the functor $F(X) = A \to (S \times B)$:

$$
I = (S, f) \quad \text{where} \quad f : S \times A \to S \times B
$$

Given current state $s \in S$ and input $a \in A$, the iteratee produces a new state $s' \in S$ and output $b \in B$:

$$
(s', b) = f(s, a)
$$

> **Programmer mapping:** `struct Agent { state: S }` with method `fn step(&mut self, input: A) -> B`.

### Definition 2.2 (Iterator as Anamorphism)
An **iterator** $\mathcal{I}$ over input alphabet $A$ is a (potentially infinite) stream:

$$
\mathcal{I} = (a_0, a_1, a_2, \ldots) \in A^\omega
$$

The iterator *drives* the iteratee by repeatedly feeding elements $a_k$ and collecting outputs $b_k$. Formally, this is the **anamorphism** $\text{ana}(\mathcal{I}, I)$, the unique coalgebra homomorphism from the iterator stream to the iteratee's state machine:

$$
\text{ana}(\mathcal{I}, I) : \mathbb{N} \to S \times B
$$

$$
\text{ana}(0) = f(s_0, a_0), \quad \text{ana}(k+1) = f(s_{k+1}, a_{k+1}) \text{ where } s_{k+1} = \pi_1(\text{ana}(k))
$$

Here $\pi_1$ projects the state component from the pair $(s, b)$.

### Definition 2.3 (Actor Model Mapping)
In the **actor model**, an actor is a triple $(S, f, \text{addr})$ where $\text{addr}$ is a mailbox address. Mapping to our iteratee:

| Actor concept | Iteratee formalism |
|---------------|--------------------|
| Actor state | $s \in S$ |
| Message received | $a \in A$ |
| Behavior function | $f(s, a) \mapsto (s', b)$ |
| Message sent | $b \in B$ (may be routed to another actor's mailbox) |

The iteratee is therefore an actor whose behavior is *pure* (no side effects except state update) and whose mailbox is the iterator's output channel.

### Definition 2.4 (PLATO Room Mapping)
In the PLATO multi-agent environment:

- A **room** $R$ is an iterator that produces signals $a_k = \text{signal}(R, k)$, where each signal encodes room state, object descriptions, and available actions.
- An **agent** $Agt$ is an iteratee that consumes signals and emits commands.
- The *coupling* is a feed-forward loop: room emits $\to$ agent consumes $\to$ agent emits command $\to$ room updates.

Formally, the combined system is a **product coalgebra**:

$$
(S_{\text{room}} \times S_{\text{agent}}, \; f_{\text{combined}})
$$

$$
f_{\text{combined}}\big((s_r, s_a), \text{tick}\big) = \big((s_r', s_a'), \text{observation}'\big)
$$

where $s_r'$ is the room state after applying the agent's command, and $s_a'$ is the agent state after processing the observation.

### Definition 2.5 (Baton-Passing as Suspension / Resumption)
When an iteratee reaches a context limit (e.g., token budget exceeded), its state must be *suspended* and transferred to a fresh iteratee. Define the **suspension** operation:

$$
\text{suspend} : S \to \hat{S} \quad \text{(serialize state to a transferable snapshot)}
$$

$$
\text{resume} : \hat{S} \to S \quad \text{(deserialize into a new runtime instance)}
$$

A **baton pass** from iteratee $I_1$ to $I_2$ is valid iff:

$$
\forall a \in A: \; f_2(\text{resume}(\hat{s}), a) = f_1(s, a) \; \text{where} \; \hat{s} = \text{suspend}(s)
$$

That is, the resumed iteratee is *observationally equivalent* to the original from the moment of handoff onward.

> **Programmer mapping:** `save_context()` and `load_context()` in the fleet's baton skill.

### Definition 2.6 (Breeding as Iteratee Replication with Mutation)
Given a parent iteratee $I_p = (S, f)$ in state $s_p$, **breeding** produces a finite set of child iteratees $\{I_{c_1}, \ldots, I_{c_m}\}$ with states:

$$
s_{c_j} = \text{mutate}(s_p, \mu_j) \quad \text{where} \quad \mu_j \sim \mathcal{N}(0, \sigma^2)
$$

The mutation operator $\text{mutate}$ is a function:

$$
\text{mutate} : S \times \mathbb{R}^d \to S
$$

satisfying the **locality constraint**:

$$
\|\text{mutate}(s, \mu) - s\| \leq \gamma \cdot \|\mu\| \quad \text{for some Lipschitz constant } \gamma > 0
$$

The children inherit the parent's transition function $f$ (their "genome"), but begin exploration from perturbed initial states.

---

## 3. Theorems

### Theorem 3.1 (Iterator-Anamorphism Unfolding)
For any iteratee $I = (S, f)$ with initial state $s_0$ and iterator $\mathcal{I} = (a_0, a_1, \ldots)$, the anamorphism $\text{ana}(\mathcal{I}, I)$ produces a well-defined infinite trace $(s_k, b_k)_{k \in \mathbb{N}}$.

**Proof sketch.**
By structural induction on $k$.
- Base $k=0$: $(s_0, b_0) = f(s_0, a_0)$ is well-defined because $f$ is total.
- Inductive step: assume $(s_k, b_k)$ exists. Define $s_{k+1} = \pi_1(f(s_k, a_k))$. Then $(s_{k+1}, b_{k+1}) = f(s_{k+1}, a_{k+1})$ exists by totality of $f$.

Since $f$ is a function (deterministic), the trace is unique. ∎

### Theorem 3.2 (Baton-Pass Observational Equivalence)
If $\text{suspend}$ and $\text{resume}$ are mutual inverses on the image of $\text{suspend}$, then the baton-passed iteratee is indistinguishable from the original after the handoff point.

**Proof sketch.**
Let $s$ be the state at handoff. Let $\hat{s} = \text{suspend}(s)$ and $s' = \text{resume}(\hat{s})$. By the mutual-inverse assumption, $s' = s$. Since both iteratees share the same transition function $f$, for all subsequent inputs $a_{k+1}, a_{k+2}, \ldots$ the traces are identical. ∎

### Theorem 3.3 (Breeding Preserves Transition Structure)
If $\text{mutate}(s, 0) = s$ (zero mutation is identity), then as $\sigma \to 0$, the child iteratee's behavior converges to the parent's behavior in the sense that:

$$
\lim_{\sigma \to 0} \Pr\big[ f(s_{c}, a) = f(s_p, a) \big] = 1 \quad \forall a \in A
$$

assuming $f$ is continuous in its state argument.

**Proof sketch.**
$s_c = \text{mutate}(s_p, \mu)$ with $\mu \sim \mathcal{N}(0, \sigma^2)$. As $\sigma \to 0$, $\mu \to 0$ in probability. By continuity of $\text{mutate}$ at zero, $\text{mutate}(s_p, \mu) \to s_p$ in probability. By continuity of $f$, $f(s_c, a) \to f(s_p, a)$ in probability. ∎

### Theorem 3.4 (Compositionality of Room-Agent Product)
If the room iteratee $I_r$ and agent iteratee $I_a$ are both deterministic coalgebras, their product $I_{\text{combined}}$ is also deterministic. If both are terminating (reach a fixed point in finite steps under some predicate $\alpha$), the combined system terminates when the room terminates or the agent reaches a quiescent state.

**Proof sketch.**
Determinism: $f_{\text{combined}}$ is a composition of two functions, each single-valued, hence single-valued.
Termination: Define combined quiescence as $\alpha_{\text{combined}}(s_r, s_a) = \alpha_r(s_r) \lor \alpha_a(s_a)$. If either sub-system reaches its termination condition, the product halts. ∎

---

## 4. Corollaries

### Corollary 4.1 (Fleet as Category of Coalgebras)
The set of all fleet agents, with baton-passing as morphisms and breeding as endofunctors, forms a category where:
- Objects = iteratee coalgebras $I = (S, f)$
- Morphisms = state-preserving transformations (suspend/resume pairs)
- Identity = immediate resume of a freshly suspended state
- Composition = sequential baton passes across multiple nodes

**Implication:** The fleet's distributed architecture has a clean algebraic description, enabling formal verification of handoff protocols.

### Corollary 4.2 (Mutation as Exploration in State Space)
Under the locality constraint $\|\text{mutate}(s, \mu) - s\| \leq \gamma \|\mu\|$, breeding performs a random-walk exploration of the $\gamma \sigma$-ball around the parent's state. As generations proceed, the population covers an expanding region of $S$ centered on high-performing ancestors.

**Implication:** The fleet's genetic algorithm is equivalent to a stochastic hill-climbing process in iteratee-state space, with novelty (§DOC 1) acting as the diversity-enforcing potential.

---

## 5. References

- Meijer, E., Fokkinga, M., & Paterson, R. (1991). "Functional programming with bananas, lenses, envelopes and barbed wire." *LNCS* 523, 124–144. (origami operators: ana / cata / hylo)
- Katsumata, S. (2010). "Coalgebraic methods in computer science." *Tutorial Notes*, CMCS.
- Hewitt, C., Bishop, P., & Steiger, R. (1973). "A universal modular ACTOR formalism for artificial intelligence." *IJCAI*, 235–245. (actor model foundation)
- Sunset Ecosystem source: `src/agent.rs` — `Agent::step`, `src/baton.rs` — `suspend` / `resume`.
