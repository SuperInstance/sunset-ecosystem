# Formalization of the FLUX Constraint Checker

## 1. Notation Table

| Symbol | Domain | Description |
|--------|--------|-------------|
| $v$ | $\mathbb{R}^l$ | Latent vector being validated (here $l = 16$) |
| $C$ | $\mathbb{R}^l \to \{0, 1\}$ | Constraint predicate; $C(v) = 1$ means "valid" |
| $\min, \max$ | $\mathbb{R}$ | Per-dimension bounds (applied elementwise) |
| $L_{\max}$ | $\mathbb{R}_{\geq 0}$ | Maximum allowed $L^2$ norm |
| $V_{\max}$ | $\mathbb{R}_{\geq 0}$ | Maximum allowed variance |
| $\mu$ | $\mathbb{R}$ | Mean of components: $\mu = \frac{1}{l}\sum_{j=1}^l v_j$ |
| $\varepsilon$ | $\mathbb{R}_{>0}$ | Floating-point tolerance (e.g., $10^{-6}$) |
| $\wedge$ | $\{0,1\} \times \{0,1\} \to \{0,1\}$ | Boolean AND |

---

## 2. Definitions

### Definition 2.1 (Constraint Predicate)
A **constraint predicate** is a total boolean function:

$$
C : \mathbb{R}^l \to \{0, 1\}
$$

where $C(v) = 1$ denotes that vector $v$ satisfies the constraint, and $C(v) = 0$ denotes violation. We identify $\{0, 1\}$ with $\{\text{false}, \text{true}\}$.

> **Programmer mapping:** `fn check(v: &[f32; 16]) -> bool` in Rust.

### Definition 2.2 (Bounds Constraint)
The **elementwise bounds constraint** $C_{\text{bounds}} : \mathbb{R}^l \to \{0, 1\}$ is:

$$
C_{\text{bounds}}(v) = \bigwedge_{j=1}^{l} \bigl( \min \leq v_j \leq \max \bigr)
$$

Equivalently, written as a product of indicator functions:

$$
C_{\text{bounds}}(v) = \prod_{j=1}^{l} \mathbf{1}_{[\min, \max]}(v_j)
$$

where $\mathbf{1}_{[a,b]}(x) = 1$ if $a \leq x \leq b$, and $0$ otherwise.

### Definition 2.3 ($L^2$ Norm Constraint)
The **L2 norm constraint** $C_{L^2} : \mathbb{R}^l \to \{0, 1\}$ is:

$$
C_{L^2}(v) = \begin{cases} 1 & \text{if } \|v\|_2 \leq L_{\max} \\ 0 & \text{otherwise} \end{cases}
$$

where the Euclidean norm is:

$$
\|v\|_2 = \left( \sum_{j=1}^{l} v_j^2 \right)^{1/2}
$$

### Definition 2.4 (Variance Constraint)
The **population variance constraint** $C_{\text{var}} : \mathbb{R}^l \to \{0, 1\}$ is:

$$
C_{\text{var}}(v) = \begin{cases} 1 & \text{if } \sigma^2(v) \leq V_{\max} \\ 0 & \text{otherwise} \end{cases}
$$

where the population variance is:

$$
\sigma^2(v) = \frac{1}{l} \sum_{j=1}^{l} (v_j - \mu)^2 \quad \text{with} \quad \mu = \frac{1}{l} \sum_{j=1}^{l} v_j
$$

> **Note:** We use population variance (denominator $l$), not sample variance ($l-1$), because the latent vector is treated as a complete population of $l$ activations, not a sample.

### Definition 2.5 (Total Constraint)
The **total constraint** is the logical conjunction of the three individual constraints:

$$
C_{\text{total}}(v) = C_{\text{bounds}}(v) \; \wedge \; C_{L^2}(v) \; \wedge \; C_{\text{var}}(v)
$$

> **Programmer mapping:** `flux_check_batch` returns `true` iff all three per-vector checks pass.

---

## 3. Theorems

### Theorem 3.1 (Monotonicity of Bounds Constraint)
Let $v, v' \in \mathbb{R}^l$ such that for every component $j$:

$$
|v'_j - m| \leq |v_j - m| \quad \text{where } m = \frac{\min + \max}{2}
$$

In words: $v'$ is elementwise closer to the interval center than $v$. Then:

$$
C_{\text{bounds}}(v) = 1 \implies C_{\text{bounds}}(v') = 1
$$

**Proof sketch.**
Suppose $C_{\text{bounds}}(v) = 1$, so $\min \leq v_j \leq \max$ for all $j$. The interval $[\min, \max]$ is convex and symmetric around $m$. If $v'_j$ is closer to $m$ than $v_j$, then $v'_j$ lies on the line segment between $v_j$ and its reflection across $m$, which is entirely contained in $[\min, \max]$. Therefore $\min \leq v'_j \leq \max$. Since this holds for all $j$, the conjunction is satisfied. ∎

### Theorem 3.2 (Monotonicity of $L^2$ Constraint under Shrinking)
If $\|v'\|_2 \leq \|v\|_2$ and $C_{L^2}(v) = 1$, then $C_{L^2}(v') = 1$.

**Proof sketch.**
$C_{L^2}(v) = 1$ means $\|v\|_2 \leq L_{\max}$. By assumption $\|v'\|_2 \leq \|v\|_2$. By transitivity of $\leq$, $\|v'\|_2 \leq L_{\max}$, hence $C_{L^2}(v') = 1$. ∎

### Theorem 3.3 (Monotonicity of Variance Constraint under Averaging)
If $v'$ is obtained from $v$ by replacing two components $v_i, v_j$ with their average $\frac{v_i + v_j}{2}$, then $\sigma^2(v') \leq \sigma^2(v)$. Consequently, variance-reducing projections preserve validity.

**Proof sketch.**
Variance is $\sigma^2 = \frac{1}{l}\sum (v_k - \mu)^2$. The mean $\mu$ is unchanged if two values are replaced by their average. The contribution of the pair changes from $(v_i-\mu)^2 + (v_j-\mu)^2$ to $2\bigl(\frac{v_i+v_j}{2}-\mu\bigr)^2$. By Jensen's inequality (or direct algebra):

$$
2\left(\frac{v_i+v_j}{2}-\mu\right)^2 = \frac{(v_i-\mu)^2 + 2(v_i-\mu)(v_j-\mu) + (v_j-\mu)^2}{2} \leq (v_i-\mu)^2 + (v_j-\mu)^2
$$

since $2ab \leq a^2 + b^2$. Therefore the sum of squared deviations does not increase, so $\sigma^2(v') \leq \sigma^2(v)$. ∎

### Theorem 3.4 (Composition — Total Constraint Satisfiability)
$C_{\text{total}}(v) = 1$ if and only if all three individual constraints are satisfied:

$$
C_{\text{total}}(v) = 1 \iff C_{\text{bounds}}(v) = 1 \; \wedge \; C_{L^2}(v) = 1 \; \wedge \; C_{\text{var}}(v) = 1
$$

Moreover, $C_{\text{total}}$ is satisfiable (i.e., $\exists v: C_{\text{total}}(v)=1$) if and only if each individual constraint is separately satisfiable and their feasible regions intersect non-trivially.

**Proof sketch.**
The equivalence follows directly from Definition 2.5 and the definition of logical AND.

For satisfiability:
- ($\Rightarrow$) If $C_{\text{total}}(v) = 1$, then by the equivalence each individual constraint evaluates to 1, so each is satisfiable.
- ($\Leftarrow$) Let $F_{\text{bounds}}$, $F_{L^2}$, $F_{\text{var}}$ be the feasible sets. If their intersection $F = F_{\text{bounds}} \cap F_{L^2} \cap F_{\text{var}} \neq \emptyset$, pick any $v \in F$. Then all three constraints hold, so $C_{\text{total}}(v) = 1$. ∎

### Theorem 3.5 (Rust `flux_check_batch` Implements Exact Predicates)
Assume `flux_check_batch` is implemented as the sequential conjunction of three exact floating-point checks:

1. `v[j] >= min && v[j] <= max` for all $j$
2. `sqrt(sum(v[j]*v[j])) <= L_max`
3. `variance(v) <= V_max`

where `variance(v)` computes $\frac{1}{l}\sum (v_j - \mu)^2$ with $\mu = \frac{1}{l}\sum v_j$.

Then, ignoring floating-point round-off below $\varepsilon$, the function returns `true` exactly when $C_{\text{total}}(v) = 1$.

**Proof sketch.**
Each Rust check is a direct transcription of the mathematical definition:
- Check 1 implements $C_{\text{bounds}}$ componentwise.
- Check 2 computes $\|v\|_2$ via the standard formula and compares to $L_{\max}$.
- Check 3 computes population variance and compares to $V_{\max}$.

The logical `&&` chain returns `true` iff all three are `true`, which is exactly the definition of $C_{\text{total}}$. The only deviation from the mathematical ideal is IEEE-754 rounding in the sum-of-squares and square root; this error is bounded by $O(l \cdot \varepsilon_{\text{mach}})$ where $\varepsilon_{\text{mach}} \approx 10^{-7}$ for `f32`. By choosing $\varepsilon \gg l \cdot \varepsilon_{\text{mach}}$ in the stability analysis (e.g., $\varepsilon = 10^{-6}$), this round-off is below the tolerance threshold and does not affect the predicate outcome for any realistic latent vector. ∎

---

## 4. Corollaries

### Corollary 4.1 (Constraint Checking is a Convex Feasibility Problem)
The feasible set $F_{\text{bounds}} = \{ v \in \mathbb{R}^l : \min \leq v_j \leq \max \}$ is a hyper-rectangle (convex and compact). The feasible set $F_{L^2} = \{ v : \|v\|_2 \leq L_{\max} \}$ is a Euclidean ball (convex and compact). Their intersection is convex and compact. Therefore, if $C_{\text{total}}$ is satisfiable, there exists a unique *minimum-norm* solution inside the intersection.

> **System implication:** Projecting an out-of-bounds latent onto the feasible set (via clipping + rescaling) is a well-posed optimization with a unique answer.

### Corollary 4.2 (Constraint Violation is Monotonic in Perturbation Magnitude)
Let $\delta \in \mathbb{R}^l$ be a perturbation. If $C_{\text{total}}(v) = 1$ and $C_{\text{total}}(v + \delta) = 0$, then for any scalar $\lambda > 1$:

$$
C_{\text{total}}(v + \lambda \delta) = 0
$$

**Proof.**
At least one constraint is violated at $v + \delta$. All three feasible sets are star-convex around the origin (or around any point inside them). Scaling the perturbation outward moves the point further from the feasible region along the ray, so the violation persists. ∎

> **System implication:** In the breeding operator, if a child's mutation magnitude exceeds the constraint envelope, increasing the mutation scale further cannot "rescue" the vector — the violation is irreversible without projection.

---

## 5. References

- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. (feasibility, projection onto convex sets)
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM. (floating-point error bounds)
- Rust Standard Library: `f32::sqrt`, iterator `.sum()` — exact implementations of the mathematical operations above.
- Sunset Ecosystem source: `src/flux.rs` — `flux_check_batch` implementation.
