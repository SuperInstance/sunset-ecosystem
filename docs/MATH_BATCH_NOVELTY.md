# Formalization of Batch Novelty Computation

## 1. Notation Table

| Symbol | Domain | Description |
|--------|--------|-------------|
| $n$ | $\mathbb{N}$ | Number of rooms (agents) in the fleet |
| $l$ | $\mathbb{N}$ | Latent dimension per room (here $l = 16$) |
| $T$ | $\mathbb{N}$ | Ring buffer depth (here $T = 20$) |
| $H$ | $\mathbb{R}^{T \times n \times l}$ | History ring buffer, $H_{t,i,:} \in \mathbb{R}^l$ is room $i$ at slot $t$ |
| $x$ | $\mathbb{R}^d$ | Input signal vector |
| $W_i$ | $\mathbb{R}^{d \times h \times l}$ | Weight tensor of room $i$ |
| $\sigma$ | $\mathbb{R} \to \mathbb{R}$ | Element-wise activation (e.g., SiLU / ReLU) |
| $\text{latent}_i$ | $\mathbb{R}^l$ | Output of room $i$: $\sigma(W_i \cdot x)$ |
| $\varepsilon$ | $\mathbb{R}_{>0}$ | Numerical stabilizer (e.g., $10^{-6}$) |
| $\|\cdot\|$ | $\mathbb{R}^l \to \mathbb{R}_{\geq 0}$ | Euclidean ($L^2$) norm |

---

## 2. Definitions

### Definition 2.1 (Ring Buffer)
The **history ring buffer** is a finite sequence indexed cyclically:

$$
H = (H^{(0)}, H^{(1)}, \ldots, H^{(T-1)}) \quad \text{with} \quad H^{(t)} \in \mathbb{R}^{n \times l}
$$

At time step $k \in \mathbb{N}$, the write index is $w_k = k \bmod T$. When a new batch of latents $L \in \mathbb{R}^{n \times l}$ arrives, we assign $H^{(w_k)}_{i,:} \leftarrow L_i$ for all rooms $i \in \{1, \ldots, n\}$.

> **Programmer mapping:** `H` is a `[T, n, l]` array; modulo indexing replaces the oldest slice.

### Definition 2.2 (Per-Room Novelty Function)
For a single room $i$, let $\ell \in \mathbb{R}^l$ be its current latent and $h^{(0)}, \ldots, h^{(T-1)} \in \mathbb{R}^l$ be the $T$ historical vectors stored for that room (some possibly uninitialized). Define the **raw novelty** as the minimum squared Euclidean distance to any history entry:

$$
\text{raw\_nv}(\ell; \{h^{(t)}\}) = \min_{t \in \{0,\ldots,T-1\}} \|\ell - h^{(t)}\|^2
$$

The **normalized novelty** is:

$$
nv(\ell; \{h^{(t)}\}) = \frac{\text{raw\_nv}(\ell; \{h^{(t)}\})}{\|\ell\|^2 + \varepsilon}
$$

### Definition 2.3 (Batch Novelty)
For a batch of latents $L \in \mathbb{R}^{n \times l}$ and history buffer $H$, the **batch novelty vector** $N \in \mathbb{R}^n$ is:

$$
N_i = nv(L_i; H_{:,i,:}) \quad \text{for } i = 1, \ldots, n
$$

---

## 3. Theorems

### Theorem 3.1 (Time Complexity — Per Batch)
Computing $N$ for all $n$ rooms requires $O(n \cdot l \cdot T)$ floating-point operations.

**Proof sketch.**
For each room $i$:
- Compute $\|\ell_i\|^2$: $l$ multiplications + $l-1$ additions.
- For each of $T$ history slots, compute $\|\ell_i - h^{(t)}\|^2$: $l$ subtractions, $l$ squares, $2l-1$ additions per slot.
- Find the minimum over $T$ values: $T-1$ comparisons.

Per room: $O(l \cdot T)$. Across $n$ rooms: $O(n \cdot l \cdot T)$. ∎

### Theorem 3.2 (Time Complexity — Per Room, Amortized)
When $T$ is treated as a fixed constant (here $T = 20$), novelty for a single room is $O(1)$ in the dimensionality sense $O(l)$, and $O(l \cdot T) = O(l)$ since $T$ is fixed. More precisely, per-room FLOPs are bounded by a constant independent of batch size or sequence length.

**Proof sketch.**
$T$ does not grow with workload; it is a hyperparameter. Therefore $O(l \cdot T) \in O(l)$ with hidden constant $T = 20$. The ring buffer update (overwriting one slice) is $O(l)$. ∎

### Theorem 3.3 (Numerical Stability)
For any $\ell \in \mathbb{R}^l$ and any $\varepsilon > 0$:

$$
0 \leq nv(\ell; \{h^{(t)}\}) \leq \frac{\|\ell - h^*\|^2}{\varepsilon} < \infty
$$

where $h^*$ is the closest history vector. In particular, the denominator $\|\ell\|^2 + \varepsilon$ is strictly positive, preventing division by zero.

**Proof sketch.**
$\|\ell\|^2 \geq 0$ and $\varepsilon > 0$, so $\|\ell\|^2 + \varepsilon \geq \varepsilon > 0$. The numerator is a squared norm, hence non-negative. The upper bound follows from replacing the denominator with its minimum $\varepsilon$. ∎

### Theorem 3.4 (Equivalence to Online k-Means Residual, T = 1)
When $T = 1$, let $h$ be the single stored history vector. Then:

$$
nv(\ell; \{h\}) = \frac{\|\ell - h\|^2}{\|\ell\|^2 + \varepsilon} \approx \frac{\|\ell - h\|^2}{\|\ell\|^2}
$$

This is the squared **residual** of $\ell$ relative to centroid $h$, normalized by the squared norm of $\ell$. When $\|\ell\| \gg \sqrt{\varepsilon}$, this coincides with the online k-means objective: the normalized distance from a point to its cluster centroid.

**Proof sketch.**
With one history slot, the minimum in Definition 2.2 collapses to a single term. The expression $\|\ell - h\|^2 / \|\ell\|^2$ is precisely the squared cosine residual if we expand:

$$
\frac{\|\ell - h\|^2}{\|\ell\|^2} = 1 - 2\frac{\ell \cdot h}{\|\ell\|^2} + \frac{\|h\|^2}{\|\ell\|^2}
$$

In online k-means, the loss for assigning $\ell$ to centroid $h$ is exactly $\|\ell - h\|^2$. Normalizing by $\|\ell\|^2$ makes the score scale-invariant, which is necessary when rooms may have divergent weight magnitudes. ∎

---

## 4. Corollaries

### Corollary 4.1 (Scale Invariance)
For any scalar $\alpha > 0$:

$$
nv(\alpha \ell; \{\alpha h^{(t)}\}) = nv(\ell; \{h^{(t)}\})
$$

**Proof.** Both numerator and denominator scale by $\alpha^2$, which cancels. ∎

> **System implication:** Rooms with larger weight magnitudes do not automatically dominate novelty scores; the system remains fair across heterogeneous agents.

### Corollary 4.2 (Fresh-Start Guarantee)
If all history slots for room $i$ are initialized to zeros and the first latent $\ell$ is non-zero, then:

$$
N_i = \frac{\|\ell\|^2}{\|\ell\|^2 + \varepsilon} \approx 1
$$

**Proof.** $\text{raw\_nv} = \|\ell - 0\|^2 = \|\ell\|^2$. Substitute into Definition 2.2. ∎

> **System implication:** A newly spawned agent (breeding child) automatically receives maximum novelty on its first step, encouraging exploration before history accumulates.

---

## 5. References

- Lloyd, S. (1982). "Least squares quantization in PCM." *IEEE Trans. Info. Theory* 28(2), 129–137. (k-means foundation)
- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations." *Proc. 5th Berkeley Symp.*, 281–297. (online k-means)
- Sunset Ecosystem source: `src/novelty.rs` — `batch_novelty` implementation.
