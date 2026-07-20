# Aegis Engine — Math Notes

Conceptual reference for the math behind each module. The code carries the
same explanations in docstrings; this file is the connected narrative —
useful when re-deriving or re-implementing a piece from scratch.

---

## Phase 1 · Week 2 — Covariance estimation

### Why Σ is the load-bearing input

Mean-variance optimization computes portfolio weights proportional to
**Σ⁻¹μ**: the *inverse* covariance matrix times expected returns. Because Σ
is inverted, error in Σ does not stay the size it started — the inverse
amplifies it, and it amplifies it most along the direction of the
**smallest eigenvalue**, which is exactly the direction Σ estimates least
well. A covariance bug is therefore a *leveraged* bug. Everything in Week 2
is about producing a Σ that is safe to invert.

Three estimators are built, in increasing sophistication. All operate on
**log returns** over the common period (where every asset has data), and
all return a symmetric, positive-semidefinite matrix with the tickers in a
fixed column order (row/col *i* is always asset *i* — a contract the
optimizer relies on).

---

### 1 · Sample covariance (baseline)

$$\Sigma_{ij} = \frac{1}{T-1}\sum_{t=1}^{T} (r_{it}-\bar r_i)(r_{jt}-\bar r_j)$$

The `(T-1)` denominator is **Bessel's correction**: dividing by `T-1`
rather than `T` makes the estimator unbiased for the population covariance
(one degree of freedom is spent estimating each mean). This is the
maximum-likelihood estimate up to that correction, and the natural baseline
every other estimator is measured against.

**Its weakness — the reason EWMA and Ledoit-Wolf exist:** with `N` assets
there are `N(N+1)/2` free parameters to estimate. When `T` is not vastly
larger than `N`, the smallest eigenvalues of Σ are estimated poorly, and
Σ⁻¹ blows those errors up. It also weights a return from 18 months ago
exactly as much as yesterday's — a problem because markets are
non-stationary (volatility clusters).

---

### 2 · EWMA covariance (RiskMetrics)

$$\Sigma = \sum_{t=1}^{T} w_t\, x_t x_t^\top,\qquad w_t \propto \lambda^{(T-1)-t},\qquad \sum_t w_t = 1$$

with `t` indexing observations oldest→newest, so the most recent day gets
the largest weight and influence decays geometrically into the past.

- **λ = 0.94** is J.P. Morgan RiskMetrics' daily standard. It sets the
  *memory*: the weight on a day `i` steps back is `∝ λ^i`, giving a
  **half-life** of `ln(0.5)/ln(0.94) ≈ 11.2` trading days. Only the last
  ~two weeks meaningfully drive the estimate.
- **No demeaning.** RiskMetrics assumes a zero daily mean and forms the raw
  second moment `E[x xᵀ]`. At daily frequency the mean is negligible next
  to the volatility, and λ = 0.94 was calibrated under that assumption — so
  we follow it (`ewma_demean = False`). A flag switches to a true weighted
  covariance for research.
- **PSD by construction:** a convex combination (weights ≥ 0 summing to 1)
  of rank-1 outer products `x_t x_tᵀ` is always positive semidefinite.

EWMA's cost is a *shorter effective sample* — roughly `1/(1-λ) ≈ 17` days
of real information — so the estimate tracks the current regime but is
noisier. A different trade-off than shrinkage, not a strictly better one.

**Boundary check that pins the implementation:** as `λ → 1` the weights
become uniform, so EWMA (with demeaning) collapses to the `1/T` sample
covariance — a known quantity the test suite asserts.

---

### 3 · Ledoit-Wolf shrinkage (primary estimator)

The sample covariance `S` is unbiased but noisy; a structured target `F` is
biased but stable. Neither is ideal alone, so take a convex combination:

$$\Sigma^* = \delta\,F + (1-\delta)\,S,\qquad F = \mu I,\qquad \mu = \frac{\operatorname{tr}(S)}{N}$$

The target `μI` asserts "every asset has the same variance μ and zero
correlation" — deliberately wrong, but perfectly conditioned (condition
number 1). Shrinking `S` toward it pulls the extreme eigenvalues inward:
the tiny ones (which `S⁻¹` explodes) rise toward μ, the huge ones fall
toward μ. That directly treats the fragility the optimizer suffers from.

**The intensity δ is not tuned — it is derived.** Ledoit-Wolf (2004) give
the δ minimizing expected squared error `E‖Σ* − Σ_true‖²`, estimated from
the data, using the normalized inner product `⟨A,B⟩ = tr(ABᵀ)/N`:

$$
\mu = \langle S, I\rangle,\quad
d^2 = \lVert S-\mu I\rVert^2,\quad
\bar b^2 = \frac{1}{T^2}\sum_{k=1}^{T}\lVert x_k x_k^\top - S\rVert^2,
$$
$$
b^2 = \min(\bar b^2, d^2),\qquad
\delta = \frac{b^2}{d^2}.
$$

Read it as a ratio: **δ = (sampling noise in S) / (distance of S from the
target)**. Shrink hard when `S` is noisy and already close to the target;
barely shrink when `S` is precise and the target is clearly wrong. Because
`b² ≤ d²`, δ ∈ [0, 1], so Σ* is a genuine convex combination — hence
symmetric and PSD for free.

**Convention notes (they matter for reproducibility):**
- `S` here is the **biased 1/T** MLE covariance, not the `1/(T-1)` one the
  baseline returns. The δ formula is derived for the 1/T version; the two
  differ only by a scalar `T/(T-1)`, which changes neither the condition
  number nor the shrinkage direction.
- Returns are demeaned by their sample mean before forming `S`.
- `b̄²` is implemented by its **direct definition** (average squared
  distance of each observation's outer product from `S`) so the code reads
  like the formula. It is cross-checked in the tests against the algebraic
  closed form `b̄² = (1/(NT²))Σ‖xₖ‖⁴ − (1/T)‖S‖²` — two independent
  derivations agreeing is the correctness guarantee (there is no
  sklearn/scipy in the environment to check against).

---

### 4 · Diagnostics

Both read-outs come from the eigenvalues of Σ, obtained with the symmetric
solver `eigh` (real, ordered eigenvalues; orthonormal eigenvectors).

**Condition number** `κ(Σ) = λ_max / λ_min`. How much Σ⁻¹ amplifies error;
`κ = 1` is spherical, thousands means a knife-edge. Reported as `log₁₀κ`
because conditioning spans orders of magnitude: ~0 excellent, 1–2 workable,
3+ painful, each unit a 10× jump. `λ_min ≤ 0` ⇒ singular ⇒ `κ = +∞`.

**Eigenstructure.** `Σ = VΛVᵀ`: each eigenvector is a portfolio whose
returns are uncorrelated with every other eigen-portfolio, and its
eigenvalue is that portfolio's variance. So the eigenvalues answer *how
many genuinely independent sources of risk exist*. Two measures:
- **Factor count** — eigenvalues explaining ≥ 5% of total variance (hard,
  threshold-based).
- **Participation ratio** `(Σλ)² / Σλ²` — a threshold-free "effective
  number of factors": `N` when variance is spread evenly, → 1 when one
  factor dominates.

---

### Week 2 empirical baseline (VXUS universe)

Common period **2024-01-12 → 2026-04-27**, T = 573 days, N = 4.

| Estimator | log₁₀ κ | κ | Note |
|-----------|--------:|----:|------|
| Sample | 1.599 | 39.7 | baseline |
| EWMA (λ=0.94) | 1.535 | 34.3 | recency-weighted |
| Ledoit-Wolf | 1.508 | 32.2 | δ = 0.017 |

- **Shrinkage is a light touch here (δ ≈ 0.017)** because `T = 573 ≫ N = 4`
  — the sample covariance is already trustworthy, so LW barely shrinks.
  This is the estimator behaving *correctly*: shrinkage earns its keep when
  `N/T` is large, which is not this regime. It would matter far more on a
  50-asset universe with the same history.
- **QQQ-VXUS correlation = 0.72**, down from VT's 0.92 (see ADR-003). The
  universe swap already did most of the conditioning work that shrinkage
  would otherwise have had to do — landmine (2) from the Week 1 review is
  largely defused.
- **One dominant risk factor:** factor 1 explains **73%** of variance (the
  market/beta direction); participation ratio **1.76**. Four assets, but
  effectively ~1.8 independent bets — the honest diversification the
  optimizer has to work with.

---

### References

- O. Ledoit and M. Wolf (2004). *A well-conditioned estimator for
  large-dimensional covariance matrices.* Journal of Multivariate Analysis
  88(2), 365–411.
- J.P. Morgan/Reuters (1996). *RiskMetrics — Technical Document*, 4th ed.
  (source of the λ = 0.94 daily decay convention).
