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

---

## Phase 1 · Week 3 — Mean-variance optimization

### The problem and its two fragile inputs

Markowitz mean-variance optimization chooses weights `w` trading expected
return `μᵀw` against risk `wᵀΣw`. Every solution is proportional to **Σ⁻¹μ**,
so the two inputs are:

- **Σ** — handled in Week 2. Ill-conditioning makes `Σ⁻¹` brittle; we feed
  the well-conditioned Ledoit-Wolf estimate and solve rather than invert.
- **μ** — handled here. The sample mean has standard error `σ/√T`, so it is
  the noisiest input, and MVO amplifies errors in μ about **10× more** than
  errors in Σ (Chopra-Ziemba 1993). This is why μ gets shrunk.

### Expected-return shrinkage (Bayes-Stein, Jorion 1986)

Raw historical means make MVO chase in-sample flukes. Shrinkage pulls the
mean vector toward a single stable anchor — the GMV portfolio's return
`μ_min`, which depends only on Σ:

$$\mu_{BS} = (1-\varphi)\hat\mu + \varphi\,\mu_{\min}\mathbf{1}, \qquad
\mu_{\min} = \frac{\mathbf{1}^\top\Sigma^{-1}\hat\mu}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}$$

$$\varphi = \frac{N+2}{(N+2) + T\,(\hat\mu-\mu_{\min}\mathbf1)^\top\Sigma^{-1}(\hat\mu-\mu_{\min}\mathbf1)}$$

φ → 1 (shrink hard) when the means are close together relative to Σ (the
spread is probably noise); φ → 0 when they are strongly separated. On the
current universe **φ ≈ 0.78** — the mean dispersion collapses from 8.0% to
1.8%, and the naive gold-heavy tangency becomes diversified.

### Closed-form results (Lagrange multipliers)

**Global minimum variance** — minimise `wᵀΣw` s.t. `𝟙ᵀw = 1`:

$$w_{gmv} = \frac{\Sigma^{-1}\mathbf1}{\mathbf1^\top\Sigma^{-1}\mathbf1}$$

**Efficient frontier** — add the return constraint `μᵀw = m`. Two
multipliers give an *affine* solution (the two-fund theorem):

$$w(m) = g + h\,m, \quad
\sigma^2(m) = \frac{A m^2 - 2Bm + C}{D}$$

with `A = 𝟙ᵀΣ⁻¹𝟙`, `B = 𝟙ᵀΣ⁻¹μ`, `C = μᵀΣ⁻¹μ`, `D = AC − B²`, and
`g, h` the corresponding combinations of `Σ⁻¹𝟙` and `Σ⁻¹μ`. Variance is a
parabola in the target return; its vertex is the GMV point `(1/A, B/A)`.

**Tangency (max Sharpe)** — with a risk-free rate `r_f`:

$$w_{tan} = \frac{\Sigma^{-1}(\mu - r_f\mathbf1)}{\mathbf1^\top\Sigma^{-1}(\mu - r_f\mathbf1)}$$

All three apply `Σ⁻¹` via a linear solve. Weights are invariant to whether
μ/Σ/r_f are daily or annualised, as long as they are consistent.

### Constrained optimization (long-only, sector caps)

Inequality constraints (`wᵢ ≥ 0`, `Σ_sector wᵢ ≤ cap`) remove the closed
form — the problem becomes a QP with an unknown active set. Solved with
scipy **SLSQP**, minimising `wᵀΣw` (or `−Sharpe`, or hitting a target
return) subject to full investment, the box bounds, and one ≤-cap
inequality per capped sector. The constrained minimum variance is always
≥ the unconstrained GMV — constraints can only cost you. See ADR-005 for
why a solver rather than a hand-rolled QP.

### Week 3 empirical result

| Portfolio | QQQ | GLDM | FBTC | VXUS | ret | vol | Sharpe |
|-----------|----:|----:|----:|----:|----:|----:|----:|
| Tangency (raw μ) | +37% | **+69%** | −2% | −4% | 32% | 17% | 1.59 |
| Tangency (shrunk μ) | +21% | +36% | −2% | +45% | 25% | 15% | 1.35 |
| Max-Sharpe (long-only) | +19% | +36% | 0% | +45% | 25% | 14% | 1.34 |

The raw-μ tangency is the landmine: 69% gold, shorting the diversifiers.
Shrinkage plus long-only/sector caps turns it into a portfolio that
actually spreads risk. Note the shrunk max-Sharpe ≈ min-variance, because
heavy shrinkage flattened the returns — with little return signal left, the
best Sharpe portfolio is essentially the lowest-risk one. That is the
honest conclusion on 573 days of data, not a bug.

### References (Week 3)

- H. Markowitz (1952). *Portfolio Selection.* Journal of Finance 7(1).
- P. Jorion (1986). *Bayes-Stein Estimation for Portfolio Analysis.*
  Journal of Financial and Quantitative Analysis 21(3), 279–292.
- V. Chopra and W. Ziemba (1993). *The Effect of Errors in Means,
  Variances, and Covariances on Optimal Portfolio Choice.* Journal of
  Portfolio Management 19(2).

---

## Phase 1 · Week 4 — Backtesting with frictions

### Why frictions are the whole point

A frictionless backtest flatters every strategy: it rebalances for free and
reports a Sharpe no live account will see. Week 4 makes the three frictions
that actually erode a rebalancing strategy explicit and measurable.

**Weight-space simulation.** The engine runs in weight space, not shares.
Each day the held weights drift with returns,

$$w_{t,i}^{\text{end}} = \frac{w_{t,i}(1+r_{t,i})}{1 + w_t^\top r_t},$$

which automatically stays fully invested (the denominator renormalises).
Gross portfolio return on day t is simply `wₜᵀrₜ`.

**Band rebalancing (the no-trade region).** Rebalancing back to target fires
only when some asset has drifted more than the band from its target:
`maxᵢ |wₜ,ᵢᵉⁿᵈ − targetᵢ| > band`. Trade every day and costs eat you; never
trade and the book drifts off target. The band is the standard compromise —
here ±5 percentage points (absolute), which produced just **2 rebalances in
573 days**. (Absolute points, not % of target, so a 0-weight asset does not
get a zero-width — always-tripping — band.)

**Costs.** On a rebalance the cost charged is
`(one-way turnover) × total_cost_bps`, where one-way turnover
`= ½·Σ|w_new − w_old|` is the fraction of the book actually traded, and
`total_cost_bps = transaction_cost_bps + slippage_bps` (15 bps here).

### Metrics and attribution

Standard risk/return summary: total return, CAGR, annualised return/vol,
Sharpe, **max drawdown computed from the equity path** (a drawdown is
path-dependent — it cannot be read off mean and variance), Calmar, win rate.

Attribution is done in **arithmetic** space because only arithmetic
contributions are additive: asset i's daily contribution `wₜ,ᵢ·rₜ,ᵢ` sums,
over assets, to that day's gross return, so the per-asset totals partition
the gross return exactly. Cost drag is subtracted as its own explicit line.

### Week 4 empirical result

Long-only max-Sharpe portfolio, ±5% bands, 15 bps costs, over 573 days:

| Strategy | CAGR | Vol | Sharpe | Max DD | Calmar |
|----------|----:|----:|----:|----:|----:|
| Optimized (banded) | 30.4% | 14.4% | **1.56** | −12.2% | 2.50 |
| Equal-weight (banded) | 31.9% | 19.4% | 1.26 | −14.6% | 2.18 |
| Optimized (buy & hold) | 30.6% | 15.0% | 1.53 | −13.3% | 2.30 |

The honest read: the optimizer earned slightly *less* than naive
equal-weight (30.4% vs 31.9%) but at meaningfully lower risk — higher
Sharpe (1.56 vs 1.26), shallower drawdown, lower vol. Its value is
risk reduction, not return maximisation, which is exactly what mean-variance
optimization promises and all it should be credited with. Cost drag was
negligible (0.02%) because the band kept turnover to 10% over two years —
demonstrating the no-trade region doing its job.

### References (Week 4)

- R. Grinold and R. Kahn (1999). *Active Portfolio Management*, 2nd ed.
  (transaction-cost-aware implementation).

---

## Phase 2 — Signal development

Phase 1 assumes the world is stationary: one Σ, one μ, one optimal portfolio
for all time. Phase 2 drops that assumption. It adds a model of *which market
we are in* (regimes), a rule for *responding to it* (adaptive weights), and a
test for *which assets are structurally tied together* (cointegration).

### 1 · Gaussian Hidden Markov Model

The model says: an unobserved state `zₜ ∈ {1..K}` evolves as a Markov chain,
and the observed return is Gaussian conditional on that state.

```
z₁ ~ π                       initial state distribution
zₜ | zₜ₋₁ ~ A[zₜ₋₁, ·]       K×K transition matrix
xₜ | zₜ = k ~ N(μₖ, Σₖ)      state-conditional emission
```

Parameters `θ = (π, A, {μₖ}, {Σₖ})`. Three problems, three algorithms:

**Evaluation — the forward algorithm.** Define `αₜ(k) = P(x₁..ₜ, zₜ=k)`.

```
α₁(k) = πₖ · N(x₁ | μₖ, Σₖ)
αₜ(k) = N(xₜ | μₖ, Σₖ) · Σⱼ αₜ₋₁(j)·A[j,k]
P(x | θ) = Σₖ α_T(k)
```

Naively this underflows: α is a product of T probabilities, so for T=573 it
is ~10⁻¹⁵⁰⁰. The implementation works entirely in **log space**, replacing
`Σᵢ αᵢ` with the log-sum-exp trick `log Σᵢ e^{aᵢ} = a* + log Σᵢ e^{aᵢ−a*}`
where `a* = max aᵢ`. This is the single most important numerical decision in
the module — the scaling-factor alternative is equivalent but harder to audit.

**Learning — Baum-Welch (EM).** The backward variable
`βₜ(k) = P(xₜ₊₁..T | zₜ=k)` completes the picture. The E-step forms the
state posteriors and the pairwise transition posteriors:

```
γₜ(k) = P(zₜ=k | x)     ∝ αₜ(k)·βₜ(k)
ξₜ(j,k) = P(zₜ=j, zₜ₊₁=k | x) ∝ αₜ(j)·A[j,k]·N(xₜ₊₁|μₖ,Σₖ)·βₜ₊₁(k)
```

The M-step is then just γ- and ξ-weighted maximum likelihood:

```
πₖ  = γ₁(k)
A[j,k] = Σₜ ξₜ(j,k) / Σₜ γₜ(j)
μₖ  = Σₜ γₜ(k)·xₜ / Σₜ γₜ(k)
Σₖ  = Σₜ γₜ(k)·(xₜ−μₖ)(xₜ−μₖ)' / Σₜ γₜ(k)
```

EM guarantees the log-likelihood is **non-decreasing** every iteration. That
guarantee is the module's primary correctness test (`test_em_monotonic`): if
the E and M steps are inconsistent, the likelihood will dip.

*One deliberate exception.* A regulariser `Σₖ += εI` (default ε=1e-6) is added
to keep the state covariances invertible when a state captures few
observations. This trades the exact monotonicity guarantee for numerical
robustness, and produces dips of order 1e-3 on a log-likelihood of ~5000.
The monotonicity test therefore runs with ε=0 to check the pure EM
implementation, while production fitting keeps the regulariser.

**Decoding — Viterbi.** The most likely *state sequence* (not the most likely
state at each time — they differ) via the max-product recursion in log space:

```
δₜ(k) = max_j [ δₜ₋₁(j) + log A[j,k] ] + log N(xₜ | μₖ, Σₖ)
```

with backpointers, then a backward pass to recover the path.

### 2 · From states to regimes

The HMM returns K anonymous states. Naming them is a modelling decision made
*after* fitting: states are ranked by fitted volatility `√Σₖ` and assigned, in
ascending order, `low_vol_trending → high_vol_meanrev → crisis`.

This ordering is deliberate. The alternative — pinning states to labels by
seeding the EM with hand-chosen means — biases the fit toward the answer you
expect. Ranking after the fact keeps the statistics honest (EM optimises
likelihood, nothing else) and the labels interpretable.

**Empirical validation.** Fit on the equal-weight portfolio return over
573 days (2024-01-12 → 2026-04-27), the crisis state captured 19 days in three
contiguous episodes:

| Episode | Days | Known event |
|---------|-----:|-------------|
| 2024-08-01 → 2024-08-08 | 6 | Yen carry-trade unwind |
| 2025-04-03 → 2025-04-11 | 7 | Tariff announcement shock |
| 2026-01-30 → 2026-02-06 | 6 | — |

| Regime | Ann. return | Ann. vol | Days | Frequency |
|--------|----:|----:|----:|----:|
| low_vol_trending | +58.9% | 11.3% | 120 | 20.9% |
| high_vol_meanrev | +29.8% | 18.7% | 434 | 75.7% |
| crisis | −165.7% | 53.1% | 19 | 3.3% |

The model was given no event calendar. It rediscovered the August 2024 and
April 2025 stress episodes from the return series alone — evidence it is
extracting genuine regime structure rather than partitioning noise.

### 3 · Adaptive weights

Given base weights `w` from the Phase 1 optimizer and a regime `r`, apply a
**multiplicative** tilt by asset class and renormalise:

```
w̃ᵢ = wᵢ · m_r(class(i))          m_r from config
w*  = w̃ / Σⱼ w̃ⱼ
```

Multiplicative-and-renormalise is chosen over additive tilts because it
preserves the Phase 1 constraints automatically. A positive weight scaled by a
positive multiplier stays positive (long-only holds), the renormalisation
restores the budget (Σw = 1 holds), and a zero weight stays zero — the engine
tilts what you already hold rather than opening new positions.

Implied turnover is `½Σᵢ|w*ᵢ − wᵢ|`, which feeds the Phase 1 cost model
directly, so a regime switch has a quantifiable price.

Empirically, on the max-Sharpe base (QQQ 19.1%, GLDM 36.2%, VXUS 44.7%):

| Regime | QQQ | GLDM | VXUS | Turnover |
|--------|----:|----:|----:|----:|
| base | 19.1% | 36.2% | 44.7% | — |
| low_vol_trending | 21.6% | 27.8% | 50.6% | 8.4% |
| high_vol_meanrev | 19.1% | 36.2% | 44.7% | 0.0% |
| crisis | 12.4% | 58.6% | 29.0% | 22.5% |

Directionally correct: risk-on lifts equity and cuts gold; crisis does the
reverse, moving gold to a majority holding.

### 4 · Cointegration

**Correlation vs cointegration.** Correlation describes co-movement of
*returns*. Cointegration is a stronger, structural claim about *prices*: two
non-stationary series `y`, `x` are cointegrated if some linear combination
`y − βx` is stationary. Correlated assets can drift apart forever;
cointegrated ones cannot. Only cointegration justifies a mean-reversion trade
on the spread, because only then is reversion statistically guaranteed.

**Engle-Granger, two steps.**

1. OLS `y = α + βx` on log prices → hedge ratio β and spread `s = y − α − βx`.
2. Test `s` for a unit root (ADF). Rejecting the unit-root null means the
   spread is stationary, so the pair is cointegrated.

The test is run on *ordered* pairs: regressing y on x gives a different hedge
ratio than x on y, so both directions are reported.

**Johansen trace test.** Engle-Granger handles one pair at a time and assumes
a single cointegrating vector. Johansen tests the whole system jointly and
returns the cointegration **rank** r — how many independent long-run
relationships exist among all N assets. The trace statistic for `H₀: rank ≤ r`
is compared to its 95% critical value; the rank is the number of hypotheses
rejected.

**Spread signal.** The spread is standardised to `z = (s − μ_s)/σ_s`, and
positions are taken with **hysteresis**: enter at `|z| > entry_z` (short the
spread when z is high, long when low — betting on reversion), exit only when
`|z| < exit_z`, with `exit_z < entry_z`. The gap between the two thresholds is
what prevents a position from flipping on every threshold cross while the
spread oscillates around the entry level — the difference between one trade
and forty.

**Empirical result — an honest negative.** On this universe, **no pair is
cointegrated** at α=0.05 (best: GLDM~VXUS, EG p=0.111), and the Johansen trace
test returns **rank r = 0**: no long-run relationship among the four assets at
all.

This is the correct and expected answer, not a failure. The universe was
built (ADR-003) so that each asset hedges a *different* failure mode — tech
beta, inflation, monetary debasement, ex-US growth. Assets deliberately chosen
not to share a common driver should not share a stochastic trend. Finding
cointegration here would have suggested the diversification thesis was wrong.

The correct statistical arbitrage universe is same-sector pairs (KO/PEP,
GLD/IAU, sector-ETF vs its own constituents), which is a different portfolio
than this one. The machinery is built, tested, and correct; it reports that
this universe has nothing to trade — which is exactly what a test is for.

### References (Phase 2)

- L. Rabiner (1989). "A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition." *Proc. IEEE* 77(2). — the canonical
  forward-backward / Baum-Welch / Viterbi reference.
- R. Engle and C. Granger (1987). "Co-integration and Error Correction."
  *Econometrica* 55(2).
- S. Johansen (1991). "Estimation and Hypothesis Testing of Cointegration
  Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59(6).
- A. Ang and G. Bekaert (2002). "International Asset Allocation with
  Regime Shifts." *Review of Financial Studies* 15(4). — regime-conditioned
  allocation.
