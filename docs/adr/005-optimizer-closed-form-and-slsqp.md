# ADR-005: Closed form where it exists, SLSQP where it doesn't; shrink the mean vector

## Status

Accepted (2026-07-20)

## Context

Week 3 turns (μ, Σ) into weights. Two design questions had real alternatives:

1. **How to solve the optimization.** The unconstrained mean-variance
   problem has a closed-form Lagrange solution. The moment you add
   inequality constraints (long-only `wᵢ ≥ 0`, sector caps
   `Σ_sector wᵢ ≤ cap`) that solution disappears — it becomes a quadratic
   program with an unknown active set, which has no formula. Options:
   hand-roll an active-set/projected-gradient QP (in the spirit of the
   from-scratch Ledoit-Wolf, ADR-004), or use a solver (scipy SLSQP).

2. **What expected-return vector to feed it.** The Week 1 review (landmine
   3) showed every asset had large positive in-sample returns, so vanilla
   MVO on raw historical means concentrates in whatever had the best
   in-sample Sharpe (gold) and shorts the rest. MVO is far more sensitive
   to errors in μ than in Σ (Chopra-Ziemba 1993). Options: raw means, or a
   shrinkage estimator, and if shrinkage, which target.

## Decision

**Split the optimizer by whether closed form exists.**

- `optimizer/mean_variance.py` — the unconstrained results in **closed
  form** (Lagrange): GMV `w = Σ⁻¹𝟙/(𝟙ᵀΣ⁻¹𝟙)`, tangency, and the two-fund
  efficient frontier `w(m) = g + h·m`. `Σ⁻¹` is always applied via
  `np.linalg.solve`, never an explicit inverse.
- `optimizer/constrained.py` — long-only + sector-capped portfolios via
  **scipy SLSQP** (sequential least-squares QP), with analytic gradients
  for the variance objective and the linear constraints. Sectors are the
  universe asset classes; the default equity cap (0.65) prevents doubling
  up on the 0.72-correlated QQQ+VXUS pair.

**Shrink the mean vector with Bayes-Stein** (Jorion 1986,
`optimizer/expected_returns.py`): pull the sample mean toward the
global-minimum-variance portfolio's return `μ_min`, with a data-derived
intensity φ. On the current sample φ ≈ 0.78 — heavy — and it turns the
raw-mean tangency (69% gold, shorts VXUS) into a diversified 36% gold /
45% VXUS portfolio.

## Consequences

### Benefits

- **The closed form is the teaching artifact**; the constrained solver is
  what a desk would actually trade. Keeping them in separate modules makes
  the "here is the math, here is the practice" distinction explicit.
- **Shrinkage is auditable and visible.** `ExpectedReturns` carries the raw
  mean, the shrunk mean, φ, and the target, so the report shows the before/
  after rather than asserting it.
- **Numerically safe.** No explicit `Σ⁻¹`; a well-conditioned Ledoit-Wolf Σ
  feeds the solve.

### Tradeoffs

- **SLSQP is a black box in *how* it optimizes** — a departure from the
  from-scratch Ledoit-Wolf decision. Accepted because a robust
  general-purpose QP is not where the insight is, and SLSQP is transparent
  in *what* it optimizes (objective + constraints are our code). A
  hand-rolled QP would be more code, more bugs, no more understanding.
- **SLSQP is ~1000× slower than the closed form** (milliseconds vs
  microseconds) and can fail to converge at the extreme edge of the frontier;
  the frontier tracer skips non-converged targets rather than aborting.
- **Bayes-Stein uses the simplified intensity** (no `T/(T−N−2)` predictive-
  covariance correction). The difference is O(1/T) and immaterial at T=573;
  documented in `docs/math.md`.

### Alternatives considered

- **Hand-roll the QP.** Rejected for the reasons above; revisit only if a
  dependency-free build becomes a hard requirement.
- **Raw historical means.** Rejected — it *is* the landmine. Kept available
  via `return_shrinkage="none"` for the explicit before/after comparison.
- **Black-Litterman.** A richer way to blend a prior with views, but it
  needs a market-cap equilibrium and view matrix that belong to a later
  phase. Bayes-Stein is the right weight of machinery for Week 3.
