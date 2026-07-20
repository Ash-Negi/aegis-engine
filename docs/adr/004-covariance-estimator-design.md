# ADR-004: Implement covariance estimators from scratch, shrink toward a scaled identity

## Status

Accepted (2026-07-20)

## Context

Week 2 needs a covariance matrix Σ for the Week 3 optimizer. Σ is inverted
(`Σ⁻¹μ`), so its conditioning — not just its point values — determines how
stable the resulting weights are. Several real choices had to be made, each
with a defensible alternative:

1. **Build the estimators or import them?** `sklearn.covariance` ships
   `LedoitWolf` and `EmpiricalCovariance`; `scipy` offers eigen-tools.
   Neither is installed in this project's venv (only numpy/pandas/matplotlib).
2. **Which Ledoit-Wolf target?** The 2004 paper shrinks toward a scaled
   identity `μI`; the 2003 variant shrinks toward a constant-correlation
   matrix. They optimize for different structures.
3. **Which normalization for the sample covariance inside LW?** The δ
   derivation assumes the biased `1/T` MLE covariance, whereas the textbook
   baseline uses the unbiased `1/(T-1)` form.
4. **Demean for EWMA?** RiskMetrics assumes a zero daily mean and does not
   demean; a "true" weighted covariance would.

The forcing function is the project's stated ethos (CLAUDE.md): *translate
textbook math into auditable code, no magic numbers buried in logic,
modules are testable functions.*

## Decision

**Implement all three estimators by hand in numpy** (`covariance/estimators.py`),
plus conditioning/eigenstructure diagnostics (`covariance/diagnostics.py`).

- **Sample** covariance uses the unbiased `1/(T-1)` normalization — the
  textbook baseline, matching `np.cov` / `pandas.cov`.
- **EWMA** follows RiskMetrics: `λ = 0.94`, weights `∝ λ^{(T-1)-t}`
  renormalized to sum to 1, and **no demeaning** by default
  (`CovarianceConfig.ewma_demean = False`), so the λ calibration is used as
  intended.
- **Ledoit-Wolf** shrinks the **biased `1/T`** sample covariance toward the
  **scaled-identity target `μI`**, with δ computed in closed form per the
  2004 paper. `b̄²` is written by its direct outer-product definition so the
  code mirrors the math; the tests cross-check it against the independent
  algebraic closed form.

Every knob lives in `CovarianceConfig`. Ledoit-Wolf has *no* free
parameter — δ is data-determined, which is the whole point.

## Consequences

### Benefits

- **Auditable.** The shrinkage intensity and the covariance are traceable
  from data to output with no library black box. `ledoit_wolf_shrinkage()`
  returns δ, the target, the sample, and μ so the report can *show* the
  mechanism, not just the answer.
- **Correctness without a reference implementation.** With no sklearn to
  diff against, correctness rests on two independent derivations of `b̄²`
  agreeing (direct loop vs. closed form) and on properties provable on
  paper — δ ∈ [0,1], PSD, conditioning strictly improves. That is a
  stronger guarantee than "matches the library."
- **Scaled-identity target directly serves the goal.** The Week 2 brief
  asks for the log condition number before/after shrinkage; `μI` is the
  target whose condition number is exactly 1, so shrinking toward it is the
  most direct lever on conditioning.
- **Zero dependency footprint.** Nothing added to `requirements.txt`; the
  math engine stays numpy/pandas only.

### Tradeoffs

- **Scaled identity ignores the correlation structure.** The 2003
  constant-correlation target would preserve the average pairwise
  correlation and is often a better fit for equity-heavy universes. We
  accepted the simpler, condition-number-focused target for Week 2; a
  constant-correlation variant can be added later as a second target
  behind the same interface if the optimizer wants it.
- **Two normalization conventions coexist.** The baseline is `1/(T-1)`, the
  LW internal sample is `1/T`. They differ by a scalar `T/(T-1)` that
  changes neither conditioning nor shrinkage direction, but the split is
  documented in the code and math notes to avoid confusion.
- **δ is tiny on the current universe (≈ 0.017).** With `T = 573 ≫ N = 4`
  the sample covariance is already trustworthy, so shrinkage barely moves
  Σ. This is correct behaviour, not a bug — but it means the estimator's
  value is latent here and only becomes material on a larger universe or a
  shorter window. Documented in `docs/math.md` so the small number is not
  mistaken for a mistake.

### Alternatives considered

- **Use `sklearn.covariance.LedoitWolf`.** Rejected: not installed, and
  importing a black box contradicts the auditable-by-design ethos. The
  from-scratch version is ~30 lines and fully tested.
- **Constant-correlation target (Ledoit-Wolf 2003).** Deferred, not
  rejected — a reasonable second target once the optimizer exists to reveal
  which conditioning behaviour it prefers.
- **Demean EWMA.** Rejected as the default because it would desynchronize
  the estimator from the λ = 0.94 RiskMetrics calibration; kept available
  behind a flag.
