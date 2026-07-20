# ADR-006: Build the HMM from scratch; delegate the cointegration critical-value tables

## Status

Accepted (2026-07-20)

## Context

Phase 2 needs two pieces of statistical machinery, and the build-vs-borrow
answer is different for each.

1. **Regime detection** requires a Gaussian HMM: forward-backward, Baum-Welch
   EM, Viterbi. `hmmlearn` provides all of it, is well tested, and was listed
   in the original tech stack.

2. **Cointegration testing** requires the Engle-Granger two-step test, the
   ADF unit-root test, and the Johansen trace test. `statsmodels` provides all
   of it.

Both are "should we implement this ourselves?" questions with the same surface
shape, so it would be tempting to answer them the same way. They should not
be answered the same way, because the thing being borrowed is different in
kind.

## Decision

**Build the HMM from scratch** (`signals/hmm.py`), and **delegate the
cointegration test statistics to statsmodels** (`signals/cointegration.py`),
keeping only the trading logic in-house.

The HMM is an *algorithm*. Its correctness is verifiable from first
principles, on data whose true parameters are known:

- EM's defining guarantee is that the log-likelihood is non-decreasing every
  iteration. A bug in either the E-step or the M-step breaks it. This is a
  complete, self-checking correctness test — and it is the primary test in
  `tests/test_hmm.py`.
- Data can be generated from a known 2-state HMM and the fitted parameters
  compared against the generators.

The cointegration tests are *tables*. The Engle-Granger and Johansen
statistics are easy to compute; what makes them valid inference is the
MacKinnon and Osterwald-Lenum critical-value tables, which were produced by
large Monte Carlo studies and cannot be re-derived. Transcribing published
tables is not learning — it is a data-entry exercise with a silent-wrong-answer
failure mode, and no test would catch a typo in row 3.

So the split follows what can be verified:

| Component | Source | Why |
|-----------|--------|-----|
| Forward-backward, Baum-Welch, Viterbi | from scratch | EM monotonicity is a complete correctness proof |
| Regime labelling, adaptive tilts, spread signal | from scratch | domain logic, the part worth owning |
| ADF / Engle-Granger / Johansen statistics | statsmodels | critical values are Monte Carlo tables, not derivable |

This is the same principle as ADR-004 (Ledoit-Wolf from scratch, because the
shrinkage intensity is a derivation) and ADR-005 (SLSQP borrowed, because a QP
active-set solver is engineering rather than finance).

## Consequences

**Accepted:**

- `statsmodels` is added as a dependency — a heavier one than the rest of the
  stack, though scipy and pandas were already required by it.
- The from-scratch HMM is slower than `hmmlearn`'s Cython implementation.
  Irrelevant at T≈600 and K=3 (fit is well under a second); it would matter
  for intraday data.
- `hmmlearn` is dropped from the tech stack entirely.

**Numerical note.** The HMM fits with a covariance regulariser `Σₖ += εI`
(ε=1e-6) so a state that captures few observations still has an invertible
covariance. This technically breaks the EM monotonicity guarantee — dips of
order 1e-3 on a log-likelihood of ~5000. Rather than weaken the test to
accommodate it, `test_em_monotonic` runs with ε=0 and asserts strict
monotonicity, so the pure implementation stays verified while production
fitting keeps the stabiliser. The knob is exposed as a constructor argument
rather than hidden.

**Validation.** The from-scratch model, given no event calendar, assigned its
crisis state to exactly three contiguous episodes: the August 2024 yen
carry-trade unwind, the April 2025 tariff shock, and a January 2026 episode.
Recovering known market stress from the return series alone is stronger
evidence of correctness than any unit test.

## Alternatives considered

**`hmmlearn` for the HMM.** Faster and battle-tested, but the whole point of
Phase 2 is understanding regime models, and an EM implementation is
self-verifying in a way most numerical code is not — the monotonicity property
means a from-scratch build carries its own proof. Rejected for the same reason
Ledoit-Wolf was built by hand in ADR-004.

**Hand-rolling the ADF and Johansen tests.** Would require transcribing
published critical-value tables. Rejected: high risk of a silent transcription
error, no meaningful learning, and no test that would catch it.

**Seeding the HMM so states map to regime labels directly.** Would let the
model emit named regimes rather than requiring a post-fit ranking step.
Rejected because initialising EM near hand-chosen "crisis" and "calm" means
biases the fit toward the answer you expect. States are instead ranked by
fitted volatility *after* convergence, so likelihood maximisation stays
unconstrained by prior belief and the labels remain a separate, auditable
mapping (`state_to_regime` on `RegimeResult`).
