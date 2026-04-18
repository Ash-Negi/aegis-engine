# ADR-002: Pipeline tests validate contracts, not market calibration

## Status

Accepted

## Context

Writing tests for a data pipeline that produces market statistics forces a choice about how tight the assertions should be. Two philosophies compete:

**Tight (calibration)**. Assert that each asset's stats fall within narrow, market-informed bands. For example, "QQQ annualized vol must be between 15% and 25%." These tests act as an alarm when market reality drifts outside expected ranges.

**Loose (contracts)**. Assert only that values are in the universe of not-broken numbers. For example, "annualized vol must be positive and below 200%." These tests catch calculation bugs (wrong annualization factor, wrong units, NaN leaking in) but stay silent about what a "normal" market looks like.

The same choice applies to Sharpe ratios, correlations, skew/kurtosis, and any other market-derived statistic the pipeline reports. This ADR picks one philosophy for the whole test suite so future contributors don't have to relitigate it every time they add a test.

## Decision

Pipeline tests validate structural and mathematical contracts only. They do **not** assert calibration bands.

Concretely:

- `test_volatility_reasonable_range` asserts `1 < vol < 200`, not "QQQ ≈ 18%".
- `test_sharpe_reasonable_range` asserts `-5 < sharpe < 10`, not "expected Sharpe ≈ 0.5".
- **Contract** tests — index alignment, column order matches `TICKERS`, FBTC NaN pre-inception, first row of returns is NaN, `ln(1 + total_return) == ann_return * n / 252` — ARE asserted with tight tolerances, because the pipeline must always satisfy them regardless of market regime.

Calibration checks ("does today's QQQ vol look normal?") belong in a monitoring layer — dashboards, alerts on live stats, or a separate regime-tracking test suite with its own failure semantics. They do not belong next to "is the data structurally correct?" tests.

## Consequences

### Benefits

- **Tests don't flap across market regimes.** If 2026 has a vol regime where QQQ realizes 35%, tight per-asset bands would fail with no pipeline bug present. Loose bounds stay green and the tests keep catching real bugs.
- **Clear test-layer semantics.** A failing test means a pipeline bug, not a "market moved" signal. Debugging starts from "what broke in the code?" rather than "did the world change?"
- **Calibration lives in its right place.** The appropriate tool for "is this number normal?" is a dashboard or regime monitor, not pytest. Keeping concerns separate means each tool does one thing well.

### Tradeoffs

- **Miscalibration can slip through range checks.** If the annualization factor were wrong (12 instead of 252) or daily returns were accidentally doubled, vol might still land inside `(1, 200)`. This ADR accepts that risk for range checks and mitigates it by requiring strong **contract** coverage instead — mathematical identities (total_return vs ann_return), structural invariants (index alignment), and precise spot-checks on return formulas. Contract tests are the real safety net; range checks are a last-ditch sanity gate.
- **Loose tests demand strong contract tests.** If contract coverage is ever allowed to thin out, the loose range checks become the only defense and this ADR's risk/benefit balance flips. Any future change that removes a contract test should re-evaluate whether the remaining coverage is still sufficient.
- **Team convention only, not enforced.** A future contributor may reflexively "tighten" a range check without reading this ADR. Mitigation: the `Review Log` at the top of `tests/test_pipeline.py` points to this decision, and the docstrings of the relevant tests explain the loose-on-purpose intent.