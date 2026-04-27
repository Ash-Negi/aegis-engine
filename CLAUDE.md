# Aegis Engine

Multi-week build of a portfolio math/optimization engine. Each phase adds a layer: data pipeline → covariance estimation → optimization → tactical/signals. Auditable by design — modules are testable functions, math is explicit, no magic numbers buried in logic. Every tunable lives in `math-engine/config.py`.

## Asset universe

Four ETFs, each chosen to hedge a different failure mode:

| Ticker | Name | Role |
|--------|------|------|
| QQQ | Invesco QQQ Trust | Equity risk premium / tech beta |
| GLDM | SPDR Gold MiniShares | Inflation / monetary debasement hedge |
| FBTC | Fidelity Wise Origin Bitcoin Fund | Stochastic noise / institutional capture hedge |
| VXUS | Vanguard Total International Stock ETF | Ex-US (developed + emerging) diversification |

## Phase 1 roadmap

- **Week 1 (done)** — `math-engine/data/pipeline.py`: fetch via yfinance, clean, log returns, descriptive stats, correlations. Entry point `math-engine/main.py`.
- **Week 2** — `math-engine/covariance/`: sample, EWMA, Ledoit-Wolf shrinkage.
- **Week 3+** — `math-engine/optimizer/`, `math-engine/signals/`.

## Sample landmines (Week 1 review, 2026-04-16)

> **Historical note:** The findings below were measured with **VT** in the universe. As of 2026-04-27, VT was replaced by VXUS to reduce the QQQ overlap that drove finding (2). See `docs/adr/003-asset-universe-vxus.md`. Numbers in this section are preserved as the historical Week 1 record; Week 2 will produce a fresh baseline on the new universe.


The current common period is **494 trading days** (2024-01-11 → 2025-12-30), bounded by FBTC inception. Three properties of this sample that *will* affect downstream work:

1. **April 2025 tariff week (Apr 3–10) dominates the second moment.** Excluding those 6 days collapses VT excess kurtosis from +16.63 → +1.32 and QQQ from +12.37 → +1.20. Sample covariance will overweight that week — a perfect natural demo of why Ledoit-Wolf shrinkage matters in Week 2. Don't be surprised when sample-cov vs. shrinkage produce very different optimal weights.

2. **QQQ-VT correlation = 0.92** — near-collinear. Σ is full-rank but ill-conditioned along that axis; naive `np.linalg.inv(Σ)` will give brittle weights. Use regularization or pseudo-inverse in any optimizer.

3. **All four assets had large positive returns in-sample** (GLDM +39%, FBTC +32%, QQQ +22%, VT +19%). Vanilla MVO using historical means as expected returns will produce a gold-heavy levered portfolio — classic in-sample overfit. Be explicit about expected-return shrinkage when building the optimizer.

## Working preferences

For data-correctness questions ("does this look right?") on this project, run actual diagnostic code (load data, compute stats, cross-check across assets, identify outliers by date) rather than eyeballing printed output. This is foundational infrastructure — "looks plausible" is not a strong enough verdict, and small data bugs propagate to every downstream module.

## Environment

Use `/Users/level_up/Documents/QuantDev/aegis-engine/.venv/bin/python` for all Python/pytest commands. The venv has pytest, numpy, pandas, yfinance installed. `python3` on PATH is a bare Homebrew install without these.

## Week 2 target (covariance estimation)
Module: `math-engine/covariance/`
- Sample covariance (baseline)
- EWMA (λ=0.94, RiskMetrics standard)
- Ledoit-Wolf shrinkage (primary estimator)
- Log condition number before and after shrinkage
- Eigendecomposition — how many real risk factors does the 4×4 matrix have?