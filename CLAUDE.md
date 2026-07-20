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
- **Week 2 (done)** — `math-engine/covariance/`: sample, EWMA, Ledoit-Wolf shrinkage, condition-number + eigenstructure diagnostics. Demo `python -m covariance.report`. Design recorded in `docs/adr/004-covariance-estimator-design.md`; math in `docs/math.md`.
- **Week 3 (done)** — `math-engine/optimizer/`: expected-return shrinkage (Bayes-Stein), closed-form MVO (GMV/tangency/frontier via Lagrange), constrained long-only + sector caps (SLSQP). Demo `python -m optimizer.report`. Design in `docs/adr/005-optimizer-closed-form-and-slsqp.md`.
- **Week 4 (done)** — `math-engine/backtest/`: band-rebalancing engine with transaction costs + slippage, performance metrics (CAGR/Sharpe/maxDD/Calmar), per-asset attribution. Demo `python -m backtest.report`. **Phase 1 (Math Engine) complete.**

## Phase 2 roadmap

- **Phase 2 (done)** — `math-engine/signals/`: from-scratch Gaussian HMM (Baum-Welch EM in log space, Viterbi), regime labeling by fitted volatility rank, regime-conditioned multiplicative weight tilts, cointegration (Engle-Granger + Johansen via statsmodels) with hysteresis spread signals. Demo `python -m signals.report`. Design in `docs/adr/006-hmm-from-scratch-stats-tests-delegated.md`.

## Phase 3 roadmap

- **Phase 3 (done, broker stubbed)** — `math-engine/publisher/` + `execution-engine/`: signal contract, Redis publisher, Java 21 / Spring Boot consumer with an order state machine, Flyway-migrated Postgres ledger, docker-compose. Demo `python -m publisher.report` or `docker compose up --build`. Design in `docs/adr/007-publish-target-weights-not-orders.md`.
- **Phase 4+** — ML feature engineering. Not started; user explicitly scoped work to stop before Phase 4.

## Phase 3 conventions

- **The wire carries target weights, not orders.** Desired state is idempotent; imperative commands are not. Every recovery path (startup replay from `aegis:signals:latest`, signal-id dedupe, the no-trade band) depends on that. Don't add order-level fields to the signal contract.
- **`contract.py` and `TargetWeightSignal.java` are one contract in two files.** Changing either means changing both and bumping `schema_version`. The Java test fixture in `TargetWeightSignalTest` is a *verbatim* copy of real Python output — regenerate it with `python -m publisher.report`, never hand-edit it.
- **Money and share counts are `BigDecimal`/`NUMERIC`, never `double`.** Accumulating fills as doubles leaves an order a few ulps short of target so it never reaches FILLED.
- **Only `OrderStateMachine` may mutate an `Order`** — every setter is package-private so all transitions go through the legality check.
- **Java tests run on H2 in PostgreSQL mode with the real Flyway migration.** Use standard SQL spellings (`TIMESTAMP WITH TIME ZONE`, not `TIMESTAMPTZ`) so the migration stays portable across both.
- Docker was not available in the dev environment, so `docker compose up` has **not** been executed end-to-end. The Dockerfiles and compose file are written but unverified; the Python↔Java seam is covered by the contract test instead.

## Phase 2 findings

1. **The HMM validated itself against real events.** Fit on the equal-weight return proxy with no event calendar, the crisis state landed on exactly three contiguous episodes: 2024-08-01→08 (yen carry unwind), 2025-04-03→11 (tariff shock), 2026-01-30→02-06. Treat this as the module's strongest correctness evidence.

2. **No pair in this universe is cointegrated** — best is GLDM~VXUS at EG p=0.111, and the Johansen trace test gives rank r=0. This is the *correct* answer for a universe built (ADR-003) so each asset hedges a different failure mode; assets with no shared driver should have no shared stochastic trend. The stat-arb machinery is built and tested but has nothing to trade here. Do not "fix" this by loosening α.

3. **EM monotonicity vs. conditioning.** `GaussianHMM` adds `Σₖ += εI` (ε=1e-6) so sparse states stay invertible, which breaks strict EM monotonicity by ~1e-3 on a log-likelihood of ~5000. `test_em_monotonic` runs with ε=0 to keep the guarantee testable; don't weaken that test to accommodate the regulariser.

## Sample landmines (Week 1 review, 2026-04-16)

> **Historical note:** The findings below were measured with **VT** in the universe. As of 2026-04-27, VT was replaced by VXUS to reduce the QQQ overlap that drove finding (2). See `docs/adr/003-asset-universe-vxus.md`. Numbers in this section are preserved as the historical Week 1 record; Week 2 will produce a fresh baseline on the new universe.


The current common period is **494 trading days** (2024-01-11 → 2025-12-30), bounded by FBTC inception. Three properties of this sample that *will* affect downstream work:

1. **April 2025 tariff week (Apr 3–10) dominates the second moment.** Excluding those 6 days collapses VT excess kurtosis from +16.63 → +1.32 and QQQ from +12.37 → +1.20. Sample covariance will overweight that week — a perfect natural demo of why Ledoit-Wolf shrinkage matters in Week 2. Don't be surprised when sample-cov vs. shrinkage produce very different optimal weights.

2. **QQQ-VT correlation = 0.92** — near-collinear. Σ is full-rank but ill-conditioned along that axis; naive `np.linalg.inv(Σ)` will give brittle weights. Use regularization or pseudo-inverse in any optimizer.

3. **All four assets had large positive returns in-sample** (GLDM +39%, FBTC +32%, QQQ +22%, VT +19%). Vanilla MVO using historical means as expected returns will produce a gold-heavy levered portfolio — classic in-sample overfit. Be explicit about expected-return shrinkage when building the optimizer.

## Working preferences

For data-correctness questions ("does this look right?") on this project, run actual diagnostic code (load data, compute stats, cross-check across assets, identify outliers by date) rather than eyeballing printed output. This is foundational infrastructure — "looks plausible" is not a strong enough verdict, and small data bugs propagate to every downstream module.

## Environment

Use `/Users/level_up/Documents/QuantDev/aegis-engine/.venv/bin/python` for all Python/pytest commands. The venv has pytest, numpy, pandas, scipy, statsmodels, yfinance, matplotlib, redis, fakeredis installed. `python3` on PATH is a bare Homebrew install without these.

For the Java execution engine, use `mvn test` from `execution-engine/`. JDK 25 is installed; the project targets Java 21. Maven's first run needs network access to resolve dependencies.