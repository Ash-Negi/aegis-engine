# Aegis Engine

A distributed quantitative execution system for multi-asset portfolio management with macro regime awareness and tactical position scaling.

![Phase 1 Week 1 — Asset Universe Overview](docs/images/phase1_week1_overview.png)
*Latest snapshot: normalized prices, return distributions, drawdown curves, and rolling vol for the four-asset universe (574-day common period, 2024-01-11 → 2026-04-27).*

## Architecture

Aegis operates on three layers:

**Macro Regime Layer** — Determines strategic asset allocation based on the current macroeconomic regime (growth/inflation dynamics). Consumes external signals and validates against an internal Hidden Markov Model for regime classification.

**Tactical Math Engine** — Computes optimal portfolio weights using mean-variance optimization with robust covariance estimation. Trades around core positions within regime-defined risk budgets using mean-reversion signals and adaptive rebalancing bands.

**Execution Engine** — Manages order routing, position tracking, and state management. Communicates with the math engine over Redis pub/sub and maintains an audit ledger in PostgreSQL.

## What this project demonstrates

- **Quantitative finance** — covariance estimation (sample, EWMA, Ledoit-Wolf shrinkage), mean-variance optimization, regime modeling with HMMs, transaction-cost-aware backtesting.
- **Systems engineering** — distributed messaging over Redis pub/sub, audit ledgers in PostgreSQL, fault-tolerant execution, containerized deployment.
- **Software craftsmanship** — Architecture Decision Records, change retrospectives, contract-based test invariants, type-safe configuration with no magic numbers.

## Engineering practices

This project is built to be auditable by design. Every nontrivial decision and meaningful change is recorded.

- **[Architecture Decision Records](docs/adr/)** — every nontrivial design choice documented with the alternatives considered and tradeoffs accepted. See [ADR-003](docs/adr/003-asset-universe-vxus.md) for an example.
- **[Change retrospectives](docs/retros/)** — debriefs after meaningful changes record what was expected, what actually happened, and what carries forward. See [Retro-001](docs/retros/001-vxus-swap.md) for an example.
- **Contract-based tests** — 20 invariants protecting downstream modules (column order, index alignment, FBTC pre-inception NaN handling, mathematical identities between return calculations). See [ADR-002](docs/adr/002-test-contracts-not-calibration.md) for the testing philosophy.
- **Single source of truth** — every tunable parameter (decay factors, lookback windows, transaction costs, asset universe) lives in `math-engine/config.py`. No magic numbers buried in module logic.

## Roadmap

A five-phase build. Each phase produces something working and demonstrable — no phase depends on completing all prior phases perfectly.

### Phase 1 — Math Engine *(in progress)*

*Proves: can handle real financial data and translate textbook math (covariance estimation, Lagrange multipliers, efficient frontier) into production-quality code with proper testing and audit trails.*

- [x] **Week 1** — Data pipeline: fetch via yfinance, clean, log returns, descriptive statistics, correlations
- [ ] **Week 2** — Covariance estimation: sample, EWMA (λ=0.94 RiskMetrics), Ledoit-Wolf shrinkage, condition number diagnostics, eigendecomposition
- [ ] **Week 3** — Mean-variance optimizer with expected-return shrinkage; efficient frontier construction; constrained optimization (long-only, sector caps)
- [ ] **Week 4** — Backtest harness with transaction costs, slippage modeling, rebalancing bands, performance attribution

### Phase 2 — Signal Development *(planned)*

*Proves: understands that markets are non-stationary and can build adaptive systems instead of static optimizers. Distinguishes statistical arbitrage thinking from passive rebalancing.*

- Hidden Markov Model regime detection (low-vol trending, high-vol mean-reverting, crisis)
- Adaptive weight engine that responds to regime classifications — risk-on tilts toward equity, risk-off tilts toward gold
- Cointegration testing (Engle-Granger, Johansen) on asset pairs; spread-based rebalance signals when deviations exceed 2σ

### Phase 3 — Execution Layer *(planned)*

*Proves: can build a polyglot async system with strong contracts between services, handle real-world execution complexity (partial fills, network failures, idempotency), and reason about audit and reconciliation.*

- Python math engine publishes target weight vectors to Redis (timestamp, confidence, regime, expected turnover)
- Java 21 / Spring Boot execution engine with virtual-thread message handling
- Alpaca paper trading integration with full order state machine (PENDING → SENT → PARTIAL_FILL → FILLED / REJECTED)
- PostgreSQL audit ledger for every order, fill, and portfolio snapshot

### Phase 4 — ML Integration *(planned)*

*Proves: can apply ML to finance without falling into the in-sample overfitting trap. Demonstrates feature-engineering judgment and an honest take on ML's role (augmenting, not replacing, well-grounded baselines).*

- Macro feature engineering: VIX, yield-curve slope (10Y–2Y), DXY, oil, credit spreads
- FinBERT sentiment pipeline on Fed minutes, FOMC statements, and financial news
- Neural regime classifier (LSTM / Transformer) ensembled with the HMM baseline; walk-forward validation throughout
- Local LLM (quantized Mistral / Llama) for structured signal extraction from macro reports — operational edge over per-token cloud inference

### Phase 5 — Production Hardening *(planned)*

*Proves: thinks about what happens when the system runs unsupervised — failure modes, observability, safety rails. The difference between a project and a piece of infrastructure.*

- Docker Compose for the full stack: Python + Java + Redis + Postgres, one command
- Health monitoring dashboard (Grafana / Prometheus): positions, P&L, regime, signal freshness
- Circuit breakers: max-drawdown halt, signal-staleness halt, position-concentration halt
- AWS EC2 deployment of the full stack

## Asset Universe

Four ETFs, each chosen to hedge a different failure mode. See [ADR-003](docs/adr/003-asset-universe-vxus.md) for the rationale behind the current selection.

| Ticker | Name | Role |
|--------|------|------|
| QQQ | Invesco QQQ Trust | Tech beta / equity risk premium / US|
| GLDM | SPDR Gold MiniShares | Inflation / monetary debasement hedge / Defensive |
| FBTC | Fidelity Wise Origin Bitcoin Fund | financial repression hedge |
| VXUS | Vanguard Total International Stock ETF | Ex-US (developed + emerging) diversification |

## Tech Stack

- **Math Engine:** Python 3.12+ (NumPy, SciPy, Pandas, yfinance, hmmlearn, scikit-learn)
- **ML / NLP:** FinBERT, quantized local LLMs (Mistral / Llama) — *planned*
- **Execution Engine:** Java 21+ (Spring Boot, Virtual Threads), Alpaca API — *planned*
- **Messaging:** Redis pub/sub — *planned*
- **Ledger:** PostgreSQL — *planned*
- **Observability:** Grafana / Prometheus — *planned*
- **Deployment:** Docker Compose → AWS EC2 — *planned*

## Quick Start

```bash
cd math-engine
pip install -r requirements.txt
python main.py
```

## Running Tests

```bash
cd math-engine
pytest tests/ -v
```

## Repository Layout

```
aegis-engine/
├── math-engine/             # Phase 1: Python math engine
│   ├── config.py            # All tunable parameters (single source of truth)
│   ├── main.py              # Entry point — runs the data pipeline
│   ├── data/pipeline.py     # Fetch, clean, transform daily prices
│   └── tests/               # Contract-based test suite
└── docs/
    ├── adr/                 # Architecture Decision Records
    ├── retros/              # Change retrospectives
    └── images/              # Charts and visualizations
```
