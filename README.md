# Aegis Engine

A distributed quantitative execution system for multi-asset portfolio management with macro regime awareness and tactical position scaling.

## Architecture

Aegis operates on three layers:

**Macro Regime Layer** — Determines strategic asset allocation based on the current macroeconomic regime (growth/inflation dynamics). Consumes external signals and validates against an internal Hidden Markov Model for regime classification.

**Tactical Math Engine** — Computes optimal portfolio weights using mean-variance optimization with robust covariance estimation. Trades around core positions within regime-defined risk budgets using mean-reversion signals and adaptive rebalancing bands.

**Execution Engine** — Manages order routing, position tracking, and state management. Communicates with the math engine over Redis pub/sub and maintains an audit ledger in PostgreSQL.

## Current Status

**Phase 1: Math Engine** — Building the quantitative foundation.

- [x] Data pipeline: fetch, clean, and transform daily prices
- [ ] Covariance estimation (sample, EWMA, Ledoit-Wolf shrinkage)
- [ ] Mean-variance optimizer and efficient frontier
- [ ] Backtest harness with transaction costs

## Asset Universe

| Ticker | Name | Role |
|--------|------|------|
| QQQ | Invesco QQQ Trust | Tech Beta |
| GLDM | SPDR Gold MiniShares | Inflation Hedge |
| FBTC | Fidelity Wise Origin Bitcoin Fund | Stochastic Noise |
| VT | Vanguard Total World Stock ETF | Global Macro |

## Tech Stack

- **Math Engine:** Python 3.12+ (NumPy, SciPy, Pandas)
- **Execution Engine:** Java 21+ (Spring Boot, Virtual Threads) — *planned*
- **Messaging:** Redis pub/sub — *planned*
- **Ledger:** PostgreSQL — *planned*
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