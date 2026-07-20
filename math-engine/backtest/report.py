"""
Aegis Engine — Phase 1, Week 4: Backtest Report
===============================================
Runs the optimizer's portfolio through the backtest harness with realistic
frictions and compares it against naive benchmarks, so the question "does the
Week 3 optimum survive costs?" gets an honest answer.

Usage (from math-engine/):
    python -m backtest.report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DataConfig, PortfolioConfig, OptimizerConfig, TICKERS, DATA_DIR
from data.pipeline import DataPipeline
from covariance import ledoit_wolf_covariance
from optimizer import estimate_expected_returns, max_sharpe_portfolio
from backtest import BacktestEngine, performance_metrics, return_attribution

RF = DataConfig().risk_free_rate
COLORS = {"QQQ": "#2196F3", "GLDM": "#FFD700", "FBTC": "#FF6B35", "VXUS": "#4CAF50"}


def _header(text):
    print(f"\n{'═' * 70}\n  {text}\n{'═' * 70}")


def _section(text):
    print(f"\n  ── {text} {'─' * max(0, 60 - len(text))}")


def _print_metrics(name, m):
    print(f"    {name:<22} CAGR {m.cagr*100:6.2f}%  vol {m.ann_volatility*100:6.2f}%  "
          f"Sharpe {m.sharpe:5.2f}  maxDD {m.max_drawdown*100:7.2f}%  "
          f"Calmar {m.calmar:5.2f}")


def backtest_report(port_config=None, opt_config=None):
    port_config = port_config or PortfolioConfig()
    opt_config = opt_config or OptimizerConfig()
    _header("AEGIS ENGINE — Phase 1, Week 4: Backtest Harness")

    pipeline = DataPipeline(DataConfig())
    dataset = pipeline.run(TICKERS, use_cache=True)
    common = pipeline.get_common_period(dataset)
    log_returns = common["log_returns"]
    returns = common["returns"]

    # Optimized target: long-only max-Sharpe on shrunk μ / Ledoit-Wolf Σ.
    cov_daily = ledoit_wolf_covariance(log_returns)
    cov = cov_daily * opt_config.trading_days_per_year
    er = estimate_expected_returns(log_returns, cov_daily, opt_config)
    optimized = max_sharpe_portfolio(er.mu, cov, RF, opt_config)
    equal = pd.Series(1 / len(TICKERS), index=TICKERS)

    print(f"\n  Period {returns.index[0].date()} → {returns.index[-1].date()}  "
          f"(T={len(returns)})   costs={port_config.total_cost_bps:.0f}bps  "
          f"band=±{port_config.rebalance_band_pct:.0f}%")
    print(f"  Optimized target: " + "  ".join(f"{t} {optimized[t]*100:.0f}%" for t in TICKERS))

    # Three runs: optimized (rebalanced), equal-weight (rebalanced), optimized buy&hold.
    engine = BacktestEngine(port_config)
    res_opt = engine.run(returns, optimized)
    res_eq = engine.run(returns, equal)
    buyhold_cfg = PortfolioConfig(transaction_cost_bps=port_config.transaction_cost_bps,
                                  slippage_bps=port_config.slippage_bps,
                                  rebalance_band_pct=1e6)
    res_bh = BacktestEngine(buyhold_cfg).run(returns, optimized)

    _section("Performance (net of costs)")
    print()
    _print_metrics("Optimized (banded)", performance_metrics(res_opt.net_returns, risk_free_rate=RF))
    _print_metrics("Equal-weight (banded)", performance_metrics(res_eq.net_returns, risk_free_rate=RF))
    _print_metrics("Optimized (buy&hold)", performance_metrics(res_bh.net_returns, risk_free_rate=RF))

    _section("Trading activity & cost drag (optimized, banded)")
    attr = return_attribution(res_opt, returns)
    print(f"\n    Rebalances: {attr.n_rebalances}   total one-way turnover: "
          f"{attr.total_turnover*100:.1f}%   total cost drag: {attr.total_cost*100:.2f}%")
    print(f"    Gross (arith) {attr.gross_return_arith*100:.2f}%  −  cost "
          f"{attr.total_cost*100:.2f}%  =  net {attr.net_return_arith*100:.2f}%")

    _section("Return attribution (per-asset contribution to gross)")
    print()
    for t in TICKERS:
        print(f"    {t:<6} {attr.contribution[t]*100:+7.2f}%   (target weight {optimized[t]*100:4.0f}%)")

    _section("Generating backtest plot")
    _plot(res_opt, res_eq, res_bh, attr, optimized)

    _header("Week 4 Complete — Phase 1 (Math Engine) done")
    print("\n  Next: Phase 2 — HMM regime detection and adaptive weights.\n")


def _plot(res_opt, res_eq, res_bh, attr, optimized):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Aegis Engine — Phase 1 Week 4: Backtest", fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: equity curves
    ax = axes[0, 0]
    ax.plot(res_opt.equity_curve, color="#2196F3", lw=2, label="optimized (banded)")
    ax.plot(res_eq.equity_curve, color="#888", lw=1.5, label="equal-weight")
    ax.plot(res_bh.equity_curve, color="#4CAF50", lw=1.5, ls="--", label="optimized (buy&hold)")
    ax.set_title("Equity Curve (net of costs)", fontweight="bold")
    ax.set_ylabel("Portfolio value"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 2: drawdown of optimized
    ax = axes[0, 1]
    eq = res_opt.equity_curve
    dd = (eq / eq.cummax() - 1) * 100
    ax.fill_between(dd.index, dd.to_numpy(), 0, color="#FF6B35", alpha=0.5)
    ax.set_title("Drawdown — Optimized (banded)", fontweight="bold")
    ax.set_ylabel("Drawdown (%)"); ax.grid(True, alpha=0.3)

    # Panel 3: weight drift over time
    ax = axes[1, 0]
    W = res_opt.weights
    bottom = np.zeros(len(W))
    for t in TICKERS:
        ax.fill_between(W.index, bottom, bottom + W[t].to_numpy() * 100,
                        color=COLORS[t], label=t, alpha=0.85)
        bottom += W[t].to_numpy() * 100
    ax.set_title("Weight Drift & Rebalancing (optimized)", fontweight="bold")
    ax.set_ylabel("Weight (%)"); ax.set_ylim(0, 100)
    ax.legend(fontsize=8, ncol=4, loc="upper center"); ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: attribution
    ax = axes[1, 1]
    contrib = [attr.contribution[t] * 100 for t in TICKERS]
    bars = ax.bar(TICKERS, contrib, color=[COLORS[t] for t in TICKERS], alpha=0.85)
    ax.bar(["cost"], [-attr.total_cost * 100], color="#c00", alpha=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Return Attribution (gross contribution & cost drag)", fontweight="bold")
    ax.set_ylabel("Contribution to return (%)"); ax.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars, contrib):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)

    plt.tight_layout()
    out = DATA_DIR / "phase1_week4_backtest.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved: {out}")


if __name__ == "__main__":
    backtest_report()
