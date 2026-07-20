"""
Aegis Engine — Phase 1, Week 3: Optimizer Report
================================================
Runs the full Week 3 stack — expected-return shrinkage, closed-form frontier,
constrained (long-only + sector-capped) frontier — and lays out the result:
the efficient frontier, the key portfolios' weights, and a direct before/after
of what shrinkage does to the naive gold-heavy solution.

Usage (from math-engine/):
    python -m optimizer.report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DataConfig, OptimizerConfig, TICKERS, DATA_DIR
from data.pipeline import DataPipeline
from covariance import ledoit_wolf_covariance
from optimizer import (
    estimate_expected_returns,
    global_minimum_variance,
    tangency_portfolio,
    efficient_frontier,
    min_variance_portfolio,
    max_sharpe_portfolio,
    efficient_frontier_constrained,
    portfolio_return,
    portfolio_volatility,
)

RF = DataConfig().risk_free_rate  # single source of truth for the risk-free rate
COLORS = {"QQQ": "#2196F3", "GLDM": "#FFD700", "FBTC": "#FF6B35", "VXUS": "#4CAF50"}


def _header(text):
    print(f"\n{'═' * 70}\n  {text}\n{'═' * 70}")


def _section(text):
    print(f"\n  ── {text} {'─' * max(0, 60 - len(text))}")


def _show_weights(name, w, mu, cov):
    r = portfolio_return(w, mu) * 100
    v = portfolio_volatility(w, cov) * 100
    sharpe = (portfolio_return(w, mu) - RF) / portfolio_volatility(w, cov)
    cells = "  ".join(f"{t}:{w[t] * 100:+6.1f}%" for t in TICKERS)
    print(f"    {name:<22} {cells}   |  ret {r:5.1f}%  vol {v:5.1f}%  SR {sharpe:4.2f}")


def optimizer_report(config: OptimizerConfig | None = None) -> None:
    config = config or OptimizerConfig()
    _header("AEGIS ENGINE — Phase 1, Week 3: Mean-Variance Optimizer")

    pipeline = DataPipeline(DataConfig())
    dataset = pipeline.run(TICKERS, use_cache=True)
    lr = pipeline.get_common_period(dataset)["log_returns"]

    cov_daily = ledoit_wolf_covariance(lr)
    cov = cov_daily * config.trading_days_per_year          # annualized Σ
    er = estimate_expected_returns(lr, cov_daily, config)   # annualized shrunk μ
    mu, raw_mu = er.mu, er.raw_mu

    print(f"\n  Common period T={len(lr)} days · Σ = Ledoit-Wolf (annualized) · "
          f"μ shrinkage = {er.method} (φ={er.shrinkage:.3f})")

    # ── Expected returns: raw vs shrunk ──────────────────────────────────
    _section("Expected returns (annualized): raw sample mean → shrunk")
    print()
    for t in TICKERS:
        print(f"    {t:<6} raw {raw_mu[t] * 100:6.2f}%   →   shrunk {mu[t] * 100:6.2f}%")
    print(f"    target (μ_min) = {er.target * 100:.2f}%   "
          f"— means pulled together, dispersion {raw_mu.std()*100:.2f}% → {mu.std()*100:.2f}%")

    # ── Key portfolios ───────────────────────────────────────────────────
    _section("Key portfolios (SR = Sharpe, rf = {:.0%})".format(RF))
    print()
    gmv = global_minimum_variance(cov)
    tan_raw = tangency_portfolio(raw_mu, cov, RF)
    tan_shr = tangency_portfolio(mu, cov, RF)
    mv_con = min_variance_portfolio(cov, config)
    ms_con = max_sharpe_portfolio(mu, cov, RF, config)

    _show_weights("GMV (unconstrained)", gmv, mu, cov)
    _show_weights("Tangency raw-μ", tan_raw, raw_mu, cov)
    _show_weights("Tangency shrunk-μ", tan_shr, mu, cov)
    _show_weights("Min-var (long-only)", mv_con, mu, cov)
    _show_weights("Max-Sharpe (constr.)", ms_con, mu, cov)
    print(f"\n    Note: raw-μ tangency concentrates in gold and shorts; shrinkage +")
    print(f"    long-only/sector caps produce a diversified, tradeable portfolio.")

    # ── Frontiers ────────────────────────────────────────────────────────
    _section("Tracing efficient frontiers (unconstrained + long-only)")
    fr_u = efficient_frontier(mu, cov, config.frontier_points, risk_free_rate=RF)
    fr_c = efficient_frontier_constrained(mu, cov, config, risk_free_rate=RF)
    print(f"    unconstrained: {len(fr_u.returns)} pts, "
          f"vol {fr_u.volatilities.min()*100:.1f}–{fr_u.volatilities.max()*100:.1f}%")
    print(f"    long-only:     {len(fr_c.returns)} pts, "
          f"vol {fr_c.volatilities.min()*100:.1f}–{fr_c.volatilities.max()*100:.1f}%")

    _section("Generating frontier plot")
    _plot(mu, cov, fr_u, fr_c, gmv, tan_shr, ms_con, raw_mu, er)

    _header("Week 3 Complete")
    print("\n  Next: Week 4 — backtest harness (transaction costs, slippage,")
    print("  rebalancing bands, performance attribution).\n")


def _plot(mu, cov, fr_u, fr_c, gmv, tangency, ms_con, raw_mu, er):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Aegis Engine — Phase 1 Week 3: Mean-Variance Optimization",
                 fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: efficient frontier + CML + assets
    ax = axes[0, 0]
    ax.plot(fr_u.volatilities * 100, fr_u.returns * 100, color="#888",
            lw=2, label="unconstrained frontier")
    ax.plot(fr_c.volatilities * 100, fr_c.returns * 100, color="#2196F3",
            lw=2, label="long-only frontier")
    # individual assets
    for t in TICKERS:
        ax.scatter(np.sqrt(cov.loc[t, t]) * 100, mu[t] * 100, s=80,
                   color=COLORS[t], edgecolor="k", zorder=5)
        ax.annotate(t, (np.sqrt(cov.loc[t, t]) * 100, mu[t] * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    # tangency + capital market line
    tv = portfolio_volatility(tangency, cov) * 100
    tr = portfolio_return(tangency, mu) * 100
    ax.scatter(tv, tr, marker="*", s=260, color="#FF6B35", edgecolor="k",
               zorder=6, label="tangency")
    ax.scatter(fr_u.gmv_vol * 100, fr_u.gmv_return * 100, marker="D", s=90,
               color="#4CAF50", edgecolor="k", zorder=6, label="GMV")
    xs = np.linspace(0, fr_u.volatilities.max() * 100, 50)
    ax.plot(xs, RF * 100 + (tr - RF * 100) / tv * xs, "--", color="#FF6B35",
            lw=1.2, alpha=0.8, label="capital market line")
    ax.set_title("Efficient Frontier", fontweight="bold")
    ax.set_xlabel("Volatility (%)"); ax.set_ylabel("Expected return (%)")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)

    # Panel 2: key-portfolio weights (grouped bars)
    ax = axes[0, 1]
    ports = {"GMV": gmv, "tangency": tangency, "max-Sharpe\n(long-only)": ms_con}
    x = np.arange(len(TICKERS)); width = 0.26
    for k, (name, w) in enumerate(ports.items()):
        ax.bar(x + (k - 1) * width, [w[t] * 100 for t in TICKERS], width, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, TICKERS)
    ax.set_title("Portfolio Weights", fontweight="bold")
    ax.set_ylabel("Weight (%)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: shrinkage of the mean vector
    ax = axes[1, 0]
    x = np.arange(len(TICKERS)); width = 0.38
    ax.bar(x - width / 2, [raw_mu[t] * 100 for t in TICKERS], width,
           color="#bbb", label="raw sample mean")
    ax.bar(x + width / 2, [er.mu[t] * 100 for t in TICKERS], width,
           color="#2196F3", label=f"Bayes-Stein (φ={er.shrinkage:.2f})")
    ax.axhline(er.target * 100, color="#FF6B35", ls="--", lw=1.2,
               label=f"target μ_min {er.target*100:.1f}%")
    ax.set_xticks(x, TICKERS)
    ax.set_title("Expected-Return Shrinkage", fontweight="bold")
    ax.set_ylabel("Annualized return (%)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: constrained frontier weight composition (stacked)
    ax = axes[1, 1]
    W = fr_c.weights
    bottom = np.zeros(len(W))
    for t in TICKERS:
        ax.bar(range(len(W)), W[t].to_numpy() * 100, bottom=bottom,
               color=COLORS[t], label=t, width=1.0)
        bottom += W[t].to_numpy() * 100
    ax.set_title("Long-Only Frontier Composition (low→high return)", fontweight="bold")
    ax.set_xlabel("frontier point"); ax.set_ylabel("Weight (%)")
    ax.set_xlim(-0.5, len(W) - 0.5); ax.legend(fontsize=8, ncol=4, loc="upper center")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = DATA_DIR / "phase1_week3_frontier.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved: {out}")


if __name__ == "__main__":
    optimizer_report()
