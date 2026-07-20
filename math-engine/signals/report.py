"""
Aegis Engine — Phase 2: Signal Development Report
=================================================
Demonstrates the three Phase 2 layers on the real universe:

    1. Regime detection   — fit a 3-state Gaussian HMM to market returns
    2. Adaptive weights   — tilt the Phase 1 optimum by the detected regime
    3. Cointegration      — find tradeable pairs and their spread signals

Usage (from math-engine/):
    python -m signals.report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    DataConfig, OptimizerConfig, SignalsConfig, TICKERS, DATA_DIR,
)
from data.pipeline import DataPipeline
from covariance import ledoit_wolf_covariance
from optimizer import estimate_expected_returns, max_sharpe_portfolio
from signals import (
    RegimeDetector, AdaptiveWeightEngine, REGIME_ORDER,
    test_all_pairs, johansen_rank, engle_granger, spread_signal,
)

RF = DataConfig().risk_free_rate
COLORS = {"QQQ": "#2196F3", "GLDM": "#FFD700", "FBTC": "#FF6B35", "VXUS": "#4CAF50"}
REGIME_COLORS = {
    "low_vol_trending": "#4CAF50",
    "high_vol_meanrev": "#FFA726",
    "crisis": "#E53935",
}


def _header(text):
    print(f"\n{'═' * 70}\n  {text}\n{'═' * 70}")


def _section(text):
    print(f"\n  ── {text} {'─' * max(0, 60 - len(text))}")


def _episodes(labels, target):
    """Contiguous runs of `target` in a label series → (start, end, length)."""
    hit = labels == target
    runs, start = [], None
    for date, flag in hit.items():
        if flag and start is None:
            start, prev = date, date
        elif flag:
            prev = date
        elif start is not None:
            runs.append((start, prev, int(hit.loc[start:prev].sum())))
            start = None
    if start is not None:
        runs.append((start, prev, int(hit.loc[start:prev].sum())))
    return runs


def signals_report(opt_config=None, sig_config=None):
    opt_config = opt_config or OptimizerConfig()
    sig_config = sig_config or SignalsConfig()
    _header("AEGIS ENGINE — Phase 2: Signal Development")

    pipeline = DataPipeline(DataConfig())
    dataset = pipeline.run(TICKERS, use_cache=True)
    common = pipeline.get_common_period(dataset)
    log_returns = common["log_returns"]
    returns = common["returns"]
    log_prices = np.log(common["prices"])

    print(f"\n  Period {returns.index[0].date()} → {returns.index[-1].date()}  "
          f"(T={len(returns)})   regimes={sig_config.n_regimes}")

    # ── 1. Regime detection ──────────────────────────────────────────────
    # Fit on the equal-weight portfolio return — a market proxy that does not
    # presume the optimizer's answer, so the regimes are a property of the
    # universe rather than of one particular portfolio.
    market = returns.mean(axis=1)
    regimes = RegimeDetector(n_regimes=sig_config.n_regimes).fit(market)

    _section("Regime detection (3-state Gaussian HMM on equal-weight proxy)")
    print()
    print(f"    {'regime':<20} {'ann ret':>9} {'ann vol':>9} {'days':>6} {'freq':>7}")
    for regime in REGIME_ORDER:
        r = regimes.regime_stats.loc[regime]
        print(f"    {regime:<20} {r['ann_return_pct']:>8.1f}% {r['ann_vol_pct']:>8.1f}% "
              f"{int(r['days']):>6} {r['frequency_pct']:>6.1f}%")
    print(f"\n    Current regime: {regimes.current}  "
          f"(confidence {regimes.proba[regimes.current].iloc[-1]*100:.0f}%)")

    # The crisis regime should coincide with known stress — a sanity check
    # that the HMM found real structure rather than fitting noise. Crisis days
    # are scattered, not one block, so report each contiguous episode.
    for start, end, n in _episodes(regimes.labels, "crisis"):
        span = f"{start.date()}" if n == 1 else f"{start.date()} → {end.date()}"
        print(f"    Crisis episode: {span}  ({n}d)")

    # ── 2. Adaptive weights ──────────────────────────────────────────────
    cov_daily = ledoit_wolf_covariance(log_returns)
    cov = cov_daily * opt_config.trading_days_per_year
    er = estimate_expected_returns(log_returns, cov_daily, opt_config)
    base = max_sharpe_portfolio(er.mu, cov, RF, opt_config)

    engine = AdaptiveWeightEngine(sig_config)
    tilted = {regime: engine.tilt(base, regime) for regime in REGIME_ORDER}

    _section("Adaptive weights (regime tilts applied to the Phase 1 optimum)")
    print()
    print(f"    {'':<20} " + "".join(f"{t:>8}" for t in TICKERS) + f"{'turnover':>11}")
    print(f"    {'base (max-Sharpe)':<20} " +
          "".join(f"{base[t]*100:>7.1f}%" for t in TICKERS) + f"{'—':>11}")
    for regime in REGIME_ORDER:
        aw = tilted[regime]
        print(f"    {regime:<20} " +
              "".join(f"{aw.weights[t]*100:>7.1f}%" for t in TICKERS) +
              f"{aw.turnover*100:>10.1f}%")

    # ── 3. Cointegration ─────────────────────────────────────────────────
    pairs = test_all_pairs(log_prices, sig_config)
    johansen = johansen_rank(log_prices, sig_config)

    _section(f"Cointegration — Engle-Granger (α = {sig_config.coint_significance})")
    print()
    print(f"    {'pair':<16} {'β':>9} {'EG p':>9} {'ADF p':>9}  cointegrated")
    for _, row in pairs.head(6).iterrows():
        mark = "✓" if row["cointegrated"] else "·"
        print(f"    {row['y']}~{row['x']:<12} {row['hedge_ratio']:>9.3f} "
              f"{row['eg_pvalue']:>9.3f} {row['adf_pvalue']:>9.3f}  {mark}")
    n_coint = int(pairs["cointegrated"].sum())
    print(f"\n    {n_coint} of {len(pairs)} ordered pairs cointegrated at α="
          f"{sig_config.coint_significance}")

    _section("Cointegration — Johansen trace test (system rank)")
    print()
    for _, row in johansen.iterrows():
        mark = "reject" if row["reject"] else "accept"
        print(f"    {row['hypothesis']:<10} trace {row['trace_stat']:>9.2f}  "
              f"crit95 {row['crit_95']:>8.2f}   {mark}")
    print(f"\n    Cointegration rank r = {johansen.attrs['rank']}")

    # Spread signal on the most-cointegrated pair (whether or not it clears α —
    # the demo is about the machinery, and the p-value is reported honestly).
    best = pairs.iloc[0]
    pair = engle_granger(log_prices, best["y"], best["x"], sig_config)
    sig = spread_signal(log_prices, pair, sig_config)

    _section(f"Spread signal — {pair.y} ~ {pair.x} "
             f"(entry ±{sig_config.spread_entry_z}σ, exit ±{sig_config.spread_exit_z}σ)")
    n_trades = int((sig.position.diff().fillna(0) != 0).sum())
    in_market = float((sig.position != 0).mean())
    print(f"\n    hedge ratio β = {pair.hedge_ratio:.3f}   EG p = {pair.eg_pvalue:.3f}")
    print(f"    position changes: {n_trades}   time in market: {in_market*100:.0f}%")
    print(f"    current z-score: {sig.zscore.iloc[-1]:+.2f}  "
          f"→ position {int(sig.position.iloc[-1]):+d}")

    _section("Generating Phase 2 plot")
    _plot(market, regimes, base, tilted, sig, pair, pairs)

    _header("Phase 2 Complete — Signal Development")
    print("\n  Next: Phase 3 — execution layer (Redis, Java, Postgres audit).\n")


def _plot(market, regimes, base, tilted, sig, pair, pairs):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Aegis Engine — Phase 2: Regimes, Adaptive Weights, Cointegration",
                 fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: cumulative market return shaded by detected regime.
    ax = axes[0, 0]
    equity = (1 + market).cumprod()
    ax.plot(equity, color="#222", lw=1.5, zorder=3)
    for regime in REGIME_ORDER:
        mask = (regimes.labels == regime).reindex(equity.index, fill_value=False)
        ax.fill_between(equity.index, 0, equity.max() * 1.05,
                        where=mask.to_numpy(), color=REGIME_COLORS[regime],
                        alpha=0.25, label=regime, step="mid")
    ax.set_ylim(equity.min() * 0.97, equity.max() * 1.03)
    ax.set_title("Detected Regimes over the Equal-Weight Equity Curve", fontweight="bold")
    ax.set_ylabel("Cumulative return"); ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 2: smoothed regime probabilities — the HMM's confidence over time.
    ax = axes[0, 1]
    bottom = np.zeros(len(regimes.proba))
    for regime in REGIME_ORDER:
        vals = regimes.proba[regime].to_numpy()
        ax.fill_between(regimes.proba.index, bottom, bottom + vals,
                        color=REGIME_COLORS[regime], alpha=0.8, label=regime)
        bottom += vals
    ax.set_title("Smoothed Regime Probabilities P(regime | data)", fontweight="bold")
    ax.set_ylabel("Probability"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=3, loc="lower center"); ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: base vs regime-tilted weights.
    ax = axes[1, 0]
    labels = ["base"] + REGIME_ORDER
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for t in TICKERS:
        vals = np.array([base[t]] + [tilted[r].weights[t] for r in REGIME_ORDER]) * 100
        ax.bar(x, vals, bottom=bottom, color=COLORS[t], label=t, alpha=0.9, width=0.6)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(["base\n(max-Sharpe)"] + [r.replace("_", "\n") for r in REGIME_ORDER],
                       fontsize=8)
    ax.set_title("Regime-Conditioned Weight Tilts", fontweight="bold")
    ax.set_ylabel("Weight (%)"); ax.set_ylim(0, 100)
    ax.legend(fontsize=8, ncol=4, loc="upper center"); ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: spread z-score with entry/exit bands and the position taken.
    ax = axes[1, 1]
    ax.plot(sig.zscore, color="#2196F3", lw=1.2, label="spread z-score")
    for level, ls, lbl in [(sig.entry_z, "--", f"entry ±{sig.entry_z}σ"),
                           (sig.exit_z, ":", f"exit ±{sig.exit_z}σ")]:
        ax.axhline(level, color="#c00" if ls == "--" else "#888", ls=ls, lw=1, label=lbl)
        ax.axhline(-level, color="#c00" if ls == "--" else "#888", ls=ls, lw=1)
    ax.axhline(0, color="k", lw=0.8)
    long_mask = (sig.position > 0).to_numpy()
    short_mask = (sig.position < 0).to_numpy()
    lo, hi = sig.zscore.min(), sig.zscore.max()
    ax.fill_between(sig.zscore.index, lo, hi, where=long_mask,
                    color="#4CAF50", alpha=0.15, step="mid", label="long spread")
    ax.fill_between(sig.zscore.index, lo, hi, where=short_mask,
                    color="#E53935", alpha=0.15, step="mid", label="short spread")
    ax.set_title(f"Spread Signal — {pair.y} ~ {pair.x} (β={pair.hedge_ratio:.2f}, "
                 f"EG p={pair.eg_pvalue:.3f})", fontweight="bold")
    ax.set_ylabel("z-score"); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = DATA_DIR / "phase2_signals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved: {out}")


if __name__ == "__main__":
    signals_report()
