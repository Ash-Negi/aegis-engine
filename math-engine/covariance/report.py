"""
Aegis Engine — Phase 1, Week 2: Covariance Report
=================================================
Runs the three covariance estimators on the common-period log returns and
lays their conditioning and risk-factor structure side by side. This is the
Week 2 counterpart to Week 1's main.py: a human-readable answer to the two
questions the brief poses —

    1. How does the log condition number change before vs after shrinkage?
    2. How many *real* risk factors does the 4×4 matrix contain?

Usage (from math-engine/):
    python -m covariance.report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DataConfig, CovarianceConfig, TICKERS, DATA_DIR
from data.pipeline import DataPipeline
from covariance import (
    sample_covariance,
    ewma_covariance,
    ledoit_wolf_shrinkage,
    log_condition_number,
    eigen_analysis,
)


# ─── Display helpers (match main.py's house style) ────────────────────────────

def _header(text: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {text}")
    print(f"{'═' * 70}")


def _section(text: str) -> None:
    print(f"\n  ── {text} {'─' * max(0, 60 - len(text))}")


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix."""
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def _ann_vol(cov: np.ndarray, trading_days: int) -> np.ndarray:
    """Annualized volatilities (%) from the covariance diagonal."""
    return np.sqrt(np.diag(cov) * trading_days) * 100


# ─── Report ───────────────────────────────────────────────────────────────────

def covariance_report(config: CovarianceConfig | None = None) -> None:
    config = config or CovarianceConfig()

    _header("AEGIS ENGINE — Phase 1, Week 2: Covariance Estimation")

    pipeline = DataPipeline(DataConfig())
    dataset = pipeline.run(TICKERS, use_cache=True)
    log_returns = pipeline.get_common_period(dataset)["log_returns"]

    print(f"\n  Common period: {log_returns.index[0].date()} → "
          f"{log_returns.index[-1].date()}   (T = {len(log_returns)} days, "
          f"N = {len(TICKERS)} assets)")

    # ── Build the three estimators ────────────────────────────────────────
    sample = sample_covariance(log_returns, config)
    ewma = ewma_covariance(log_returns, config)
    lw = ledoit_wolf_shrinkage(log_returns, config)

    estimators = {
        "sample": sample.to_numpy(),
        "ewma": ewma.to_numpy(),
        "ledoit-wolf": lw.covariance.to_numpy(),
    }

    # ── Conditioning comparison ───────────────────────────────────────────
    _section("Conditioning (log₁₀ condition number — lower is safer to invert)")
    print()
    print(f"    {'estimator':<14}{'log₁₀ κ':>10}{'κ':>12}")
    for name, cov_np in estimators.items():
        frame = {"sample": sample, "ewma": ewma, "ledoit-wolf": lw.covariance}[name]
        logk = log_condition_number(frame)
        print(f"    {name:<14}{logk:>10.3f}{10**logk:>12.1f}")
    print(f"\n    Ledoit-Wolf shrinkage intensity δ = {lw.shrinkage:.4f}")
    print(f"    (δ small ⇒ T={len(log_returns)} ≫ N={len(TICKERS)}, the sample "
          f"cov is already trustworthy; shrinkage is a light touch here.)")

    # ── Annualized volatilities ───────────────────────────────────────────
    _section("Annualized volatility (%) by estimator")
    print()
    print(f"    {'ticker':<8}" + "".join(f"{n:>14}" for n in estimators))
    for i, t in enumerate(TICKERS):
        row = "".join(f"{_ann_vol(cov, config.trading_days_per_year)[i]:>14.2f}"
                      for cov in estimators.values())
        print(f"    {t:<8}{row}")

    # ── Eigenstructure: how many real risk factors? ───────────────────────
    _section("Risk factors (eigen-decomposition of the sample covariance)")
    ea = eigen_analysis(sample, config)
    print()
    print(f"    {'factor':<10}{'eigenvalue':>14}{'var %':>10}{'cum %':>10}")
    for i in range(len(ea.eigenvalues)):
        print(f"    factor_{i + 1:<3}{ea.eigenvalues[i]:>14.3e}"
              f"{ea.variance_explained[i] * 100:>10.1f}{ea.cumulative_variance[i] * 100:>10.1f}")
    print(f"\n    Factors above {config.factor_variance_threshold:.0%} of variance: "
          f"{ea.n_factors} of {len(TICKERS)}")
    print(f"    Participation ratio (effective # of factors): {ea.participation_ratio:.2f}")
    print(f"    Reading: {ea.variance_explained[0] * 100:.0f}% of all portfolio "
          f"variance lives in the first factor — the market/beta direction.")

    # ── Plot ──────────────────────────────────────────────────────────────
    _section("Generating diagnostic plot")
    _plot(sample.to_numpy(), lw, ea, estimators, config)

    _header("Week 2 Complete")
    print("\n  Next: Week 3 — mean-variance optimizer (Σ⁻¹μ), efficient frontier,")
    print("  long-only + sector-cap constraints.\n")


# ─── Visualization ─────────────────────────────────────────────────────────────

def _plot(sample_np, lw, ea, estimators, config) -> None:
    """Four-panel Week 2 diagnostic, in main.py's visual style."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Aegis Engine — Phase 1 Week 2: Covariance Estimation",
                 fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: sample correlation heatmap
    ax = axes[0, 0]
    corr = _cov_to_corr(sample_np)
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TICKERS)), TICKERS)
    ax.set_yticks(range(len(TICKERS)), TICKERS)
    ax.set_title("Sample Correlation Matrix", fontweight="bold")
    for i in range(len(TICKERS)):
        for j in range(len(TICKERS)):
            v = corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: scree plot — variance explained + cumulative
    ax = axes[0, 1]
    factors = [f"F{i + 1}" for i in range(len(ea.eigenvalues))]
    ax.bar(factors, ea.variance_explained * 100, color="#2196F3", alpha=0.8,
           label="per factor")
    ax.plot(factors, ea.cumulative_variance * 100, color="#FF6B35",
            marker="o", linewidth=1.8, label="cumulative")
    ax.axhline(config.factor_variance_threshold * 100, color="#888",
               linestyle="--", linewidth=1, label=f"{config.factor_variance_threshold:.0%} threshold")
    ax.set_title(f"Risk-Factor Spectrum (participation ratio {ea.participation_ratio:.2f})",
                 fontweight="bold")
    ax.set_ylabel("Variance explained (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: log condition number by estimator
    ax = axes[1, 0]
    frames = {"sample": estimators["sample"], "ewma": estimators["ewma"],
              "ledoit-wolf": estimators["ledoit-wolf"]}
    names = list(frames)
    logks = [log_condition_number(pd.DataFrame(frames[n], index=TICKERS, columns=TICKERS))
             for n in names]
    bars = ax.bar(names, logks, color=["#888", "#4CAF50", "#2196F3"], alpha=0.85)
    ax.set_title("Conditioning by Estimator (lower = safer inverse)", fontweight="bold")
    ax.set_ylabel("log₁₀ condition number")
    for b, v in zip(bars, logks):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: shrinkage — sample vs LW eigenvalues pulled toward μ
    ax = axes[1, 1]
    sample_eigs = np.sort(np.linalg.eigvalsh(sample_np))[::-1]
    lw_eigs = np.sort(np.linalg.eigvalsh(lw.covariance.to_numpy()))[::-1]
    x = np.arange(len(sample_eigs))
    ax.plot(x, sample_eigs, marker="o", label="sample", color="#888", linewidth=1.8)
    ax.plot(x, lw_eigs, marker="s", label="ledoit-wolf", color="#2196F3", linewidth=1.8)
    ax.axhline(lw.mu, color="#FFD700", linestyle="--", linewidth=1.5,
               label=f"target μ (δ={lw.shrinkage:.3f})")
    ax.set_title("Shrinkage Pulls Eigenvalues Toward μ", fontweight="bold")
    ax.set_xlabel("factor (largest → smallest)")
    ax.set_ylabel("eigenvalue (daily variance)")
    ax.set_xticks(x, [f"F{i + 1}" for i in x])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = DATA_DIR / "phase1_week2_covariance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved: {out}")


if __name__ == "__main__":
    covariance_report()
