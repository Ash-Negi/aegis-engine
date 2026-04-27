"""
Aegis Engine — Phase 1, Week 1
================================
Run the data pipeline and inspect the asset universe.

This script is your first interaction with real financial data.
Run it, look at the output, and ask yourself:
    - Do these numbers make sense?
    - Which asset is the most volatile?
    - Which asset has the fattest tails?
    - How does the correlation structure look?

Usage:
    cd math-engine
    python main.py
"""
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import DataConfig, PortfolioConfig, TICKERS, UNIVERSE, DATA_DIR
from data.pipeline import DataPipeline

# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─── Display Helpers ──────────────────────────────────────────────────────────

def print_header(text: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {text}")
    print(f"{'═' * 70}")


def print_section(text: str) -> None:
    print(f"\n  ── {text} {'─' * max(0, 60 - len(text))}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:

    print_header("AEGIS ENGINE - Phase 1, Week 1: Data Pipeline.")

    # ── Initialize ────────────────────────────────────────────────────────
    data_config = DataConfig()
    portfolio_config = PortfolioConfig()
    pipeline = DataPipeline(data_config)

    print("\n  Asset Universe:")
    for asset in UNIVERSE:
        print(f"    {asset.ticker:<6} {asset.name:<40} [{asset.role}]")

    # ── Run Pipeline ──────────────────────────────────────────────────────
    print_section("Fetching and Cleaning Data")
    #the pipeline returns a data container of type Dataset
    dataset = pipeline.run(TICKERS, use_cache=True)

    # ── Data Availability Report ──────────────────────────────────────────
    print_section("Data Availability")
    for ticker, info in dataset.metadata["availability"].items():
        print(
            f"    {ticker:<6} {info['valid_days']:>5} days  "
            f"({info['coverage_pct']:>5.1f}%)  "
            f"from {info['first_date']}"
        )

    print(f"\n    Common start date: {dataset.common_start.date()}")
    common = pipeline.get_common_period(dataset)
    print(f"    Common period trading days: {len(common['prices'])}")

    # ── Descriptive Statistics ────────────────────────────────────────────
    print_section("Descriptive Statistics (Common Period)")
    print()

    # Format the stats table for display
    stats = dataset.stats
    fmt_rows = []
    for ticker in stats.index:
        row = stats.loc[ticker]
        fmt_rows.append(
            f"    {ticker:<6} "
            f"ret={row['ann_return_pct']:>+7.2f}%  "
            f"vol={row['ann_vol_pct']:>6.2f}%  "
            f"sharpe={row['sharpe']:>6.3f}  "
            f"maxDD={row['max_drawdown_pct']:>7.2f}%  "
            f"skew={row['skewness']:>6.3f}  "
            f"kurt={row['excess_kurtosis']:>6.3f}"
        )
    for row in fmt_rows:
        print(row)

    # ── Interpretation Guide ──────────────────────────────────────────────
    print_section("What to Look For")

    # Find extremes
    most_volatile = stats["ann_vol_pct"].idxmax()
    least_volatile = stats["ann_vol_pct"].idxmin()
    best_sharpe = stats["sharpe"].idxmax()
    fattest_tails = stats["excess_kurtosis"].idxmax()
    most_skewed = stats.loc[stats["skewness"].abs().idxmax()]

    print(f"    Most volatile:    {most_volatile} ({stats.loc[most_volatile, 'ann_vol_pct']:.1f}% ann. vol)")
    print(f"    Least volatile:   {least_volatile} ({stats.loc[least_volatile, 'ann_vol_pct']:.1f}% ann. vol)")
    print(f"    Best Sharpe:      {best_sharpe} ({stats.loc[best_sharpe, 'sharpe']:.3f})")
    print(f"    Fattest tails:    {fattest_tails} (excess kurtosis: {stats.loc[fattest_tails, 'excess_kurtosis']:.3f})")
    print()
    print("    NOTE: Excess kurtosis > 0 means fatter tails than a normal distribution.")
    print("    The optimizer assumes normality — high kurtosis assets will surprise you")
    print(f"    with larger moves than the model predicts. {fattest_tails} has the fattest")
    print("    tails in this sample — this is why risk management matters more than optimization.")

    # ── Correlation Matrix ────────────────────────────────────────────────
    print_section("Correlation Matrix (Log Returns, Common Period)")
    corr = common["log_returns"].corr()
    print()
    # Header
    print(f"    {'':>6}", end="")
    for t in corr.columns:
        print(f" {t:>8}", end="")
    print()

    # Rows
    for t in corr.index:
        print(f"    {t:<6}", end="")
        for t2 in corr.columns:
            val = corr.loc[t, t2]
            print(f" {val:>8.4f}", end="")
        print()

    print()
    print("    Low correlation between assets = diversification benefit.")
    print("    This is the raw material that makes portfolio optimization work.")
    print("    If all correlations were 1.0, there would be nothing to optimize.")

    # ── Generate Plots ────────────────────────────────────────────────────
    print_section("Generating Plots")
    plot_results(dataset, common, pipeline)

    print_header("Pipeline Complete")
    print("\n  Next steps:")
    print("    1. Review the plots in data/store/")
    print("    2. Examine the correlation matrix — which pairs diversify best?")
    print("    3. Look at the kurtosis values — which assets violate normality most?")
    print("    4. Week 2: Build covariance estimation (sample, EWMA, Ledoit-Wolf)")
    print()

    # ─── Visualization ────────────────────────────────────────────────────────────

def plot_results(dataset, common, pipeline) -> None:
    """Generate Phase 1 Week 1 diagnostic plots"""

    fig, axes = plt.subplots(2,2, figsize=(16,10))
    fig.suptitle(
    "Aegis Engine — Phase 1 Week 1: Asset Universe Overview",
    fontsize=14,
    fontweight="bold",
    y=0.98,
    )
    #avoid hardcoding, fix this later
    colors = {
        "QQQ": "#2196F3",
        "GLDM": "#FFD700",
        "FBTC": "#FF6B35",
        "VXUS": "#4CAF50",
    }

    # ── Panel 1: Normalized Prices (common period) ────────────────────────
    ax = axes[0, 0]
    norm = common["prices"] / common["prices"].iloc[0] * 100
    for ticker in TICKERS:
        ax.plot(
            norm.index, norm[ticker],
            label=ticker, linewidth=1.5, color=colors[ticker],
        )
    ax.set_title("Normalized Prices (100 = common start)", fontweight="bold")
    ax.set_ylabel("Normalized Price")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Rolling 60-Day Volatility ────────────────────────────────
    ax = axes[0, 1]
    window = 60
    for ticker in TICKERS:
        rolling_vol = (
            common["log_returns"][ticker].rolling(window).std()
            * np.sqrt(252)
            * 100
        )
        ax.plot(
            rolling_vol.index, rolling_vol,
            label=ticker, linewidth=1.2, color=colors[ticker],
        )
    ax.set_title(f"Rolling {window}-Day Annualized Volatility", fontweight="bold")
    ax.set_ylabel("Volatility (%)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Return Distributions ─────────────────────────────────────
    ax = axes[1, 0]
    for ticker in TICKERS:
        daily_lr = common["log_returns"][ticker].dropna()
        ax.hist(
            daily_lr, bins=80, alpha=0.5, density=True,
            label=ticker, color=colors[ticker],
        )
    ax.set_title("Daily Log Return Distributions", fontweight="bold")
    ax.set_xlabel("Daily Log Return")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Correlation Heatmap ──────────────────────────────────────
    ax = axes[1, 1]
    corr = common["log_returns"].corr()
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TICKERS)))
    ax.set_xticklabels(TICKERS)
    ax.set_yticks(range(len(TICKERS)))
    ax.set_yticklabels(TICKERS)
    ax.set_title("Correlation Matrix (Log Returns)", fontweight="bold")

    # Add correlation values as text
    for i in range(len(TICKERS)):
        for j in range(len(TICKERS)):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8)

    # ── Save ──────────────────────────────────────────────────────────────
    plt.tight_layout()
    output_path = DATA_DIR / "phase1_week1_overview.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved: {output_path}")

# ─── Entry Point ──────────────────────────────────────────────────────────────
#tells Python: "If this script is being executed directly from the terminal (e.g., python main.py), run the main() function. But if this file is being imported by another script, do not run it."
if __name__ == "__main__":
    main()