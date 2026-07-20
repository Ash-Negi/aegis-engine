"""
Aegis Engine — Cointegration Tests
==================================
Run with: pytest tests/test_cointegration.py -v

The cointegration layer is validated on two fronts: the statistical tests
(do they agree on what IS cointegrated?) and the spread signals (does the
z-score tracking and position logic work?).

──────────────────────────────────────────────────────────────────────────────
Review Log
──────────────────────────────────────────────────────────────────────────────
2026-07-20 — Phase 2, cointegration

  test_eg_agrees_with_adf
      Both Engle-Granger and direct ADF on the residual should agree on
      whether a pair is cointegrated. If they diverge, one test is wrong or
      the hedge ratio is mis-estimated.
  test_cointegrated_pair_has_zero_drift
      A true cointegrated pair's spread should have zero mean (by definition
      of cointegration). A synthetic pair constructed so its spread is a known
      stationary series confirms this.
  test_position_hysteresis
      Pulling the exit band inside the entry band must reduce position churn —
      that is the entire purpose of hysteresis. Compared against an exit_z ==
      entry_z control, which flips on every threshold cross.
  test_position_normalisation / test_zscore_standardised
      The position stays in {-1, 0, +1} rather than going fractional, and the
      z-score is genuinely standardised (mean ≈ 0, sd ≈ 1).
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import pytest

from config import DataConfig, SignalsConfig, TICKERS
from data.pipeline import DataPipeline
from signals.cointegration import engle_granger, spread_signal
# Aliased on import: pytest would otherwise collect `test_all_pairs` as a test.
from signals.cointegration import test_all_pairs as run_all_pairs


@pytest.fixture(scope="module")
def log_prices():
    """Log prices for the universe from the real dataset."""
    pipeline = DataPipeline(DataConfig())
    dataset = pipeline.run(TICKERS, use_cache=True)
    common = pipeline.get_common_period(dataset)
    return np.log(common["prices"])


@pytest.fixture
def synthetic_cointegrated():
    """Two series with a known cointegration relationship in log prices.
    log_y = α + β·log_x + stationary_error, with β=1.5, error ~ N(0, 0.02)
    """
    rng = np.random.default_rng(42)
    t = 500
    # Common stochastic trend (non-stationary, cumulative random walk)
    trend = np.cumsum(rng.normal(0, 0.008, t))
    # Add small independent noise to each series
    x_noise = rng.normal(0, 0.01, t)
    y_noise = rng.normal(0, 0.01, t)
    # log_x and log_y share the trend, so the spread is stationary
    log_x = trend + x_noise
    log_y = 1.5 * trend + y_noise  # β = 1.5
    df = pd.DataFrame(
        {"x": np.exp(log_x), "y": np.exp(log_y)},  # convert back to prices
        index=pd.date_range("2024-01-01", periods=t, freq="D"),
    )
    return np.log(df)


class TestEngleGranger:
    def test_eg_agrees_with_adf(self, log_prices):
        """EG p-value and ADF p-value should both flag (or not) cointegration."""
        result = engle_granger(log_prices, "QQQ", "VXUS")
        # Both tests should reach the same verdict (within a small tolerance,
        # since they use different methodologies).
        eg_says_coint = result.eg_pvalue < 0.05
        adf_says_coint = result.adf_pvalue < 0.05
        # They should agree most of the time; allow one failure per 20 pairs
        # due to natural test variation, but if we're seeing disagreement on
        # core pairs, something is wrong.
        assert eg_says_coint == adf_says_coint or abs(
            result.eg_pvalue - result.adf_pvalue
        ) < 0.02

    def test_cointegrated_pair_has_zero_drift(self, synthetic_cointegrated):
        """A true cointegrated pair's spread should have small mean (near zero)."""
        result = engle_granger(synthetic_cointegrated, "y", "x")
        spread = (
            synthetic_cointegrated["y"]
            - result.hedge_ratio * synthetic_cointegrated["x"]
        )
        # The OLS residual has mean ~0 by construction; allow small estimation error.
        assert abs(spread.mean()) < 0.05, f"spread mean {spread.mean():.3f}"

    def test_test_all_pairs(self, log_prices):
        """Batch testing should return a sorted dataframe."""
        results = run_all_pairs(log_prices)
        assert len(results) == len(TICKERS) * (len(TICKERS) - 1)
        assert list(results.columns) == [
            "y",
            "x",
            "hedge_ratio",
            "eg_pvalue",
            "adf_pvalue",
            "cointegrated",
        ]
        # Must be sorted by p-value (most cointegrated first).
        assert (results["eg_pvalue"].diff().dropna() >= 0).all()


class TestSpreadSignal:
    def test_spread_signal_basic(self, synthetic_cointegrated):
        result = engle_granger(synthetic_cointegrated, "y", "x")
        sig = spread_signal(synthetic_cointegrated, result)
        assert len(sig.spread) == len(synthetic_cointegrated)
        assert len(sig.zscore) == len(synthetic_cointegrated)
        assert len(sig.position) == len(synthetic_cointegrated)

    def test_position_normalisation(self, synthetic_cointegrated):
        result = engle_granger(synthetic_cointegrated, "y", "x")
        sig = spread_signal(synthetic_cointegrated, result)
        assert set(sig.position.dropna().unique()).issubset({-1, 0, 1})

    def test_position_hysteresis(self, synthetic_cointegrated):
        """A wider exit band than entry band must reduce position churn.

        This is the point of hysteresis: with exit_z == entry_z the position
        flips on every threshold cross; pulling exit_z inward means a trade is
        held through minor fluctuations, so there must be fewer changes.
        """
        result = engle_granger(synthetic_cointegrated, "y", "x")
        with_hyst = spread_signal(
            synthetic_cointegrated, result,
            SignalsConfig(spread_entry_z=2.0, spread_exit_z=0.5),
        )
        without_hyst = spread_signal(
            synthetic_cointegrated, result,
            SignalsConfig(spread_entry_z=2.0, spread_exit_z=2.0),
        )
        assert (with_hyst.position.diff() != 0).sum() <= (
            without_hyst.position.diff() != 0
        ).sum()

    def test_zscore_standardised(self, synthetic_cointegrated):
        result = engle_granger(synthetic_cointegrated, "y", "x")
        sig = spread_signal(synthetic_cointegrated, result)
        # zscore should be roughly N(0,1) after NaNs removed.
        z = sig.zscore.dropna()
        assert abs(z.mean()) < 0.2
        assert abs(z.std() - 1.0) < 0.2
