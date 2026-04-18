"""
Aegis Engine — Data Pipeline Tests
====================================
Tests that verify the pipeline produces correct, clean data.

Run with: pytest tests/ -v

Why test a data pipeline?
    Because silent data bugs are the most dangerous bugs in a trading system.
    A crash is obvious. A wrong number that looks plausible will propagate
    through your optimizer, produce bad weights, and lose real money.
    These tests are your first line of defense.

──────────────────────────────────────────────────────────────────────────────
Review Log
──────────────────────────────────────────────────────────────────────────────
A running record of test changes, with the reasoning behind each one. New
entries on top. Keep entries terse: name the test, say WHAT invariant it
protects, and WHY a bug here would be easy to miss without it.

2026-04-18 — Claude review (Phase 1 / Week 1)

  Guiding principle for the additions:
    The existing tests checked math (return formulas) and rough sanity
    (stats in plausible ranges). What they didn't check were CONTRACTS —
    structural promises the pipeline makes to every downstream module
    (covariance, optimizer, signals). A broken contract is the kind of
    bug where every individual test still passes but the overall output
    is subtly wrong. Covariance estimation in particular is sensitive
    to column ordering, index alignment, and the pre-inception handling
    of FBTC — so those are the contracts to lock down first.

  Added — contract-level invariants:
    test_common_start_matches_fbtc_inception
        Pins common_start to 2024-01-11 (FBTC inception). If this ever
        drifts (e.g. a future ticker launches later), every downstream
        stat silently changes. Hard-coding the expected date makes the
        sample boundary explicit.

    test_fbtc_nan_before_inception
        Guards against a silent bug where forward-fill or a fetch change
        accidentally populates FBTC with values before it existed. Every
        pre-inception "price" would be fabricated and would poison any
        long-window analysis.

    test_returns_first_row_is_nan
        pct_change() and np.log(P_t / P_{t-1}) both need a prior price,
        so row 0 must be all-NaN. Downstream modules will rely on this
        (e.g. ".dropna()" assumes only the first row needs dropping).

    test_column_order_matches_tickers
        _fetch() explicitly does raw["Close"][tickers] to enforce input
        order, because yfinance returns columns alphabetically. If this
        regresses, covariance matrices get built with permuted rows/cols
        — a classic silent bug where weights come out assigned to the
        wrong assets.

    test_index_alignment_across_frames
        prices, returns, log_returns must share the same DatetimeIndex.
        Any drift (e.g. dropna applied unevenly) would mean a "return
        on date X" and "price on date X" refer to different rows. Cheap
        to check, catastrophic if it ever breaks.

    test_total_return_matches_ann_return
        total_return is computed from price endpoints; ann_return from
        mean log returns × 252. They come from the same series via
        different paths, so they MUST satisfy
            ln(1 + total_return) ≈ ann_return * n / 252
        If they disagree, one of the two calcs has drifted. This is a
        cheap cross-check that would catch a subtle indexing bug in
        either calculation.

  Tightened — existing tests whose docstrings disagreed with their code:
    test_volatility_reasonable_range
        Docstring claimed per-ticker bands (equity 10-50%, gold 10-25%,
        BTC 30-100%+) but the assertion was 1 < vol < 200. Docstring
        rewritten to match reality: the test catches obviously broken
        numbers (0%, 1000%), not miscalibration. Calibration is a
        modeling concern, not a data-pipeline concern.

    test_sharpe_reasonable_range
        Same drift: docstring said [-3, 5], assertion was [-5, 10].
        Docstring rewritten to reflect the loose-sanity intent.

    test_simple_returns_not_additive
        Previous version used atol=1e-15, which only catches bit-level
        equality — a test that's almost impossible to fail even with
        buggy code. Tightened to assert the difference is on the order
        of r1*r2 (the mathematical size of the non-additivity term),
        so the test now actually validates the property it claims to.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import pytest

from config import DataConfig, TICKERS
from data.pipeline import DataPipeline


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline():
    """Create a pipeline with default config."""
    return DataPipeline(DataConfig())


@pytest.fixture
def dataset(pipeline):
    """Run the pipeline once and reuse across tests."""
    return pipeline.run(TICKERS, use_cache=True)


# ─── Data Integrity Tests ─────────────────────────────────────────────────────

class TestDataIntegrity:
    """Verify the raw data is structurally sound."""

    def test_all_tickers_present(self, dataset):
        """Every ticker in the universe must have a column."""
        for ticker in TICKERS:
            assert ticker in dataset.prices.columns, (
                f"Missing ticker: {ticker}"
            )

    def test_prices_are_positive(self, dataset):
        """Prices must be positive. Negative prices = data corruption."""
        common_prices = dataset.prices.loc[dataset.common_start:].dropna()
        for ticker in TICKERS:
            assert (common_prices[ticker] > 0).all(), (
                f"{ticker} has non-positive prices"
            )

    def test_no_duplicate_dates(self, dataset):
        """Each date should appear exactly once."""
        assert not dataset.prices.index.duplicated().any(), (
            "Duplicate dates found in price index"
        )

    def test_index_is_sorted(self, dataset):
        """Dates must be in ascending order."""
        assert dataset.prices.index.is_monotonic_increasing, (
            "Price index is not sorted ascending"
        )

    def test_common_period_has_no_nans(self, dataset, pipeline):
        """The common period (all assets have data) should be NaN-free."""
        common = pipeline.get_common_period(dataset)
        assert not common["prices"].isna().any().any(), (
            "NaN values found in common period prices"
        )

    # ── Contract tests (added 2026-04-18) ──
    # See "Review Log" at top of file for why these exist.

    def test_common_start_matches_fbtc_inception(self, dataset):
        """
        common_start must be FBTC's inception date (2024-01-11).

        FBTC is the youngest asset in the universe, so it bounds the
        common period from below. Hard-coding the expected date here
        makes the sample boundary an explicit, auditable fact — if a
        future change causes this date to drift (e.g. a data-source
        bug that misreports FBTC's first bar), every downstream stat
        changes silently. We'd rather fail loudly here.
        """
        assert dataset.common_start == pd.Timestamp("2024-01-11"), (
            f"common_start={dataset.common_start} "
            "does not match FBTC inception (2024-01-11)"
        )

    def test_fbtc_nan_before_inception(self, dataset):
        """
        FBTC must be NaN for every trading day before 2024-01-11.

        The fund did not exist. If anything appears there, it was
        fabricated — either by a bad yfinance response or by a
        forward-fill that reached too far. Fabricated pre-inception
        data would poison any long-window analysis that thinks it's
        looking at real prices.
        """
        pre_inception = dataset.prices.loc[:"2024-01-10", "FBTC"]
        assert pre_inception.isna().all(), (
            f"{pre_inception.notna().sum()} non-NaN FBTC values found "
            "before inception (2024-01-11)"
        )

    def test_column_order_matches_tickers(self, dataset):
        """
        prices/returns/log_returns must list columns in TICKERS order.

        yfinance returns columns alphabetically. The pipeline does
        `raw["Close"][tickers]` in _fetch() specifically to enforce the
        caller's order. If that line regresses, a covariance matrix
        built by zipping TICKERS with the DataFrame columns would line
        up the wrong assets to the wrong rows/columns — optimizer
        weights would be assigned to the wrong tickers, silently.
        """
        assert list(dataset.prices.columns) == list(TICKERS)
        assert list(dataset.returns.columns) == list(TICKERS)
        assert list(dataset.log_returns.columns) == list(TICKERS)

    def test_index_alignment_across_frames(self, dataset):
        """
        prices, returns, and log_returns must share the same DatetimeIndex.

        If they drift (for instance, because .dropna() is applied
        differently to one but not another), then "return on date X"
        and "price on date X" point to different rows. Every downstream
        lookup-by-date would then be off-by-one in a hard-to-spot way.
        """
        assert dataset.prices.index.equals(dataset.returns.index), (
            "prices and returns have misaligned indices"
        )
        assert dataset.prices.index.equals(dataset.log_returns.index), (
            "prices and log_returns have misaligned indices"
        )


# ─── Return Calculation Tests ─────────────────────────────────────────────────

class TestReturns:
    """Verify return calculations are mathematically correct."""

    def test_simple_returns_formula(self, dataset):
        """Simple return = (P_t - P_{t-1}) / P_{t-1}"""
        # Use date-based alignment, not positional indexing.
        # Why: positional indexing (.iloc) is fragile — if .dropna()
        # removes different rows from prices vs returns (e.g. because
        # FBTC has no return on its first day), the positions diverge.
        # Date-based lookup (.loc[date]) is immune to this.
        prices = dataset.prices.loc[dataset.common_start:].dropna()
        ticker = TICKERS[0]

        # Pick two consecutive dates from the price index
        date_yesterday = prices.index[4]
        date_today = prices.index[5]

        # Manual calculation: (P_today - P_yesterday) / P_yesterday
        p_today = prices[ticker].loc[date_today]
        p_yesterday = prices[ticker].loc[date_yesterday]
        expected = (p_today - p_yesterday) / p_yesterday

        # Look up the pipeline's return at the same date
        actual_val = dataset.returns.loc[date_today, ticker]

        np.testing.assert_almost_equal(actual_val, expected, decimal=10)

    def test_log_returns_formula(self, dataset):
        """Log return = ln(P_t / P_{t-1})"""
        # Same date-based approach as test_simple_returns_formula.
        prices = dataset.prices.loc[dataset.common_start:].dropna()
        ticker = TICKERS[0]

        date_yesterday = prices.index[4]
        date_today = prices.index[5]

        p_today = prices[ticker].loc[date_today]
        p_yesterday = prices[ticker].loc[date_yesterday]
        expected = np.log(p_today / p_yesterday)

        actual_val = dataset.log_returns.loc[date_today, ticker]

        np.testing.assert_almost_equal(actual_val, expected, decimal=10)

    def test_log_return_additivity(self, dataset):
        """
        Log returns must be additive:
        log_ret(day1→day3) = log_ret(day1→day2) + log_ret(day2→day3)

        This is the core mathematical property that justifies using
        log returns instead of simple returns.
        """
        # Use dates, not positions, so this test doesn't depend on
        # which rows .dropna() removes.
        prices = dataset.prices.loc[dataset.common_start:].dropna()
        ticker = TICKERS[0]

        # Pick three consecutive trading dates
        date1 = prices.index[10]
        date2 = prices.index[11]
        date3 = prices.index[12]

        # Two-day log return computed directly from prices
        two_day_direct = np.log(prices[ticker].loc[date3] / prices[ticker].loc[date1])

        # Sum of individual log returns from the pipeline
        two_day_sum = (
            dataset.log_returns.loc[date2, ticker]
            + dataset.log_returns.loc[date3, ticker]
        )

        np.testing.assert_almost_equal(two_day_direct, two_day_sum, decimal=10)

    def test_simple_returns_not_additive(self, dataset):
        """
        Simple returns are NOT additive. The gap between compounded and
        summed returns is exactly r1 * r2 — i.e.

            (1 + r1)(1 + r2) - 1  -  (r1 + r2)  =  r1 * r2

        (Tightened 2026-04-18 — previous version used atol=1e-15 which
        only caught bit-equality. This now asserts the exact shape of
        the non-additivity term, which validates the underlying math
        rather than a weak inequality.)
        """
        prices = dataset.prices.loc[dataset.common_start:].dropna()
        ticker = TICKERS[0]

        date1 = prices.index[10]
        date2 = prices.index[11]
        date3 = prices.index[12]

        # Two-day simple return computed directly from prices
        two_day_direct = (
            prices[ticker].loc[date3] - prices[ticker].loc[date1]
        ) / prices[ticker].loc[date1]

        r1 = dataset.returns.loc[date2, ticker]
        r2 = dataset.returns.loc[date3, ticker]
        two_day_sum = r1 + r2

        # The gap must equal r1 * r2 exactly (to floating-point precision).
        np.testing.assert_almost_equal(
            two_day_direct - two_day_sum, r1 * r2, decimal=10
        )

    # ── Contract test (added 2026-04-18) ──
    def test_returns_first_row_is_nan(self, dataset):
        """
        The first row of returns and log_returns must be entirely NaN.

        pct_change and log(P_t / P_{t-1}) both need a prior price, which
        doesn't exist for row 0. Downstream code commonly does
        .loc[common_start:].dropna() expecting to lose only the first
        row — if some cell in row 0 is non-NaN due to a bug, dropna()
        will silently leave the row in place and misalign subsequent
        operations.
        """
        assert dataset.returns.iloc[0].isna().all(), (
            "First row of returns must be all-NaN"
        )
        assert dataset.log_returns.iloc[0].isna().all(), (
            "First row of log_returns must be all-NaN"
        )


# ─── Statistics Tests ─────────────────────────────────────────────────────────

class TestStatistics:
    """Verify computed statistics are in reasonable ranges."""

    def test_volatility_positive(self, dataset):
        """Annualized volatility must be positive for all assets."""
        for ticker in dataset.stats.index:
            assert dataset.stats.loc[ticker, "ann_vol_pct"] > 0

    def test_volatility_reasonable_range(self, dataset):
        """
        Broad sanity check: annualized vol must be in (1%, 200%).

        This is deliberately loose — it catches obviously broken numbers
        (0%, 1000%) but does NOT validate calibration. Narrow per-asset
        bands (e.g. "QQQ vol must be 15-25%") would be regime-dependent
        and cause tests to flap when market conditions change. The job
        of this test is to catch calculation errors, not to validate
        market reality.
        """
        for ticker in dataset.stats.index:
            vol = dataset.stats.loc[ticker, "ann_vol_pct"]
            assert 1 < vol < 200, (
                f"{ticker} has unreasonable annualized vol: {vol:.1f}%"
            )

    def test_sharpe_reasonable_range(self, dataset):
        """
        Broad sanity check: Sharpe ratio must be in (-5, 10).

        Same philosophy as test_volatility_reasonable_range — catches
        calculation errors (values of 100 or 1000) without asserting
        anything about market calibration. A Sharpe of 10 is implausible
        but not impossible in short windows; we only fail on clearly
        broken numbers.
        """
        for ticker in dataset.stats.index:
            sharpe = dataset.stats.loc[ticker, "sharpe"]
            assert -5 < sharpe < 10, (
                f"{ticker} has unreasonable Sharpe ratio: {sharpe:.3f}"
            )

    def test_max_drawdown_is_negative(self, dataset):
        """Max drawdown should always be negative (or zero for cash)."""
        for ticker in dataset.stats.index:
            dd = dataset.stats.loc[ticker, "max_drawdown_pct"]
            assert dd <= 0, (
                f"{ticker} has positive max drawdown: {dd:.2f}%"
            )

    def test_win_rate_between_0_and_100(self, dataset):
        """Daily win rate must be a valid percentage."""
        for ticker in dataset.stats.index:
            wr = dataset.stats.loc[ticker, "daily_win_rate_pct"]
            assert 0 <= wr <= 100
    # ── Contract test (added 2026-04-18) ──
    def test_total_return_matches_ann_return(self, dataset):
        """
        Cross-check: total_return and ann_return are computed from the
        same price series via different paths and must agree.

            total_return = P_last / P_first - 1              (endpoints)
            ann_return   = mean(log_return) * 252            (mean of logs)

        Since sum(log_return) = ln(P_last / P_first), the two must satisfy

            ln(1 + total_return) = ann_return * n / 252

        where n = trading_days in the common period. If a subtle bug
        creeps into EITHER calc (off-by-one in indexing, wrong
        annualization factor, price vs. adj-close mismatch), this
        equality breaks. Cheap cross-check, high diagnostic value.

        Tolerance is 0.5 percentage points — well above the rounding
        noise introduced by stats being stored to 2 decimals, and well
        below the error any real bug would produce.
        """
        for ticker in dataset.stats.index:
            total_r = dataset.stats.loc[ticker, "total_return_pct"] / 100
            ann_r = dataset.stats.loc[ticker, "ann_return_pct"] / 100
            n = dataset.stats.loc[ticker, "trading_days"]
            expected_total = np.exp(ann_r * n / 252) - 1
            np.testing.assert_allclose(
                total_r, expected_total, atol=0.005,
                err_msg=(
                    f"{ticker}: total_return ({total_r:.4f}) and "
                    f"ann_return ({ann_r:.4f} over {n} days) disagree"
                ),
            )
