"""
Aegis Engine — Backtest Harness Tests
=====================================
Run with: pytest tests/test_backtest.py -v

A backtester's bugs are silent and flattering — they make returns look
better than reality. So the tests pin the frictions in the direction that
matters: costs can only REDUCE returns, a wide band means NO trading, and
the attribution must sum back to the gross return exactly (no return
manufactured or lost in the decomposition).

──────────────────────────────────────────────────────────────────────────────
Review Log
──────────────────────────────────────────────────────────────────────────────
2026-07-20 — Week 4, backtest harness

  test_zero_cost_zero_band_is_fixed_weight
      With no costs and a zero band (rebalance daily), the engine must equal
      the textbook daily-rebalanced fixed-weight portfolio. Anchors the
      simulation to a closed-form equity curve.
  test_costs_only_reduce / test_wide_band_never_trades
      The two friction directions: costs never help, a huge band never
      rebalances (turnover ≡ 0). A sign flip in either would be a
      return-inflating bug.
  test_attribution_sums_to_gross
      Per-asset contributions + the cost line reconcile to the gross/net
      totals exactly — the decomposition conserves return.
  test_weights_sum_to_one_daily
      Drift preserves full investment; a leak here would silently lever or
      de-lever the book.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import pytest

from config import DataConfig, PortfolioConfig, TICKERS
from data.pipeline import DataPipeline
from backtest import (
    BacktestEngine,
    performance_metrics,
    return_attribution,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_returns():
    pipeline = DataPipeline(DataConfig())
    dataset = pipeline.run(TICKERS, use_cache=True)
    return pipeline.get_common_period(dataset)["returns"]


@pytest.fixture
def synth():
    """Deterministic 2-asset returns and an equal-weight target."""
    r = pd.DataFrame(
        {"A": [0.01, 0.02, -0.01, 0.03], "B": [0.0, -0.01, 0.02, -0.02]},
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    target = pd.Series({"A": 0.5, "B": 0.5})
    return r, target


# ─── Engine ───────────────────────────────────────────────────────────────────

class TestEngine:
    def test_zero_cost_zero_band_is_fixed_weight(self, synth):
        r, target = synth
        cfg = PortfolioConfig(transaction_cost_bps=0, slippage_bps=0,
                              rebalance_band_pct=0.0, initial_capital=1000.0)
        res = BacktestEngine(cfg).run(r, target)

        port_ret = r @ target                      # daily fixed-weight return
        expected = 1000.0 * (1 + port_ret).cumprod()
        pd.testing.assert_series_equal(
            res.equity_curve, expected, check_names=False
        )
        # A zero band rebalances daily (turnover > 0), but at zero cost the
        # equity curve is still exactly the fixed-weight one checked above.
        assert res.turnover.sum() > 0

    def test_costs_only_reduce(self, real_returns):
        target = pd.Series(1 / len(TICKERS), index=TICKERS)
        free = PortfolioConfig(transaction_cost_bps=0, slippage_bps=0, rebalance_band_pct=2.0)
        costly = PortfolioConfig(transaction_cost_bps=10, slippage_bps=5, rebalance_band_pct=2.0)
        r_free = BacktestEngine(free).run(real_returns, target)
        r_cost = BacktestEngine(costly).run(real_returns, target)
        assert (r_cost.net_returns <= r_free.net_returns + 1e-15).all()
        assert r_cost.equity_curve.iloc[-1] <= r_free.equity_curve.iloc[-1]

    def test_wide_band_never_trades(self, real_returns):
        target = pd.Series(1 / len(TICKERS), index=TICKERS)
        cfg = PortfolioConfig(rebalance_band_pct=1000.0)
        res = BacktestEngine(cfg).run(real_returns, target)
        assert res.turnover.sum() == pytest.approx(0.0)
        assert len(res.rebalance_dates) == 0

    def test_weights_sum_to_one_daily(self, real_returns):
        target = pd.Series(1 / len(TICKERS), index=TICKERS)
        res = BacktestEngine(PortfolioConfig()).run(real_returns, target)
        np.testing.assert_allclose(res.weights.sum(axis=1).to_numpy(), 1.0, atol=1e-12)

    def test_rebalance_fires_on_drift(self, real_returns):
        target = pd.Series(1 / len(TICKERS), index=TICKERS)
        cfg = PortfolioConfig(rebalance_band_pct=1.0)  # tight band ⇒ frequent trades
        res = BacktestEngine(cfg).run(real_returns, target)
        assert len(res.rebalance_dates) > 0
        assert res.turnover.sum() > 0

    def test_target_must_sum_to_one(self, real_returns):
        bad = pd.Series({t: 0.1 for t in TICKERS})  # sums to 0.4
        with pytest.raises(ValueError, match="sum to 1"):
            BacktestEngine().run(real_returns, bad)

    def test_rejects_nan(self, synth):
        r, target = synth
        r = r.copy()
        r.iloc[1, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            BacktestEngine().run(r, target)


# ─── Metrics ──────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_constant_return_series(self):
        r = pd.Series([0.001] * 252)
        m = performance_metrics(r, trading_days=252)
        assert m.ann_return == pytest.approx(0.252)
        assert m.ann_volatility == pytest.approx(0.0, abs=1e-12)
        assert m.sharpe == 0.0                 # guarded div-by-zero
        assert m.max_drawdown == pytest.approx(0.0)  # monotone up ⇒ no drawdown
        assert m.win_rate == pytest.approx(1.0)

    def test_drawdown_is_negative(self):
        r = pd.Series([0.1, 0.1, -0.5, 0.1])   # a real peak-to-trough dip
        m = performance_metrics(r)
        assert m.max_drawdown < 0
        assert -1.0 <= m.max_drawdown <= 0.0


# ─── Attribution ──────────────────────────────────────────────────────────────

class TestAttribution:
    def test_attribution_sums_to_gross(self, real_returns):
        target = pd.Series(1 / len(TICKERS), index=TICKERS)
        res = BacktestEngine(PortfolioConfig()).run(real_returns, target)
        attr = return_attribution(res, real_returns)
        assert attr.contribution.sum() == pytest.approx(attr.gross_return_arith, rel=1e-9)
        assert attr.net_return_arith == pytest.approx(
            attr.gross_return_arith - attr.total_cost, rel=1e-12
        )

    def test_zero_cost_net_equals_gross(self, real_returns):
        target = pd.Series(1 / len(TICKERS), index=TICKERS)
        cfg = PortfolioConfig(transaction_cost_bps=0, slippage_bps=0)
        res = BacktestEngine(cfg).run(real_returns, target)
        attr = return_attribution(res, real_returns)
        assert attr.total_cost == pytest.approx(0.0)
        assert attr.net_return_arith == pytest.approx(attr.gross_return_arith, rel=1e-12)
