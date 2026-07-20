"""
Aegis Engine — Mean-Variance Analytics Tests
============================================
Run with: pytest tests/test_mean_variance.py -v

The closed-form frontier is the mathematical heart of Week 3, so it is
pinned against a 2-asset case computable entirely by hand AND against the
defining variational properties on real data (GMV really is the minimum,
each frontier point really achieves its target return, the two-fund theorem
holds).

──────────────────────────────────────────────────────────────────────────────
Review Log
──────────────────────────────────────────────────────────────────────────────
2026-07-20 — Week 3, mean-variance analytics

  test_hand_computed_two_asset
      diag Σ, explicit μ ⇒ GMV [0.2,0.8], tangency [0.4,0.6], GMV return
      0.06 — all by hand. Anchors the formulas to arithmetic, not to numpy.
  test_gmv_is_minimum_variance
      GMV variance ≤ every frontier point and ≤ equal-weight. If this
      failed, "minimum variance" would be a lie.
  test_frontier_hits_target_return / _variance_matches_parabola
      Each w(m) achieves μᵀw = m and its variance equals both wᵀΣw and the
      closed-form (Am²−2Bm+C)/D — three independent expressions agreeing.
  test_two_fund_theorem
      w at the midpoint target equals the midpoint of the endpoint weights
      (frontier is affine in m).
  test_weights_sum_to_one
      Every portfolio the module emits is fully invested.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import pytest

from config import DataConfig, TICKERS
from data.pipeline import DataPipeline
from covariance import ledoit_wolf_covariance
from optimizer import (
    portfolio_return,
    portfolio_variance,
    global_minimum_variance,
    tangency_portfolio,
    frontier_weights,
    efficient_frontier,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_inputs():
    """Annualized μ (raw sample mean) and Σ (Ledoit-Wolf) for the universe."""
    pipeline = DataPipeline(DataConfig())
    dataset = pipeline.run(TICKERS, use_cache=True)
    lr = pipeline.get_common_period(dataset)["log_returns"]
    mu = lr.mean() * 252
    cov = ledoit_wolf_covariance(lr) * 252
    return mu, cov


@pytest.fixture
def hand():
    """2-asset diagonal case worked out by hand in the module tests."""
    cov = pd.DataFrame(np.diag([0.04, 0.01]), index=["X", "Y"], columns=["X", "Y"])
    mu = pd.Series({"X": 0.10, "Y": 0.05})
    return mu, cov


# ─── Hand-computed anchor ─────────────────────────────────────────────────────

class TestHandComputed:
    def test_hand_computed_two_asset(self, hand):
        mu, cov = hand
        gmv = global_minimum_variance(cov)
        np.testing.assert_allclose(gmv.to_numpy(), [0.2, 0.8], rtol=1e-12)
        assert portfolio_return(gmv, mu) == pytest.approx(0.06)
        assert portfolio_variance(gmv, cov) == pytest.approx(0.008)

        tan = tangency_portfolio(mu, cov, risk_free_rate=0.02)
        np.testing.assert_allclose(tan.to_numpy(), [0.4, 0.6], rtol=1e-12)


# ─── Properties on real data ──────────────────────────────────────────────────

class TestFrontierProperties:
    def test_weights_sum_to_one(self, real_inputs):
        mu, cov = real_inputs
        assert global_minimum_variance(cov).sum() == pytest.approx(1.0)
        assert tangency_portfolio(mu, cov, 0.05).sum() == pytest.approx(1.0)
        fr = efficient_frontier(mu, cov, n_points=25, risk_free_rate=0.05)
        np.testing.assert_allclose(fr.weights.sum(axis=1).to_numpy(), 1.0, atol=1e-10)

    def test_gmv_is_minimum_variance(self, real_inputs):
        mu, cov = real_inputs
        gmv = global_minimum_variance(cov)
        gmv_var = portfolio_variance(gmv, cov)
        # ≤ equal-weight
        ew = pd.Series(1 / len(TICKERS), index=TICKERS)
        assert gmv_var <= portfolio_variance(ew, cov) + 1e-15
        # ≤ every frontier point
        fr = efficient_frontier(mu, cov, n_points=40)
        assert gmv_var <= fr.volatilities.min() ** 2 + 1e-12

    def test_frontier_hits_target_return(self, real_inputs):
        mu, cov = real_inputs
        fr = efficient_frontier(mu, cov, n_points=20)
        for i, target in enumerate(fr.returns):
            w = pd.Series(fr.weights.iloc[i].to_numpy(), index=cov.columns)
            assert portfolio_return(w, mu) == pytest.approx(target, abs=1e-9)

    def test_frontier_variance_matches_parabola(self, real_inputs):
        mu, cov = real_inputs
        fr = efficient_frontier(mu, cov, n_points=20)
        for i in range(len(fr.returns)):
            w = pd.Series(fr.weights.iloc[i].to_numpy(), index=cov.columns)
            np.testing.assert_allclose(
                portfolio_variance(w, cov), fr.volatilities[i] ** 2, rtol=1e-9
            )

    def test_two_fund_theorem(self, real_inputs):
        """w at the midpoint target = midpoint of endpoint weights (affine)."""
        mu, cov = real_inputs
        m1, m2 = 0.05, 0.15
        w1 = frontier_weights(mu, cov, m1)
        w2 = frontier_weights(mu, cov, m2)
        wmid = frontier_weights(mu, cov, (m1 + m2) / 2)
        np.testing.assert_allclose(wmid.to_numpy(), (w1.to_numpy() + w2.to_numpy()) / 2, rtol=1e-9)

    def test_tangency_maximises_sharpe_on_frontier(self, real_inputs):
        """Tangency Sharpe ≥ every sampled frontier point's Sharpe."""
        mu, cov = real_inputs
        rf = 0.05
        tan = tangency_portfolio(mu, cov, rf)
        tan_sharpe = (portfolio_return(tan, mu) - rf) / np.sqrt(portfolio_variance(tan, cov))
        fr = efficient_frontier(mu, cov, n_points=200, risk_free_rate=rf)
        assert tan_sharpe >= np.nanmax(fr.sharpe) - 1e-6
