"""
Aegis Engine — Constrained Optimizer Tests
==========================================
Run with: pytest tests/test_constrained.py -v

The numerical solver is only trustworthy if its output actually satisfies
the constraints and dominates in the right direction. These tests check the
three things a QP bug would violate: feasibility (long-only, caps, target
hit), and the ordering property that constraints can only raise the minimum
variance relative to the unconstrained closed-form solution.

──────────────────────────────────────────────────────────────────────────────
Review Log
──────────────────────────────────────────────────────────────────────────────
2026-07-20 — Week 3, constrained optimizer

  test_min_var_feasible / test_respects_sector_cap / test_respects_max_weight
      Feasibility of the solver output — no negative weights, sector sums
      under cap, per-asset ceiling honoured. A solver that silently returns
      an infeasible point is worse than none.
  test_constrained_min_var_geq_unconstrained
      Adding constraints cannot lower the minimum variance. Ties the
      numerical answer to the closed-form GMV as a lower bound.
  test_target_return_hit
      The equality constraint μᵀw = m is actually satisfied.
  test_frontier_feasible_everywhere
      Every point on the traced frontier is long-only, capped, and summed.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pytest

from config import DataConfig, OptimizerConfig, TICKERS
from data.pipeline import DataPipeline
from covariance import ledoit_wolf_covariance
from optimizer import (
    global_minimum_variance,
    portfolio_variance,
    portfolio_return,
    min_variance_portfolio,
    max_sharpe_portfolio,
    target_return_portfolio,
    efficient_frontier_constrained,
    SECTOR_MAP,
)

EQUITY = [t for t in TICKERS if SECTOR_MAP[t] == "equity"]


@pytest.fixture(scope="module")
def real_inputs():
    pipeline = DataPipeline(DataConfig())
    dataset = pipeline.run(TICKERS, use_cache=True)
    lr = pipeline.get_common_period(dataset)["log_returns"]
    mu = lr.mean() * 252
    cov = ledoit_wolf_covariance(lr) * 252
    return mu, cov


class TestFeasibility:
    def test_min_var_feasible(self, real_inputs):
        _, cov = real_inputs
        w = min_variance_portfolio(cov, OptimizerConfig())
        assert (w >= -1e-9).all(), "long-only violated"
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_respects_sector_cap(self, real_inputs):
        _, cov = real_inputs
        cfg = OptimizerConfig(sector_caps={"equity": 0.30})
        w = min_variance_portfolio(cov, cfg)
        assert w[EQUITY].sum() <= 0.30 + 1e-5

    def test_respects_max_weight(self, real_inputs):
        mu, cov = real_inputs
        cfg = OptimizerConfig(max_weight=0.40, sector_caps={})
        w = max_sharpe_portfolio(mu, cov, 0.05, cfg)
        assert (w <= 0.40 + 1e-5).all()

    def test_max_sharpe_feasible(self, real_inputs):
        mu, cov = real_inputs
        w = max_sharpe_portfolio(mu, cov, 0.05, OptimizerConfig())
        assert (w >= -1e-9).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert w[EQUITY].sum() <= 0.65 + 1e-5


class TestOrdering:
    def test_constrained_min_var_geq_unconstrained(self, real_inputs):
        """Constraints cannot reduce the minimum variance below closed-form GMV."""
        _, cov = real_inputs
        gmv_var = portfolio_variance(global_minimum_variance(cov), cov)
        con_var = portfolio_variance(min_variance_portfolio(cov, OptimizerConfig()), cov)
        assert con_var >= gmv_var - 1e-10


class TestTarget:
    def test_target_return_hit(self, real_inputs):
        mu, cov = real_inputs
        # a target comfortably inside the long-only feasible band
        target = 0.20
        w = target_return_portfolio(mu, cov, target, OptimizerConfig())
        assert portfolio_return(w, mu) == pytest.approx(target, abs=1e-5)
        assert (w >= -1e-9).all()


class TestFrontier:
    def test_frontier_feasible_everywhere(self, real_inputs):
        mu, cov = real_inputs
        cfg = OptimizerConfig(frontier_points=15)
        fr = efficient_frontier_constrained(mu, cov, cfg, risk_free_rate=0.05)
        assert len(fr.returns) >= 2
        W = fr.weights
        assert (W.to_numpy() >= -1e-6).all(), "a frontier point shorted"
        np.testing.assert_allclose(W.sum(axis=1).to_numpy(), 1.0, atol=1e-5)
        assert (W[EQUITY].sum(axis=1).to_numpy() <= 0.65 + 1e-5).all()
        # volatility should be non-decreasing as we climb toward higher return
        assert fr.volatilities[-1] >= fr.volatilities[0] - 1e-9
