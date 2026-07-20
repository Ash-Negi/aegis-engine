"""
Aegis Engine — Expected Returns Tests
=====================================
Run with: pytest tests/test_expected_returns.py -v

μ is the input MVO is most sensitive to, and shrinkage is what makes it
usable. These tests pin the shrinkage math (independent recompute of the
Bayes-Stein intensity) and the property that matters downstream: shrinkage
pulls the mean vector together, so the optimizer stops chasing spurious
cross-sectional differences.

──────────────────────────────────────────────────────────────────────────────
Review Log
──────────────────────────────────────────────────────────────────────────────
2026-07-20 — Week 3, expected returns

  test_bayes_stein_matches_reference
      Recomputes φ and μ_min via an explicit inverse (the estimator uses a
      linear solve) — an independent path to the same numbers.
  test_bayes_stein_pulls_toward_target / _reduces_dispersion
      The economic point: every shrunk mean sits between its raw value and
      μ_min, and the cross-sectional spread shrinks. If this regressed, the
      optimizer would be back to trusting noise.
  test_grand_mean_endpoints
      φ=0 returns raw means, φ=1 flattens to the grand mean — the two
      anchors of the linear shrink.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import pytest

from config import DataConfig, OptimizerConfig, TICKERS
from data.pipeline import DataPipeline
from covariance import ledoit_wolf_covariance
from optimizer import (
    sample_mean_returns,
    bayes_stein_shrinkage,
    grand_mean_shrinkage,
    estimate_expected_returns,
)


@pytest.fixture(scope="module")
def data():
    pipeline = DataPipeline(DataConfig())
    dataset = pipeline.run(TICKERS, use_cache=True)
    lr = pipeline.get_common_period(dataset)["log_returns"]
    cov = ledoit_wolf_covariance(lr)  # daily
    return lr, cov


class TestSampleMean:
    def test_matches_pandas(self, data):
        lr, _ = data
        pd.testing.assert_series_equal(sample_mean_returns(lr), lr.mean())

    def test_annualize(self, data):
        lr, _ = data
        daily = sample_mean_returns(lr, annualize=False)
        annual = sample_mean_returns(lr, annualize=True, trading_days=252)
        pd.testing.assert_series_equal(annual, daily * 252)


class TestBayesStein:
    def test_bayes_stein_matches_reference(self, data):
        lr, cov = data
        mean_daily = lr.mean()
        T = len(lr)
        shrunk, phi, mu_min = bayes_stein_shrinkage(mean_daily, cov, T)

        # Independent recompute with an explicit inverse.
        mu = mean_daily.to_numpy()
        S = cov.to_numpy()
        N = len(mu)
        Sinv = np.linalg.inv(S)
        ones = np.ones(N)
        mu_min_ref = (ones @ Sinv @ mu) / (ones @ Sinv @ ones)
        diff = mu - mu_min_ref * ones
        quad = diff @ Sinv @ diff
        phi_ref = (N + 2) / ((N + 2) + T * quad)
        shrunk_ref = (1 - phi_ref) * mu + phi_ref * mu_min_ref * ones

        np.testing.assert_allclose(mu_min, mu_min_ref, rtol=1e-9)
        np.testing.assert_allclose(phi, phi_ref, rtol=1e-9)
        np.testing.assert_allclose(shrunk.to_numpy(), shrunk_ref, rtol=1e-9)

    def test_phi_in_unit_interval(self, data):
        lr, cov = data
        _, phi, _ = bayes_stein_shrinkage(lr.mean(), cov, len(lr))
        assert 0.0 < phi <= 1.0

    def test_pulls_toward_target(self, data):
        """Each shrunk mean lies between its raw value and μ_min."""
        lr, cov = data
        raw = lr.mean()
        shrunk, _, mu_min = bayes_stein_shrinkage(raw, cov, len(lr))
        for t in TICKERS:
            lo, hi = sorted([raw[t], mu_min])
            assert lo - 1e-18 <= shrunk[t] <= hi + 1e-18

    def test_reduces_dispersion(self, data):
        """Cross-sectional spread of the mean vector shrinks."""
        lr, cov = data
        raw = lr.mean()
        shrunk, _, _ = bayes_stein_shrinkage(raw, cov, len(lr))
        assert shrunk.std() < raw.std()


class TestGrandMean:
    def test_grand_mean_endpoints(self):
        mean = pd.Series({"A": 0.10, "B": 0.20, "C": 0.30})
        raw, phi0, _ = grand_mean_shrinkage(mean, 0.0)
        pd.testing.assert_series_equal(raw, mean)
        assert phi0 == 0.0

        flat, phi1, grand = grand_mean_shrinkage(mean, 1.0)
        assert phi1 == 1.0
        np.testing.assert_allclose(flat.to_numpy(), [0.2, 0.2, 0.2])
        assert grand == pytest.approx(0.2)

    def test_rejects_bad_intensity(self):
        with pytest.raises(ValueError):
            grand_mean_shrinkage(pd.Series({"A": 0.1}), 1.5)


class TestEstimateExpectedReturns:
    def test_none_returns_raw(self, data):
        lr, cov = data
        cfg = OptimizerConfig(return_shrinkage="none", annualize=False)
        er = estimate_expected_returns(lr, cov, cfg)
        pd.testing.assert_series_equal(er.mu, lr.mean())
        assert er.shrinkage == 0.0

    def test_annualized_basis(self, data):
        lr, cov = data
        cfg = OptimizerConfig(return_shrinkage="none", annualize=True)
        er = estimate_expected_returns(lr, cov, cfg)
        pd.testing.assert_series_equal(er.mu, lr.mean() * 252)

    def test_bayes_stein_flows_through(self, data):
        lr, cov = data
        cfg = OptimizerConfig(return_shrinkage="bayes_stein", annualize=True)
        er = estimate_expected_returns(lr, cov, cfg)
        assert er.method == "bayes_stein"
        assert 0.0 < er.shrinkage <= 1.0
        # Annualized shrunk dispersion still below raw dispersion.
        assert er.mu.std() < er.raw_mu.std()
