"""
Aegis Engine — Covariance Estimator Tests
=========================================
Tests that verify the Week 2 covariance estimators are mathematically
correct and honour the shared contract (symmetric, PSD, label-preserving,
NaN-rejecting).

Run with: pytest tests/test_covariance.py -v

Why test covariance estimators this hard?
    Σ is inverted by the optimizer. An error in Σ does not stay the size
    it started — Σ⁻¹ magnifies error along the smallest eigenvalue, which
    is exactly the direction least well estimated. A covariance bug is
    therefore a leveraged bug. These tests pin the properties every
    downstream module (optimizer, risk attribution) silently assumes.

──────────────────────────────────────────────────────────────────────────────
Review Log
──────────────────────────────────────────────────────────────────────────────
A running record of test changes, newest on top. Keep entries terse: name
the test, say WHAT invariant it protects, and WHY a bug there would be easy
to miss.

2026-07-20 — Week 2, EWMA (RiskMetrics)

  test_ewma_hand_computed
      T=2, λ=0.5 fixture where the normalized weights are exactly
      [1/3, 2/3]. Pins BOTH the geometric weighting and the
      newest-gets-most-weight orientation — a reversed weight vector would
      still be symmetric/PSD and pass every structural test, so this is the
      one that catches an off-by-one or flipped-order bug.
  test_ewma_limit_lambda_to_one_is_biased_sample
      As λ→1 the weights become uniform, so EWMA(demean) must collapse to
      the 1/T (ddof=0) sample covariance. Ties the exotic estimator back to
      a known quantity at its boundary.
  test_ewma_symmetric / _psd / _preserves_order / _annualize / _rejects_bad_lambda
      Same structural contract as the sample estimator, re-checked because
      EWMA builds Σ by a completely different path (weighted outer products).

2026-07-20 — Week 2, sample covariance

  test_matches_pandas_cov
      Pins sample_covariance to the pandas .cov(ddof=1) reference so a
      future refactor (e.g. switching to a manual accumulation) can't
      silently change the normalization.
  test_known_2x2_hand_computed
      A tiny fixture whose covariance is computable by hand, so the test
      does not merely check "agrees with numpy" but "agrees with the
      textbook formula". Guards against both implementations sharing a bug.
  test_symmetric / test_positive_semidefinite
      The two structural promises the optimizer relies on. PSD in
      particular: a negative eigenvalue would let the optimizer find a
      "negative variance" portfolio and blow up.
  test_diagonal_matches_variances
      Σ_ii must equal Var(asset_i). Cheap cross-check that catches a
      transpose or normalization bug that off-diagonal tests might miss.
  test_preserves_column_order
      Σ's row/col order must equal the input's. yfinance sorts columns
      alphabetically; if ordering ever regresses, Σ lines the wrong asset
      to the wrong row and every optimizer weight is mislabelled silently.
  test_annualize_scales_cleanly
      Annualizing is a scalar (×252). Verifies it neither reorders nor
      distorts — and by construction leaves correlations unchanged.
  test_rejects_nan / test_rejects_non_dataframe
      The _prepare contract. A stray NaN must fail loudly, not propagate
      into a plausible-but-wrong matrix.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import pytest

from config import CovarianceConfig, DataConfig, TICKERS
from data.pipeline import DataPipeline
from covariance import sample_covariance, ewma_covariance


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def log_returns() -> pd.DataFrame:
    """Common-period log returns for the real universe (cached fetch)."""
    pipeline = DataPipeline(DataConfig())
    dataset = pipeline.run(TICKERS, use_cache=True)
    return pipeline.get_common_period(dataset)["log_returns"]


@pytest.fixture
def known_returns() -> pd.DataFrame:
    """
    A hand-checkable fixture. Both columns have mean exactly 0, and B = 2·A,
    so the covariance is computable on paper:

        Var(A)  = Σ a_t² / (T-1)
        Cov(A,B)= Σ a_t·b_t / (T-1) = 2·Var(A)
        Var(B)  = 4·Var(A)
        Corr(A,B) = 1

    with a_t = [0.01, -0.01, 0.02, -0.02].
    """
    a = np.array([0.01, -0.01, 0.02, -0.02])
    return pd.DataFrame({"A": a, "B": 2 * a})


# ─── Sample covariance ────────────────────────────────────────────────────────

class TestSampleCovariance:
    """Verify the baseline estimator against the textbook formula."""

    def test_matches_pandas_cov(self, log_returns):
        """sample_covariance == pandas .cov() with ddof=1 (same reference)."""
        got = sample_covariance(log_returns)
        expected = log_returns.cov()  # pandas default is ddof=1
        pd.testing.assert_frame_equal(got, expected, atol=1e-15)

    def test_known_2x2_hand_computed(self, known_returns):
        """Match values derived by hand, not just by another library."""
        a = np.array([0.01, -0.01, 0.02, -0.02])
        var_a = np.sum(a**2) / (len(a) - 1)  # mean is 0 by construction

        cov = sample_covariance(known_returns)

        np.testing.assert_allclose(cov.loc["A", "A"], var_a, rtol=1e-12)
        np.testing.assert_allclose(cov.loc["B", "B"], 4 * var_a, rtol=1e-12)
        np.testing.assert_allclose(cov.loc["A", "B"], 2 * var_a, rtol=1e-12)

        # Perfectly collinear ⇒ correlation 1 ⇒ singular Σ (a rank-1 matrix).
        corr = cov.loc["A", "B"] / np.sqrt(cov.loc["A", "A"] * cov.loc["B", "B"])
        np.testing.assert_allclose(corr, 1.0, rtol=1e-12)

    def test_symmetric(self, log_returns):
        cov = sample_covariance(log_returns).to_numpy()
        np.testing.assert_allclose(cov, cov.T, atol=1e-18)

    def test_positive_semidefinite(self, log_returns):
        """All eigenvalues ≥ 0 (up to floating-point noise)."""
        cov = sample_covariance(log_returns).to_numpy()
        eigenvalues = np.linalg.eigvalsh(cov)
        assert eigenvalues.min() > -1e-12, (
            f"Σ has a negative eigenvalue {eigenvalues.min():.2e}; not PSD"
        )

    def test_diagonal_matches_variances(self, log_returns):
        """Σ_ii must equal the per-asset variance (ddof=1)."""
        cov = sample_covariance(log_returns)
        for ticker in TICKERS:
            expected = log_returns[ticker].var(ddof=1)
            np.testing.assert_allclose(cov.loc[ticker, ticker], expected, rtol=1e-12)

    def test_preserves_column_order(self, log_returns):
        cov = sample_covariance(log_returns)
        assert list(cov.index) == list(TICKERS)
        assert list(cov.columns) == list(TICKERS)

    def test_annualize_scales_cleanly(self, log_returns):
        """Annualized Σ = 252 · daily Σ, order preserved, corr unchanged."""
        config = CovarianceConfig()
        daily = sample_covariance(log_returns, config)
        annual = sample_covariance(log_returns, config, annualize=True)

        np.testing.assert_allclose(
            annual.to_numpy(),
            daily.to_numpy() * config.trading_days_per_year,
            rtol=1e-12,
        )
        assert list(annual.columns) == list(TICKERS)

    def test_rejects_nan(self):
        bad = pd.DataFrame({"A": [0.01, np.nan, 0.02], "B": [0.0, 0.01, -0.01]})
        with pytest.raises(ValueError, match="non-finite"):
            sample_covariance(bad)

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            sample_covariance(np.array([[0.1, 0.2], [0.3, 0.4]]))


# ─── EWMA (RiskMetrics) ───────────────────────────────────────────────────────

class TestEWMACovariance:
    """Verify the exponentially-weighted estimator."""

    def test_ewma_hand_computed(self):
        """
        T=2, λ=0.5, no demean. Weights: raw = 0.5·[0.5, 1] = [0.25, 0.5],
        normalized = [1/3, 2/3]. Σ = (1/3)x₀x₀ᵀ + (2/3)x₁x₁ᵀ.
        """
        df = pd.DataFrame({"A": [0.02, -0.04], "B": [0.01, 0.03]})
        config = CovarianceConfig(ewma_lambda=0.5, ewma_demean=False)
        cov = ewma_covariance(df, config)

        w = np.array([1 / 3, 2 / 3])
        a = df["A"].to_numpy()
        b = df["B"].to_numpy()
        exp_aa = np.sum(w * a * a)
        exp_bb = np.sum(w * b * b)
        exp_ab = np.sum(w * a * b)

        np.testing.assert_allclose(cov.loc["A", "A"], exp_aa, rtol=1e-12)
        np.testing.assert_allclose(cov.loc["B", "B"], exp_bb, rtol=1e-12)
        np.testing.assert_allclose(cov.loc["A", "B"], exp_ab, rtol=1e-12)
        # Sanity on the concrete numbers: ΣAA = 0.0012 exactly.
        np.testing.assert_allclose(cov.loc["A", "A"], 0.0012, rtol=1e-12)

    def test_ewma_limit_lambda_to_one_is_biased_sample(self, log_returns):
        """As λ→1 the weights go uniform ⇒ EWMA(demean) → 1/T sample cov."""
        config = CovarianceConfig(ewma_lambda=1 - 1e-8, ewma_demean=True)
        cov = ewma_covariance(log_returns, config).to_numpy()
        biased_sample = np.cov(log_returns.to_numpy(), rowvar=False, ddof=0)
        np.testing.assert_allclose(cov, biased_sample, rtol=1e-4)

    def test_ewma_symmetric(self, log_returns):
        cov = ewma_covariance(log_returns).to_numpy()
        np.testing.assert_allclose(cov, cov.T, atol=1e-18)

    def test_ewma_psd(self, log_returns):
        """A convex combination of outer products x·xᵀ is always PSD."""
        cov = ewma_covariance(log_returns).to_numpy()
        assert np.linalg.eigvalsh(cov).min() > -1e-12

    def test_ewma_preserves_order(self, log_returns):
        cov = ewma_covariance(log_returns)
        assert list(cov.index) == list(TICKERS)
        assert list(cov.columns) == list(TICKERS)

    def test_ewma_annualize_scales(self, log_returns):
        config = CovarianceConfig()
        daily = ewma_covariance(log_returns, config).to_numpy()
        annual = ewma_covariance(log_returns, config, annualize=True).to_numpy()
        np.testing.assert_allclose(annual, daily * config.trading_days_per_year, rtol=1e-12)

    def test_ewma_rejects_bad_lambda(self, log_returns):
        for bad in (0.0, 1.0, 1.5, -0.1):
            with pytest.raises(ValueError, match="ewma_lambda"):
                ewma_covariance(log_returns, CovarianceConfig(ewma_lambda=bad))
