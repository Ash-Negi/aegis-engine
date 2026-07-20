"""
Aegis Engine — Covariance Diagnostics Tests
===========================================
Tests for the conditioning and eigenstructure read-outs.

Run with: pytest tests/test_covariance_diagnostics.py -v

These diagnostics are the EVIDENCE behind the Week 2 thesis ("estimator
choice changes how safely Σ inverts"). If the evidence is miscomputed, the
whole argument is unsound — so the diagnostics get pinned against matrices
whose eigenstructure is known on paper, not just against library output.

──────────────────────────────────────────────────────────────────────────────
Review Log
──────────────────────────────────────────────────────────────────────────────
2026-07-20 — Week 2, diagnostics

  test_condition_number_known / test_eigen_known_diagonal
      diag([4,1]) and identity: condition number, variance explained, and
      participation ratio are all hand-computable. Anchors the code to the
      definitions rather than to numpy agreeing with itself.
  test_singular_matrix_is_inf
      A rank-deficient Σ (perfectly collinear assets) must report κ = +inf,
      not a huge finite number or a crash — that is the signal the optimizer
      needs to refuse to invert.
  test_eigenvectors_orthonormal / test_reconstructs_matrix
      Σ = VΛVᵀ with orthonormal V. If this breaks, "factor i" no longer
      means an uncorrelated risk portfolio and the interpretation collapses.
  test_variance_explained_partitions / test_participation_ratio_bounds
      variance_explained must sum to 1 and PR must lie in [1, N]; both are
      how the "how many real factors" claim is quantified.
  test_n_factors_threshold
      The hard count honours factor_variance_threshold (a 1%/3%/96% split
      collapses to one factor at the 5% threshold).
  test_rejects_non_symmetric / _non_square
      Guards the "this is really a covariance matrix" contract.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import pytest

from config import CovarianceConfig, DataConfig, TICKERS
from data.pipeline import DataPipeline
from covariance import (
    ledoit_wolf_covariance,
    condition_number,
    log_condition_number,
    eigen_analysis,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_cov() -> pd.DataFrame:
    """A real (Ledoit-Wolf) covariance to exercise diagnostics on live data."""
    pipeline = DataPipeline(DataConfig())
    dataset = pipeline.run(TICKERS, use_cache=True)
    log_returns = pipeline.get_common_period(dataset)["log_returns"]
    return ledoit_wolf_covariance(log_returns)


def _frame(matrix, labels) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=labels, columns=labels)


# ─── Condition number ─────────────────────────────────────────────────────────

class TestConditionNumber:

    def test_condition_number_known(self):
        """diag([4, 1]) ⇒ κ = 4, log₁₀κ = log₁₀4."""
        cov = _frame(np.diag([4.0, 1.0]), ["A", "B"])
        np.testing.assert_allclose(condition_number(cov), 4.0, rtol=1e-12)
        np.testing.assert_allclose(log_condition_number(cov), np.log10(4.0), rtol=1e-12)

    def test_identity_is_perfectly_conditioned(self):
        cov = _frame(np.eye(3), ["A", "B", "C"])
        np.testing.assert_allclose(condition_number(cov), 1.0, rtol=1e-12)
        np.testing.assert_allclose(log_condition_number(cov), 0.0, atol=1e-12)

    def test_singular_matrix_is_inf(self):
        """Perfectly collinear assets ⇒ λ_min = 0 ⇒ κ = +inf."""
        cov = _frame(np.array([[1.0, 1.0], [1.0, 1.0]]), ["A", "B"])
        assert condition_number(cov) == np.inf

    def test_matches_numpy_cond(self, real_cov):
        """On a real SPD matrix, κ must match numpy's 2-norm condition number."""
        np.testing.assert_allclose(
            condition_number(real_cov), np.linalg.cond(real_cov.to_numpy()), rtol=1e-9
        )


# ─── Eigenstructure ───────────────────────────────────────────────────────────

class TestEigenAnalysis:

    def test_eigen_known_diagonal(self):
        """diag([4,1]): variances 0.8/0.2, PR = 25/17, factors ordered."""
        cov = _frame(np.diag([4.0, 1.0]), ["A", "B"])
        ea = eigen_analysis(cov)

        np.testing.assert_allclose(ea.eigenvalues, [4.0, 1.0], rtol=1e-12)
        np.testing.assert_allclose(ea.variance_explained, [0.8, 0.2], rtol=1e-12)
        np.testing.assert_allclose(ea.cumulative_variance, [0.8, 1.0], rtol=1e-12)
        np.testing.assert_allclose(ea.participation_ratio, 25.0 / 17.0, rtol=1e-12)
        assert ea.n_factors == 2

    def test_identity_participation_is_n(self):
        """Perfectly even variance ⇒ participation ratio = N."""
        cov = _frame(np.eye(4), list("ABCD"))
        ea = eigen_analysis(cov)
        np.testing.assert_allclose(ea.participation_ratio, 4.0, rtol=1e-12)
        assert ea.n_factors == 4

    def test_eigenvalues_descending(self, real_cov):
        ea = eigen_analysis(real_cov)
        assert np.all(np.diff(ea.eigenvalues) <= 0), "eigenvalues not descending"

    def test_eigenvectors_orthonormal(self, real_cov):
        ea = eigen_analysis(real_cov)
        V = ea.eigenvectors
        np.testing.assert_allclose(V @ V.T, np.eye(V.shape[0]), atol=1e-10)

    def test_reconstructs_matrix(self, real_cov):
        """Σ = V diag(λ) Vᵀ must recover the original matrix."""
        ea = eigen_analysis(real_cov)
        rebuilt = ea.eigenvectors @ np.diag(ea.eigenvalues) @ ea.eigenvectors.T
        np.testing.assert_allclose(rebuilt, real_cov.to_numpy(), atol=1e-14)

    def test_variance_explained_partitions(self, real_cov):
        ea = eigen_analysis(real_cov)
        np.testing.assert_allclose(ea.variance_explained.sum(), 1.0, rtol=1e-12)
        np.testing.assert_allclose(ea.cumulative_variance[-1], 1.0, rtol=1e-12)

    def test_participation_ratio_bounds(self, real_cov):
        ea = eigen_analysis(real_cov)
        assert 1.0 <= ea.participation_ratio <= len(TICKERS) + 1e-9

    def test_n_factors_threshold(self):
        """A 96%/3%/1% variance split collapses to one factor at 5%."""
        cov = _frame(np.diag([0.96, 0.03, 0.01]), ["A", "B", "C"])
        ea = eigen_analysis(cov, CovarianceConfig(factor_variance_threshold=0.05))
        assert ea.n_factors == 1

    def test_summary_table_shape(self, real_cov):
        ea = eigen_analysis(real_cov)
        summ = ea.summary()
        assert list(summ.columns) == [
            "eigenvalue",
            "variance_explained",
            "cumulative_variance",
        ]
        assert len(summ) == len(TICKERS)


# ─── Input contract ───────────────────────────────────────────────────────────

class TestDiagnosticsContract:

    def test_rejects_non_symmetric(self):
        bad = _frame(np.array([[1.0, 0.5], [0.2, 1.0]]), ["A", "B"])
        with pytest.raises(ValueError, match="symmetric"):
            eigen_analysis(bad)

    def test_rejects_non_square(self):
        bad = pd.DataFrame(np.ones((2, 3)))
        with pytest.raises(ValueError, match="square"):
            condition_number(bad)
