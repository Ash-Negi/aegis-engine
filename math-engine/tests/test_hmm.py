"""
Aegis Engine — Gaussian HMM Tests
=================================
Run with: pytest tests/test_hmm.py -v

The HMM is validated the way any EM implementation should be: the likelihood
must not go DOWN (the defining guarantee of EM), and on data generated from a
known model it must recover that model. Both are checked here, plus the
inference contracts (Viterbi shape, posterior normalisation, determinism).

──────────────────────────────────────────────────────────────────────────────
Review Log
──────────────────────────────────────────────────────────────────────────────
2026-07-20 — Phase 2, Gaussian HMM

  test_em_monotonic
      With covariance_reg=0 (exact EM) the log-likelihood is non-decreasing
      every iteration — the property that proves the E/M steps are consistent.
  test_recovers_states / test_recovers_parameters
      On data drawn from a known 2-state HMM, decoded states match truth
      (up to label swap) and fitted μ/σ match the generators. This is the
      end-to-end correctness check.
  test_predict_proba_normalised / test_viterbi_range / test_reproducible
      Inference contracts: γ rows sum to 1, Viterbi returns valid state
      indices, and a fixed seed is deterministic.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pytest

from signals.hmm import GaussianHMM


def _generate(seed=0, T=1500):
    """Draw a sequence from a known persistent 2-state Gaussian HMM."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.97, 0.03], [0.06, 0.94]])
    means = [0.0008, -0.0010]
    sds = [0.006, 0.020]
    z = np.zeros(T, dtype=int)
    x = np.zeros(T)
    for t in range(T):
        if t > 0:
            z[t] = rng.choice(2, p=A[z[t - 1]])
        x[t] = rng.normal(means[z[t]], sds[z[t]])
    return x, z, means, sds


class TestEM:
    def test_em_monotonic(self):
        """Exact EM (no covariance reg) never decreases the log-likelihood."""
        x, *_ = _generate()
        m = GaussianHMM(2, covariance_reg=0.0, random_state=1).fit(x)
        ll = np.array(m.loglikelihood_history_)
        assert np.all(np.diff(ll) >= -1e-9), "EM log-likelihood decreased"

    def test_recovers_states(self):
        x, z, *_ = _generate()
        m = GaussianHMM(2, random_state=1).fit(x)
        pred = m.predict(x)
        acc = max((pred == z).mean(), (pred == (1 - z)).mean())
        assert acc > 0.85, f"state recovery only {acc:.2f}"

    def test_recovers_parameters(self):
        x, z, means, sds = _generate()
        m = GaussianHMM(2, random_state=1).fit(x)
        fitted_means = np.sort(m.params_.means.ravel())
        fitted_sds = np.sort(np.sqrt(m.params_.covars.ravel()))
        # Means of a high-vol state are inherently hard to pin down: the
        # standard error is σ/√n ≈ 0.02/√400 ≈ 1e-3, so the tolerance is set
        # to a couple of standard errors, not to machine precision.
        np.testing.assert_allclose(fitted_means, np.sort(means), atol=1.5e-3)
        # Volatilities, by contrast, are estimated well and pinned tighter.
        np.testing.assert_allclose(fitted_sds, np.sort(sds), rtol=0.15)


class TestInference:
    def test_predict_proba_normalised(self):
        x, *_ = _generate()
        m = GaussianHMM(2, random_state=1).fit(x)
        pp = m.predict_proba(x)
        assert pp.shape == (len(x), 2)
        np.testing.assert_allclose(pp.sum(axis=1), 1.0, atol=1e-9)

    def test_viterbi_range(self):
        x, *_ = _generate()
        m = GaussianHMM(3, random_state=1).fit(x)
        states = m.predict(x)
        assert states.shape == (len(x),)
        assert set(np.unique(states)).issubset({0, 1, 2})

    def test_reproducible(self):
        x, *_ = _generate()
        a = GaussianHMM(2, random_state=7).fit(x).predict(x)
        b = GaussianHMM(2, random_state=7).fit(x).predict(x)
        np.testing.assert_array_equal(a, b)

    def test_score_finite(self):
        x, *_ = _generate()
        m = GaussianHMM(2, random_state=1).fit(x)
        assert np.isfinite(m.score(x))
