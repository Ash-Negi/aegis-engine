"""
Aegis Engine — Covariance Estimators
====================================
Three ways to estimate the return covariance matrix Σ, in increasing order
of sophistication:

    sample_covariance      baseline — the maximum-likelihood / textbook Σ
    ewma_covariance        RiskMetrics exponential weighting (recency bias)
    ledoit_wolf_covariance shrinkage toward a structured target (primary)

Why three? Because Σ is inverted by the optimizer, and the sample estimator
— while unbiased — is *noisy* when the number of assets is not tiny relative
to the number of observations. That noise concentrates in the smallest
eigenvalues, which the inverse blows up. EWMA and Ledoit-Wolf are two
different answers to "how do I get a Σ the optimizer can safely invert?":
EWMA by trusting recent data more, Ledoit-Wolf by pulling the whole matrix
toward a stable, well-conditioned target. Week 2 builds all three so the
optimizer (Week 3) can be handed the estimator that behaves best on this
universe rather than a default.

Shared contract for every estimator:
    input   returns : pd.DataFrame, rows = dates, cols = tickers, NO NaNs
    output  Σ       : pd.DataFrame, index == columns == input columns,
                      symmetric, positive semi-definite
The column-order preservation is deliberate — see module docstring in
covariance/__init__.py.
"""

import numpy as np
import pandas as pd

from config import CovarianceConfig


# ─── Shared input handling ────────────────────────────────────────────────────

def _prepare(returns: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Validate a returns frame and extract a clean (T × N) matrix.

    Every estimator funnels through here so the NaN / shape / dtype
    contract is enforced in exactly one place. A covariance built on a
    frame with a stray NaN or a misordered column is the kind of silent
    bug that produces plausible-looking but wrong weights, so we fail
    loudly and early instead.

    Returns:
        X       : float64 ndarray, shape (T, N), guaranteed finite
        columns : list of ticker labels, in the frame's column order
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(
            f"returns must be a pandas DataFrame, got {type(returns).__name__}"
        )

    columns = list(returns.columns)
    X = returns.to_numpy(dtype=float)

    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError(f"returns must be a 2-D frame with ≥1 column, got shape {X.shape}")

    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).sum())
        raise ValueError(
            f"returns contains {n_bad} non-finite value(s) (NaN/inf). "
            "Estimators expect a clean common-period frame — call "
            "pipeline.get_common_period(dataset)['log_returns'] first."
        )

    return X, columns


def _as_frame(cov: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Wrap a raw covariance ndarray back into a labelled, symmetric frame."""
    # Force exact symmetry: floating-point accumulation can leave Σ and Σ.T
    # differing in the last bit, which trips strict symmetry checks and some
    # eigensolvers. Averaging with the transpose is the standard clean-up.
    cov = 0.5 * (cov + cov.T)
    return pd.DataFrame(cov, index=columns, columns=columns)


# ─── Sample covariance ────────────────────────────────────────────────────────

def sample_covariance(
    returns: pd.DataFrame,
    config: CovarianceConfig | None = None,
    annualize: bool = False,
) -> pd.DataFrame:
    r"""
    The textbook sample covariance matrix.

        Σ_ij = 1/(T - ddof) · Σ_t (r_it - r̄_i)(r_jt - r̄_j)

    with ddof = 1 (Bessel's correction) by default, so the estimator is
    unbiased for the population covariance. This is the maximum-likelihood
    estimate up to that correction and the natural baseline every other
    estimator is compared against.

    Its weakness — the entire motivation for EWMA and Ledoit-Wolf — is
    estimation noise. With N assets there are N(N+1)/2 free parameters to
    estimate from T observations; when T is not hugely larger than N, the
    smallest eigenvalues of Σ are estimated badly, and the optimizer's
    Σ⁻¹ amplifies exactly those errors. On this 4-asset universe with
    ~490 days that is mild, but the QQQ/VXUS equity pair is correlated
    enough to still make the smallest eigenvalue the fragile one — which
    is what the Week 2 conditioning diagnostics are there to quantify.

    Args:
        returns:   clean returns frame (rows = dates, cols = tickers).
        config:    CovarianceConfig; only sample_ddof and (if annualizing)
                   trading_days_per_year are read. Defaults if None.
        annualize: multiply by trading_days_per_year to report Σ on an
                   annual basis. Note this is a scalar multiple, so it does
                   NOT change the condition number or correlation structure.

    Returns:
        Labelled (N × N) covariance DataFrame, symmetric and PSD.
    """
    config = config or CovarianceConfig()
    X, columns = _prepare(returns)

    # np.cov expects variables in rows, so transpose the (T × N) matrix.
    # rowvar=False keeps variables in columns and is equivalent; we pass the
    # transpose explicitly to make the ddof normalization unambiguous.
    cov = np.cov(X, rowvar=False, ddof=config.sample_ddof)
    # np.cov collapses to a 0-d array for a single column; re-expand so the
    # single-asset case still returns a 1×1 matrix.
    cov = np.atleast_2d(cov)

    if annualize:
        cov = cov * config.trading_days_per_year

    return _as_frame(cov, columns)
