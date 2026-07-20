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

from dataclasses import dataclass

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


# ─── EWMA (RiskMetrics) covariance ────────────────────────────────────────────

def ewma_covariance(
    returns: pd.DataFrame,
    config: CovarianceConfig | None = None,
    annualize: bool = False,
) -> pd.DataFrame:
    r"""
    Exponentially-weighted moving-average covariance (J.P. Morgan RiskMetrics).

        Σ = Σ_t  w_t · x_t x_tᵀ ,     w_t ∝ λ^{(T-1) - t} ,   Σ_t w_t = 1

    where t indexes observations oldest→newest, so the most recent day
    carries the largest weight and influence decays geometrically into the
    past. λ = 0.94 (the RiskMetrics daily default) gives a half-life of
    ln(0.5)/ln(0.94) ≈ 11.2 trading days: roughly, only the last ~two weeks
    dominate the estimate.

    Why this exists alongside the sample estimator:
        The sample covariance weights every day in the window equally — a
        return from 18 months ago counts as much as yesterday's. Markets
        are non-stationary (volatility clusters), so equal weighting blends
        calm and turbulent regimes into one blurred average. EWMA fixes
        that by *forgetting*: it tracks the current regime's covariance
        rather than the whole sample's. The cost is a shorter effective
        sample (≈ 1/(1-λ) ≈ 17 days of "real" information), which makes the
        estimate noisier — a different trade-off than Ledoit-Wolf's.

    Mean handling:
        RiskMetrics assumes a zero daily mean and forms the second moment
        E[x xᵀ] directly, WITHOUT demeaning (config.ewma_demean=False). At
        daily frequency the mean is tiny relative to the volatility, and
        the λ=0.94 calibration was derived under that assumption, so we
        follow it by default. Set ewma_demean=True to subtract the sample
        mean first (a genuine weighted covariance) for research.

    Args:
        returns:   clean returns frame (rows = dates, cols = tickers).
        config:    CovarianceConfig; reads ewma_lambda, ewma_demean, and
                   (if annualizing) trading_days_per_year.
        annualize: multiply by trading_days_per_year. Scalar multiple, so
                   it leaves conditioning and correlations unchanged.

    Returns:
        Labelled (N × N) covariance DataFrame, symmetric and PSD (a convex
        combination of rank-1 outer products x_t x_tᵀ is always PSD).
    """
    config = config or CovarianceConfig()
    lam = config.ewma_lambda
    if not 0.0 < lam < 1.0:
        raise ValueError(f"ewma_lambda must be in (0, 1), got {lam}")

    X, columns = _prepare(returns)
    T = X.shape[0]

    if config.ewma_demean:
        X = X - X.mean(axis=0, keepdims=True)

    # Weight for observation t (0 = oldest, T-1 = newest) is proportional to
    # λ^{(T-1)-t}: the newest row gets λ^0 = 1, each step back multiplies by
    # λ. Normalizing by the sum makes the weights a proper probability
    # distribution even though the geometric series is truncated at T terms.
    exponents = np.arange(T - 1, -1, -1)
    weights = (1.0 - lam) * lam**exponents
    weights /= weights.sum()

    # Σ = Xᵀ diag(w) X  accumulates the weighted outer products x_t x_tᵀ.
    cov = X.T @ (weights[:, None] * X)

    if annualize:
        cov = cov * config.trading_days_per_year

    return _as_frame(cov, columns)


# ─── Ledoit-Wolf shrinkage covariance ─────────────────────────────────────────

@dataclass
class LedoitWolfResult:
    """
    Full output of the Ledoit-Wolf estimator, so the report can show not
    just the shrunk matrix but *how much* was shrunk and toward what.

        covariance : the estimator's output Σ* = δ·target + (1-δ)·sample
        shrinkage  : δ ∈ [0, 1], the weight placed on the structured target
        target     : the shrinkage target μ·I (a labelled frame)
        sample     : the biased (1/T) sample covariance being shrunk
        mu         : μ = average variance = trace(sample)/N (target's scale)
    """
    covariance: pd.DataFrame
    shrinkage: float
    target: pd.DataFrame
    sample: pd.DataFrame
    mu: float


def ledoit_wolf_shrinkage(
    returns: pd.DataFrame,
    config: CovarianceConfig | None = None,
    annualize: bool = False,
) -> LedoitWolfResult:
    r"""
    Ledoit-Wolf (2004) shrinkage toward a scaled identity — the "well-
    conditioned estimator". This is Aegis's PRIMARY covariance estimator.

    The idea. The sample covariance S is unbiased but noisy; a structured
    target F is biased but stable. Neither is ideal alone, so form a convex
    combination that trades one against the other:

        Σ* = δ·F + (1 - δ)·S ,     F = μ·I ,     μ = trace(S)/N

    The target μ·I says "every asset has the same variance μ and zero
    correlation" — deliberately wrong, but perfectly conditioned (condition
    number 1). Shrinking S toward it pulls the extreme eigenvalues back in:
    the tiny ones (which S⁻¹ would explode) rise toward μ, the huge ones
    fall toward μ. That is exactly the fragility the optimizer suffers from,
    treated at the source.

    The magic is that δ is not a hand-tuned knob — Ledoit-Wolf derive the
    δ that minimizes expected squared error E‖Σ* − Σ_true‖² in closed form,
    estimated purely from the data:

        μ   = ⟨S, I⟩ = trace(S)/N                         (mean eigenvalue)
        d²  = ‖S − μI‖²                                   (how far S is from the sphere)
        b̄²  = (1/T²) Σ_k ‖xₖxₖᵀ − S‖²                     (sampling noise in S)
        b²  = min(b̄², d²) ,   a² = d² − b²
        δ   = b² / d²                                     (shrinkage intensity)

    using the normalized inner product ⟨A,B⟩ = trace(ABᵀ)/N. Intuitively δ
    is (noise in S) / (distance of S from the target): shrink hard when S is
    noisy and close to the target, barely at all when S is precise and the
    target is clearly wrong. Because b² ≤ d², δ ∈ [0, 1] and Σ* is a genuine
    convex combination — hence symmetric and PSD.

    Convention notes (documented because they matter for reproducibility):
      * S here is the biased 1/T maximum-likelihood covariance, not the
        1/(T-1) one sample_covariance() returns. The δ formula is derived
        for the 1/T version; the two differ only by a scalar T/(T-1), which
        does not change the condition number or the shrinkage direction.
      * Returns are demeaned by their sample mean before forming S.
      * We implement b̄² by the direct definition (average squared distance
        of each observation's outer product from S) rather than a shortcut,
        so the code reads like the formula. N=4, T≈490 makes cost a non-issue.

    Args:
        returns:   clean returns frame (rows = dates, cols = tickers).
        config:    CovarianceConfig; only trading_days_per_year is read
                   (for annualize). Ledoit-Wolf has no free parameters — δ
                   is data-determined, which is the whole point.
        annualize: multiply Σ*, target and sample by trading_days_per_year.
                   δ is scale-free and unchanged.

    Returns:
        LedoitWolfResult with the shrunk covariance and its provenance.
    """
    config = config or CovarianceConfig()
    X, columns = _prepare(returns)
    T, N = X.shape

    # Demean, then the biased (1/T) sample covariance the LW derivation uses.
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / T

    identity = np.eye(N)
    mu = np.trace(S) / N                       # ⟨S, I⟩ : mean eigenvalue / avg variance

    # d² = ‖S − μI‖², using the normalized norm ‖A‖² = trace(AAᵀ)/N.
    d2 = np.sum((S - mu * identity) ** 2) / N

    if d2 <= 0.0:
        # S already equals the target μI (e.g. a single asset, or perfectly
        # isotropic returns). Shrinkage is undefined and unnecessary — S is
        # already perfectly conditioned. Return it unshrunk.
        shrinkage = 0.0
        cov = S
    else:
        # b̄² = (1/T²) Σ_k ‖xₖxₖᵀ − S‖²  (same normalized norm).
        acc = 0.0
        for k in range(T):
            outer = np.outer(Xc[k], Xc[k])
            acc += np.sum((outer - S) ** 2)
        b2_bar = acc / (N * T**2)

        # b² is capped at d²: sampling noise cannot exceed the total
        # dispersion, and the cap guarantees δ ≤ 1.
        b2 = min(b2_bar, d2)
        shrinkage = b2 / d2
        cov = shrinkage * mu * identity + (1.0 - shrinkage) * S

    target = mu * identity

    if annualize:
        f = config.trading_days_per_year
        cov = cov * f
        S = S * f
        target = target * f
        mu = mu * f

    return LedoitWolfResult(
        covariance=_as_frame(cov, columns),
        shrinkage=float(shrinkage),
        target=_as_frame(target, columns),
        sample=_as_frame(S, columns),
        mu=float(mu),
    )


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    config: CovarianceConfig | None = None,
    annualize: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper returning just the shrunk covariance matrix, so
    Ledoit-Wolf honours the same (returns → Σ frame) contract as the other
    estimators. Use ledoit_wolf_shrinkage() when you also want δ and the
    decomposition for diagnostics.
    """
    return ledoit_wolf_shrinkage(returns, config, annualize).covariance
