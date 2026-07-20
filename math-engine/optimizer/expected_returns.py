"""
Aegis Engine — Expected Returns
===============================
Estimating the expected-return vector μ for mean-variance optimization.

Why this is its own module, separate from the covariance layer:
    MVO weights are w ∝ Σ⁻¹μ. The optimizer is far MORE sensitive to errors
    in μ than in Σ (Chopra & Ziemba 1993: errors in means are ~10× as
    costly as errors in variances). And μ is exactly the input we estimate
    worst — a sample mean has standard error σ/√T, so over a 2-year window
    the mean is buried in noise. Feeding raw historical means into MVO is
    the classic error that produces absurd, concentrated, whipsawing
    portfolios. This module's job is to produce a μ the optimizer can trust,
    which means SHRINKING the raw estimate toward something stable.

The Week 1 review flagged this concretely (landmine 3): every asset had
large positive in-sample returns, so vanilla MVO would pile into gold. The
shrinkage here is the antidote.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExpectedReturns:
    """
    The expected-return estimate plus its provenance, so the optimizer and
    report can show how much the raw means were shrunk and toward what.

        mu         estimated expected returns (annualized iff annualize=True)
        method     "none" | "bayes_stein" | "grand_mean"
        shrinkage  intensity φ ∈ [0,1] actually applied (0 ⇒ raw means)
        raw_mu     the unshrunk sample mean, same basis as mu
        target     the scalar return everything was shrunk toward, same basis
    """
    mu: pd.Series
    method: str
    shrinkage: float
    raw_mu: pd.Series
    target: float


def sample_mean_returns(
    log_returns: pd.DataFrame,
    annualize: bool = False,
    trading_days: int = 252,
) -> pd.Series:
    r"""
    The raw sample mean of daily log returns, μ̂_i = (1/T) Σ_t r_it.

    Annualized by ×252 (means scale linearly with horizon). Note this is a
    LOG expected return — the geometric growth rate — kept consistent with
    the covariance layer, which is also estimated on log returns, so μ and
    Σ live in the same space.
    """
    mu = log_returns.mean()
    if annualize:
        mu = mu * trading_days
    return mu


def bayes_stein_shrinkage(
    mean_daily: pd.Series,
    cov_daily: pd.DataFrame,
    n_obs: int,
) -> tuple[pd.Series, float, float]:
    r"""
    Jorion (1986) Bayes-Stein shrinkage of the mean vector.

    Shrinks the noisy sample mean μ̂ toward a single common target — the
    expected return of the global-minimum-variance portfolio, μ_min — which
    is the most stable point on the return axis (it depends only on Σ, not
    on the fragile mean estimates):

        μ_BS = (1 − φ)·μ̂ + φ·μ_min·𝟙

    with, for N assets and T observations, using x = Σ⁻¹𝟙 and Σ⁻¹ applied
    via a linear solve (never an explicit inverse — Σ can be ill-conditioned):

        μ_min = (𝟙ᵀΣ⁻¹μ̂) / (𝟙ᵀΣ⁻¹𝟙)
        φ     = (N + 2) / ( (N + 2) + T·(μ̂ − μ_min𝟙)ᵀ Σ⁻¹ (μ̂ − μ_min𝟙) )

    Read φ as: shrink hard (φ→1) when the assets' mean returns are close
    together relative to their covariance (the spread is probably noise);
    barely shrink (φ→0) when the means are strongly separated (the signal is
    real). Because the quadratic form is ≥ 0, φ ∈ (0, 1].

    All inputs must be on the SAME (daily) basis; annualize afterwards.

    Returns:
        (shrunk_mean_daily, φ, μ_min)   — the last two for auditing/reporting.
    """
    mu = mean_daily.to_numpy(dtype=float)
    Sigma = cov_daily.to_numpy(dtype=float)
    N = len(mu)
    ones = np.ones(N)

    # Σ⁻¹ applied by solving, for numerical stability on ill-conditioned Σ.
    sig_inv_ones = np.linalg.solve(Sigma, ones)    # Σ⁻¹𝟙
    sig_inv_mu = np.linalg.solve(Sigma, mu)        # Σ⁻¹μ̂

    mu_min = float((ones @ sig_inv_mu) / (ones @ sig_inv_ones))
    diff = mu - mu_min * ones
    quad = float(diff @ np.linalg.solve(Sigma, diff))   # (μ̂−μ_min𝟙)ᵀΣ⁻¹(…)

    phi = (N + 2) / ((N + 2) + n_obs * quad)
    phi = float(np.clip(phi, 0.0, 1.0))

    shrunk = (1.0 - phi) * mu + phi * mu_min * ones
    return pd.Series(shrunk, index=mean_daily.index), phi, mu_min


def grand_mean_shrinkage(
    mean: pd.Series,
    intensity: float,
) -> tuple[pd.Series, float, float]:
    r"""
    Simple linear shrinkage toward the cross-sectional average return.

        μ_shrunk = (1 − φ)·μ̂ + φ·μ̄·𝟙,     μ̄ = mean_i(μ̂_i)

    A fixed-intensity alternative to Bayes-Stein for when you want a knob
    you set by hand rather than one the data chooses. Basis-agnostic (works
    on daily or annualized means since it is purely linear).

    Returns:
        (shrunk_mean, φ, grand_mean)
    """
    if not 0.0 <= intensity <= 1.0:
        raise ValueError(f"intensity must be in [0, 1], got {intensity}")
    grand = float(mean.mean())
    shrunk = (1.0 - intensity) * mean + intensity * grand
    return shrunk, float(intensity), grand


def estimate_expected_returns(log_returns, cov_daily, config) -> ExpectedReturns:
    """
    High-level entry point: compute μ per the configured shrinkage method.

    Shrinkage is always computed on the DAILY basis (Bayes-Stein's intensity
    depends on T and the daily covariance), then annualized for reporting if
    config.annualize is set.

    Args:
        log_returns: clean common-period log returns (rows=dates, cols=tickers).
        cov_daily:   DAILY covariance frame (from the covariance layer) —
                     used by Bayes-Stein for the μ_min target and intensity.
        config:      OptimizerConfig; reads return_shrinkage, grand_mean_intensity,
                     annualize, trading_days_per_year.
    """
    mean_daily = sample_mean_returns(log_returns, annualize=False)
    T = len(log_returns)

    if config.return_shrinkage == "bayes_stein":
        shrunk_daily, phi, target_daily = bayes_stein_shrinkage(mean_daily, cov_daily, T)
    elif config.return_shrinkage == "grand_mean":
        shrunk_daily, phi, target_daily = grand_mean_shrinkage(
            mean_daily, config.grand_mean_intensity
        )
    elif config.return_shrinkage == "none":
        shrunk_daily, phi, target_daily = mean_daily.copy(), 0.0, float(mean_daily.mean())
    else:
        raise ValueError(f"unknown return_shrinkage: {config.return_shrinkage!r}")

    if config.annualize:
        f = config.trading_days_per_year
        mu = shrunk_daily * f
        raw = mean_daily * f
        target = target_daily * f
    else:
        mu, raw, target = shrunk_daily, mean_daily, target_daily

    return ExpectedReturns(
        mu=mu,
        method=config.return_shrinkage,
        shrinkage=phi,
        raw_mu=raw,
        target=float(target),
    )
