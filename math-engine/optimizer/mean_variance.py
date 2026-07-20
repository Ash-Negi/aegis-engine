"""
Aegis Engine — Mean-Variance Analytics (closed form)
====================================================
The textbook mean-variance results, derived with Lagrange multipliers and
implemented directly. These are the *unconstrained* solutions (weights may
be negative — i.e. short positions are allowed); the long-only versions
live in optimizer/constrained.py.

Everything hangs off the same object, Σ⁻¹, applied through np.linalg.solve
rather than an explicit inverse (Σ can be ill-conditioned — Week 1 landmine
2 — and solving is both faster and more stable).

The three results:

  Global minimum variance (GMV)
      Minimise wᵀΣw subject to 𝟙ᵀw = 1. One Lagrange multiplier gives
          w_gmv = Σ⁻¹𝟙 / (𝟙ᵀΣ⁻¹𝟙).
      Depends only on Σ — no return forecast — which is why it is the most
      robust portfolio on the frontier and the anchor Bayes-Stein shrinks
      toward.

  Efficient frontier
      Minimise wᵀΣw subject to 𝟙ᵀw = 1 AND μᵀw = m (a target return). Two
      multipliers give the affine solution w(m) = g + h·m, so the whole
      frontier is a straight line in weight space (the two-fund theorem).
      Variance traces a parabola σ²(m) = (A m² − 2B m + C)/D.

  Tangency (maximum Sharpe)
      With a risk-free rate r_f, the portfolio maximising the Sharpe ratio
      (wᵀμ − r_f)/√(wᵀΣw) is
          w_tan = Σ⁻¹(μ − r_f𝟙) / (𝟙ᵀΣ⁻¹(μ − r_f𝟙)).

Basis note: pass μ, Σ and r_f on a consistent basis (all daily or all
annualised). The WEIGHTS are invariant to that choice; only the reported
return/vol numbers change. The report passes annualised inputs.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ─── Portfolio functionals ────────────────────────────────────────────────────

def portfolio_return(weights: pd.Series, mu: pd.Series) -> float:
    """Expected portfolio return wᵀμ (aligns on the shared ticker index)."""
    w, m = weights.align(mu, join="inner")
    return float(w @ m)


def portfolio_variance(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Portfolio variance wᵀΣw."""
    w = weights.reindex(cov.columns).to_numpy(dtype=float)
    return float(w @ cov.to_numpy() @ w)


def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Portfolio volatility √(wᵀΣw)."""
    return float(np.sqrt(portfolio_variance(weights, cov)))


# ─── Building block ───────────────────────────────────────────────────────────

def _sigma_inv(cov: pd.DataFrame, b: np.ndarray) -> np.ndarray:
    """Return Σ⁻¹b via a linear solve (stable) rather than forming Σ⁻¹."""
    return np.linalg.solve(cov.to_numpy(dtype=float), b)


# ─── Global minimum variance ──────────────────────────────────────────────────

def global_minimum_variance(cov: pd.DataFrame) -> pd.Series:
    r"""GMV weights w = Σ⁻¹𝟙 / (𝟙ᵀΣ⁻¹𝟙). Uses only Σ, no return forecast."""
    ones = np.ones(cov.shape[0])
    z = _sigma_inv(cov, ones)          # Σ⁻¹𝟙
    w = z / (ones @ z)
    return pd.Series(w, index=cov.columns)


# ─── Tangency (max Sharpe) ────────────────────────────────────────────────────

def tangency_portfolio(mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float) -> pd.Series:
    r"""
    Tangency (maximum-Sharpe) weights w ∝ Σ⁻¹(μ − r_f𝟙), renormalised to sum
    to 1. μ and r_f must share a basis with Σ.
    """
    mu_v = mu.reindex(cov.columns).to_numpy(dtype=float)
    excess = mu_v - risk_free_rate * np.ones(len(mu_v))
    z = _sigma_inv(cov, excess)        # Σ⁻¹(μ − r_f𝟙)
    denom = np.ones(len(mu_v)) @ z
    if abs(denom) < 1e-300:
        raise ValueError("tangency undefined: 𝟙ᵀΣ⁻¹(μ − r_f𝟙) ≈ 0")
    return pd.Series(z / denom, index=cov.columns)


# ─── Efficient frontier (two-fund, closed form) ───────────────────────────────

@dataclass
class FrontierConstants:
    """The scalars A, B, C, D and the affine coefficients g, h of w(m)=g+h·m."""
    A: float
    B: float
    C: float
    D: float
    g: np.ndarray
    h: np.ndarray


def frontier_constants(mu: pd.Series, cov: pd.DataFrame) -> FrontierConstants:
    r"""
    The efficient-set constants from the two-constraint Lagrangian:

        A = 𝟙ᵀΣ⁻¹𝟙,  B = 𝟙ᵀΣ⁻¹μ,  C = μᵀΣ⁻¹μ,  D = AC − B²
        g = (C·Σ⁻¹𝟙 − B·Σ⁻¹μ)/D,   h = (A·Σ⁻¹μ − B·Σ⁻¹𝟙)/D

    so the minimum-variance portfolio achieving return m is w(m) = g + h·m.
    (D > 0 whenever Σ ≻ 0 and μ is not a multiple of 𝟙.)
    """
    mu_v = mu.reindex(cov.columns).to_numpy(dtype=float)
    ones = np.ones(len(mu_v))

    sig_inv_ones = _sigma_inv(cov, ones)     # Σ⁻¹𝟙
    sig_inv_mu = _sigma_inv(cov, mu_v)       # Σ⁻¹μ

    A = float(ones @ sig_inv_ones)
    B = float(ones @ sig_inv_mu)
    C = float(mu_v @ sig_inv_mu)
    D = A * C - B * B
    if D <= 0:
        raise ValueError("frontier undefined: AC − B² ≤ 0 (μ collinear with 𝟙?)")

    g = (C * sig_inv_ones - B * sig_inv_mu) / D
    h = (A * sig_inv_mu - B * sig_inv_ones) / D
    return FrontierConstants(A=A, B=B, C=C, D=D, g=g, h=h)


def frontier_weights(mu: pd.Series, cov: pd.DataFrame, target_return: float) -> pd.Series:
    """Minimum-variance weights achieving exactly target_return: w = g + h·m."""
    k = frontier_constants(mu, cov)
    w = k.g + k.h * target_return
    return pd.Series(w, index=cov.columns)


@dataclass
class FrontierResult:
    """
    A traced efficient frontier plus the two reference portfolios.

        returns      target returns swept (ascending)
        volatilities matching σ at each target
        weights      DataFrame rows=frontier points, cols=tickers
        sharpe       (return − r_f)/σ per point (NaN if no r_f)
        gmv_weights  global-minimum-variance portfolio
        tangency_weights  max-Sharpe portfolio (None if no r_f)
        gmv_return, gmv_vol   coordinates of the frontier's leftmost tip
    """
    returns: np.ndarray
    volatilities: np.ndarray
    weights: pd.DataFrame
    sharpe: np.ndarray
    gmv_weights: pd.Series
    tangency_weights: pd.Series | None
    gmv_return: float
    gmv_vol: float


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 50,
    risk_free_rate: float | None = None,
) -> FrontierResult:
    r"""
    Trace the efficient frontier from the GMV return up to the highest
    single-asset expected return, using the closed-form w(m) = g + h·m.

    σ²(m) = (A m² − 2B m + C)/D is evaluated directly rather than by
    recomputing wᵀΣw at each point — same answer, but it makes the parabola
    explicit.
    """
    k = frontier_constants(mu, cov)
    gmv_return = k.B / k.A                       # return of the GMV portfolio
    gmv_vol = float(np.sqrt(1.0 / k.A))          # its volatility

    top = float(mu.max())
    if top <= gmv_return:                        # degenerate; give a small span
        top = gmv_return + abs(gmv_return) + 1e-6
    targets = np.linspace(gmv_return, top, n_points)

    weights = np.vstack([k.g + k.h * m for m in targets])
    variances = (k.A * targets**2 - 2 * k.B * targets + k.C) / k.D
    vols = np.sqrt(variances)

    if risk_free_rate is not None:
        sharpe = (targets - risk_free_rate) / vols
        tangency = tangency_portfolio(mu, cov, risk_free_rate)
    else:
        sharpe = np.full_like(targets, np.nan)
        tangency = None

    return FrontierResult(
        returns=targets,
        volatilities=vols,
        weights=pd.DataFrame(weights, columns=cov.columns),
        sharpe=sharpe,
        gmv_weights=global_minimum_variance(cov),
        tangency_weights=tangency,
        gmv_return=gmv_return,
        gmv_vol=gmv_vol,
    )
