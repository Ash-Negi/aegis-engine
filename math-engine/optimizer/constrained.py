"""
Aegis Engine — Constrained Optimization
=======================================
Long-only, sector-capped portfolios — what an actual desk trades, and what
the closed-form module cannot give you.

Why a numerical solver here (and closed form next door):
    Adding INEQUALITY constraints (wᵢ ≥ 0, sector sums ≤ cap) destroys the
    clean Lagrange solution. Equality constraints keep the problem linear in
    the multipliers; inequalities make it a quadratic program whose active
    set is not known in advance. There is no formula — you search. We use
    scipy's SLSQP (sequential least-squares QP), the standard tool for a
    small smooth QP like this. This is a deliberate split from the Ledoit-
    Wolf decision (ADR-004) to build from scratch: a 30-line shrinkage
    estimator is worth hand-coding for insight; a robust active-set QP is
    not the value-add, and SLSQP is auditable in what it optimises even if
    not in how. See ADR-005.

Sectors are the asset CLASSES from the universe (equity / commodity /
crypto). The default equity cap stops the optimiser from doubling up on the
0.72-correlated QQQ + VXUS pair.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import UNIVERSE

# ticker → asset class, the "sector" a cap applies to.
SECTOR_MAP = {a.ticker: a.asset_class for a in UNIVERSE}

_SLSQP_OPTS = {"ftol": 1e-12, "maxiter": 1000}


# ─── Constraint construction ──────────────────────────────────────────────────

def _bounds(n: int, config) -> list[tuple[float, float]]:
    """Per-asset box constraints: [0, max_weight] long-only, else [-max, max]."""
    lo = 0.0 if config.long_only else -config.max_weight
    return [(lo, config.max_weight)] * n


def _base_constraints(tickers: list[str], config) -> list[dict]:
    """Full-investment equality plus one ≤-cap inequality per capped sector."""
    cons = [{
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
        "jac": lambda w: np.ones_like(w),
    }]
    for sector, cap in config.sector_caps.items():
        idx = [i for i, t in enumerate(tickers) if SECTOR_MAP.get(t) == sector]
        if idx:
            # cap − Σ_{i∈sector} wᵢ ≥ 0
            cons.append({
                "type": "ineq",
                "fun": (lambda w, idx=idx, cap=cap: cap - np.sum(w[idx])),
                "jac": (lambda w, idx=idx: -np.array([1.0 if i in idx else 0.0
                                                      for i in range(len(w))])),
            })
    return cons


def _clean(weights: np.ndarray, columns) -> pd.Series:
    """Zero numerical dust, renormalise to sum 1, wrap in a labelled Series."""
    w = np.where(np.abs(weights) < 1e-9, 0.0, weights)
    total = w.sum()
    if total != 0:
        w = w / total
    return pd.Series(w, index=columns)


def _check(res, what: str) -> None:
    if not res.success:
        raise RuntimeError(f"{what} failed to converge: {res.message}")


# ─── Constrained portfolios ───────────────────────────────────────────────────

def min_variance_portfolio(cov: pd.DataFrame, config) -> pd.Series:
    """Long-only, sector-capped minimum-variance portfolio (needs only Σ)."""
    n = cov.shape[0]
    Sigma = cov.to_numpy(dtype=float)
    res = minimize(
        fun=lambda w: w @ Sigma @ w,
        x0=np.full(n, 1.0 / n),
        jac=lambda w: 2.0 * Sigma @ w,
        method="SLSQP",
        bounds=_bounds(n, config),
        constraints=_base_constraints(list(cov.columns), config),
        options=_SLSQP_OPTS,
    )
    _check(res, "min_variance_portfolio")
    return _clean(res.x, cov.columns)


def max_sharpe_portfolio(mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float,
                         config) -> pd.Series:
    """Long-only, sector-capped maximum-Sharpe portfolio."""
    n = cov.shape[0]
    Sigma = cov.to_numpy(dtype=float)
    mu_v = mu.reindex(cov.columns).to_numpy(dtype=float)

    def neg_sharpe(w):
        excess = w @ mu_v - risk_free_rate
        vol = np.sqrt(w @ Sigma @ w)
        return -excess / vol if vol > 0 else 0.0

    res = minimize(
        fun=neg_sharpe,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=_bounds(n, config),
        constraints=_base_constraints(list(cov.columns), config),
        options=_SLSQP_OPTS,
    )
    _check(res, "max_sharpe_portfolio")
    return _clean(res.x, cov.columns)


def target_return_portfolio(mu: pd.Series, cov: pd.DataFrame, target_return: float,
                            config) -> pd.Series:
    """Long-only, sector-capped minimum-variance portfolio hitting a target return."""
    n = cov.shape[0]
    Sigma = cov.to_numpy(dtype=float)
    mu_v = mu.reindex(cov.columns).to_numpy(dtype=float)

    cons = _base_constraints(list(cov.columns), config) + [{
        "type": "eq",
        "fun": (lambda w, m=target_return: w @ mu_v - m),
        "jac": lambda w: mu_v,
    }]
    res = minimize(
        fun=lambda w: w @ Sigma @ w,
        x0=np.full(n, 1.0 / n),
        jac=lambda w: 2.0 * Sigma @ w,
        method="SLSQP",
        bounds=_bounds(n, config),
        constraints=cons,
        options=_SLSQP_OPTS,
    )
    _check(res, f"target_return_portfolio({target_return:.4f})")
    return _clean(res.x, cov.columns)


def _max_feasible_return(mu: pd.Series, cov: pd.DataFrame, config) -> float:
    """Largest expected return reachable under the constraints (an LP via SLSQP)."""
    n = cov.shape[0]
    mu_v = mu.reindex(cov.columns).to_numpy(dtype=float)
    res = minimize(
        fun=lambda w: -(w @ mu_v),
        x0=np.full(n, 1.0 / n),
        jac=lambda w: -mu_v,
        method="SLSQP",
        bounds=_bounds(n, config),
        constraints=_base_constraints(list(cov.columns), config),
        options=_SLSQP_OPTS,
    )
    _check(res, "max_feasible_return")
    return float(res.x @ mu_v)


@dataclass
class ConstrainedFrontier:
    """The long-only, sector-capped efficient frontier plus reference portfolios."""
    returns: np.ndarray
    volatilities: np.ndarray
    weights: pd.DataFrame
    sharpe: np.ndarray
    min_var_weights: pd.Series
    max_sharpe_weights: pd.Series


def efficient_frontier_constrained(mu: pd.Series, cov: pd.DataFrame, config,
                                   risk_free_rate: float | None = None) -> ConstrainedFrontier:
    """
    Trace the constrained frontier from the min-variance portfolio's return
    up to the maximum feasible return, solving a QP at each target. Targets
    that fail to converge (usually at the very top edge) are skipped rather
    than aborting the whole sweep.
    """
    mvp = min_variance_portfolio(cov, config)
    r_lo = float(mvp.reindex(cov.columns) @ mu.reindex(cov.columns))
    r_hi = _max_feasible_return(mu, cov, config)

    targets = np.linspace(r_lo, r_hi, config.frontier_points)
    rows, rets, vols = [], [], []
    for m in targets:
        try:
            w = target_return_portfolio(mu, cov, m, config)
        except RuntimeError:
            continue
        rows.append(w.to_numpy())
        rets.append(float(w.reindex(cov.columns) @ mu.reindex(cov.columns)))
        vols.append(float(np.sqrt(w.reindex(cov.columns) @ cov.to_numpy() @ w.reindex(cov.columns))))

    rets = np.array(rets)
    vols = np.array(vols)
    sharpe = ((rets - risk_free_rate) / vols) if risk_free_rate is not None \
        else np.full_like(rets, np.nan)

    max_sharpe = (max_sharpe_portfolio(mu, cov, risk_free_rate, config)
                  if risk_free_rate is not None else None)

    return ConstrainedFrontier(
        returns=rets,
        volatilities=vols,
        weights=pd.DataFrame(rows, columns=cov.columns),
        sharpe=sharpe,
        min_var_weights=mvp,
        max_sharpe_weights=max_sharpe,
    )
