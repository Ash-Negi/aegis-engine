"""
Aegis Engine — Covariance Estimation (Phase 1, Week 2)
======================================================
Estimators for the asset-return covariance matrix Σ, plus conditioning
diagnostics. Σ is the central input to mean-variance optimization: the
optimizer inverts it, so how well Σ is estimated determines how stable
the resulting weights are.

Public API:
    sample_covariance(returns)     — textbook baseline, (T-1) normalization
    ewma_covariance(returns)       — RiskMetrics exponential weighting, λ=0.94
    ledoit_wolf_covariance(returns)— shrinkage estimator (primary)
    ledoit_wolf_shrinkage(returns) — as above, plus δ and the decomposition

Each estimator takes a DataFrame of returns (rows = dates, cols = tickers)
and returns a labelled DataFrame covariance matrix in the SAME column
order — that ordering contract is what lets the optimizer trust that
row/col i of Σ corresponds to ticker i everywhere downstream.
"""

from covariance.estimators import (
    sample_covariance,
    ewma_covariance,
    ledoit_wolf_covariance,
    ledoit_wolf_shrinkage,
    LedoitWolfResult,
)

__all__ = [
    "sample_covariance",
    "ewma_covariance",
    "ledoit_wolf_covariance",
    "ledoit_wolf_shrinkage",
    "LedoitWolfResult",
]
