"""
Aegis Engine — Covariance Estimation (Phase 1, Week 2)
======================================================
Estimators for the asset-return covariance matrix Σ, plus conditioning
diagnostics. Σ is the central input to mean-variance optimization: the
optimizer inverts it, so how well Σ is estimated determines how stable
the resulting weights are.

Estimators (returns → Σ):
    sample_covariance(returns)     — textbook baseline, (T-1) normalization
    ewma_covariance(returns)       — RiskMetrics exponential weighting, λ=0.94
    ledoit_wolf_covariance(returns)— shrinkage estimator (primary)
    ledoit_wolf_shrinkage(returns) — as above, plus δ and the decomposition

Diagnostics (Σ → conditioning / risk-factor read-outs):
    condition_number(cov) / log_condition_number(cov)
    eigen_analysis(cov)            — factors, variance explained, participation

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
from covariance.diagnostics import (
    condition_number,
    log_condition_number,
    eigen_analysis,
    EigenAnalysis,
)

__all__ = [
    "sample_covariance",
    "ewma_covariance",
    "ledoit_wolf_covariance",
    "ledoit_wolf_shrinkage",
    "LedoitWolfResult",
    "condition_number",
    "log_condition_number",
    "eigen_analysis",
    "EigenAnalysis",
]
