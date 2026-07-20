"""
Aegis Engine — Mean-Variance Optimization (Phase 1, Week 3)
===========================================================
Turns the return/risk inputs (μ, Σ) into portfolio weights.

The layer is split by how much math is available in closed form:

    expected_returns  — estimate μ with shrinkage (μ dominates MVO error)
    mean_variance     — CLOSED-FORM analytics via Lagrange multipliers:
                        global-minimum-variance, tangency, efficient frontier
    constrained       — long-only + sector caps, where no closed form exists,
                        solved numerically (scipy SLSQP)

The closed-form module is the textbook showcase (Σ⁻¹ and Lagrange); the
constrained module is what a real desk actually trades (no shorting, caps).
Both consume the same μ and Σ and return weights that sum to 1, indexed by
ticker in the canonical universe order.

Public API grows as Week 3 lands, piece by piece.
"""

from optimizer.expected_returns import (
    sample_mean_returns,
    bayes_stein_shrinkage,
    grand_mean_shrinkage,
    estimate_expected_returns,
    ExpectedReturns,
)

__all__ = [
    "sample_mean_returns",
    "bayes_stein_shrinkage",
    "grand_mean_shrinkage",
    "estimate_expected_returns",
    "ExpectedReturns",
]
