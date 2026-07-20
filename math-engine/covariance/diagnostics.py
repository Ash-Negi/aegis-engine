"""
Aegis Engine — Covariance Diagnostics
=====================================
Tools to interrogate a covariance matrix Σ *before* it is handed to the
optimizer. Two questions matter:

    1. How safe is Σ to invert?   → condition number
    2. How many independent bets does Σ actually contain?  → eigenstructure

Both are answered from the eigenvalues of Σ. Because Σ is symmetric PSD we
use np.linalg.eigh (the symmetric solver): it is faster and, crucially,
returns real, ordered eigenvalues with orthonormal eigenvectors — exactly
the guarantees the general eig() does not give.

Why this is a first-class module and not a few inline print statements:
the whole thesis of Week 2 is that HOW you estimate Σ changes how safely
the optimizer can use it. You cannot make that argument without measuring
conditioning, so these diagnostics are the evidence, and they need to be
as tested as the estimators themselves.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CovarianceConfig


# ─── Input handling ───────────────────────────────────────────────────────────

def _cov_matrix(cov: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Validate a covariance frame (square, symmetric) and extract (M, labels)."""
    if not isinstance(cov, pd.DataFrame):
        raise TypeError(f"cov must be a pandas DataFrame, got {type(cov).__name__}")

    M = cov.to_numpy(dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"cov must be square, got shape {M.shape}")
    if not np.allclose(M, M.T, atol=1e-12):
        raise ValueError("cov is not symmetric — is this really a covariance matrix?")

    return M, list(cov.columns)


# ─── Condition number ─────────────────────────────────────────────────────────

def condition_number(cov: pd.DataFrame) -> float:
    r"""
    Spectral condition number κ(Σ) = λ_max / λ_min.

    κ measures how much Σ⁻¹ amplifies error. A mean-variance optimizer
    computes weights ∝ Σ⁻¹μ, so a relative error ε in the inputs can be
    magnified up to κ·ε in the weights. κ = 1 is a perfectly conditioned
    (spherical) matrix; κ in the thousands means the optimizer is balancing
    on a knife-edge and tiny data changes flip the allocation.

    For a symmetric PSD matrix the 2-norm condition number is exactly the
    ratio of largest to smallest eigenvalue, which is what we return. If
    the smallest eigenvalue is ≤ 0 (numerically singular / rank-deficient)
    the matrix is not safely invertible and we return +inf.
    """
    M, _ = _cov_matrix(cov)
    eigenvalues = np.linalg.eigvalsh(M)
    lam_min, lam_max = eigenvalues[0], eigenvalues[-1]  # eigvalsh returns ascending
    if lam_min <= 0:
        return np.inf
    return float(lam_max / lam_min)


def log_condition_number(cov: pd.DataFrame) -> float:
    """
    log₁₀ of the condition number — the natural scale to compare estimators.

    Conditioning spans orders of magnitude, so log₁₀ is how it is read in
    practice: ~0 is excellent, 1–2 is workable, 3+ starts to hurt, and each
    unit is a 10× jump in error amplification. This is the "before vs after
    shrinkage" number the Week 2 brief asks for.
    """
    return float(np.log10(condition_number(cov)))


# ─── Eigenstructure ───────────────────────────────────────────────────────────

@dataclass
class EigenAnalysis:
    """
    The eigen-decomposition of Σ with the derived risk-factor read-outs.

        eigenvalues         descending; each is the variance of one
                            principal portfolio (uncorrelated risk factor)
        eigenvectors        columns aligned to eigenvalues; column i is the
                            asset loadings of factor i
        variance_explained  eigenvalue / Σ eigenvalues, descending
        cumulative_variance running sum of variance_explained
        n_factors           # factors each explaining ≥ threshold of variance
        participation_ratio (Σλ)²/Σλ² — a continuous "effective number of
                            factors": N if risk is spread evenly, →1 if one
                            factor dominates
        condition_number    λ_max/λ_min
        log_condition_number
        tickers             label order the eigenvectors are expressed in
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    variance_explained: np.ndarray
    cumulative_variance: np.ndarray
    n_factors: int
    participation_ratio: float
    condition_number: float
    log_condition_number: float
    tickers: list[str]

    def summary(self) -> pd.DataFrame:
        """Per-factor table: eigenvalue, variance explained, cumulative."""
        return pd.DataFrame(
            {
                "eigenvalue": self.eigenvalues,
                "variance_explained": self.variance_explained,
                "cumulative_variance": self.cumulative_variance,
            },
            index=[f"factor_{i + 1}" for i in range(len(self.eigenvalues))],
        )


def eigen_analysis(cov: pd.DataFrame, config: CovarianceConfig | None = None) -> EigenAnalysis:
    r"""
    Decompose Σ into its principal risk factors.

    Every symmetric PSD Σ can be written Σ = V Λ Vᵀ, where the columns of V
    are orthonormal eigenvectors and Λ = diag(λ₁…λ_N) holds the eigenvalues.
    Read financially: each eigenvector is a portfolio of the assets whose
    returns are UNCORRELATED with every other eigen-portfolio, and its
    eigenvalue is that portfolio's variance. So the eigenvalues answer "how
    many genuinely independent sources of risk are in this universe?"

    With four assets there are at most four factors, but they are rarely of
    equal size. If one eigenvalue holds most of the variance, the universe
    is effectively a one-bet universe wearing a four-asset costume — the
    optimizer has far less to diversify across than the asset count implies.
    We quantify this two ways: a hard count of factors above a variance
    threshold, and the participation ratio (a soft, threshold-free "effective
    number of factors").

    Args:
        cov:    a labelled covariance frame from any estimator.
        config: CovarianceConfig; reads factor_variance_threshold. Default.

    Returns:
        EigenAnalysis (see the dataclass for field meanings).
    """
    config = config or CovarianceConfig()
    M, tickers = _cov_matrix(cov)

    # eigh returns ascending eigenvalues + orthonormal eigenvectors; reverse
    # to descending so factor_1 is the dominant source of variance.
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    total = eigenvalues.sum()
    variance_explained = eigenvalues / total
    cumulative_variance = np.cumsum(variance_explained)

    n_factors = int((variance_explained >= config.factor_variance_threshold).sum())

    # Participation ratio: (Σλ)² / Σλ². Equals N when all λ are equal
    # (variance spread perfectly evenly) and tends to 1 when a single λ
    # dominates. A threshold-free companion to n_factors.
    participation_ratio = float(total**2 / np.sum(eigenvalues**2))

    lam_min = eigenvalues[-1]
    cond = float(eigenvalues[0] / lam_min) if lam_min > 0 else np.inf

    return EigenAnalysis(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        variance_explained=variance_explained,
        cumulative_variance=cumulative_variance,
        n_factors=n_factors,
        participation_ratio=participation_ratio,
        condition_number=cond,
        log_condition_number=float(np.log10(cond)) if np.isfinite(cond) else np.inf,
        tickers=tickers,
    )
