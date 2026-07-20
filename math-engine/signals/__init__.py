"""
Aegis Engine — Signals (Phase 2)
================================
The adaptive layer: detect market regimes and respond to them, plus
statistical-arbitrage (cointegration) signals. Phase 1 assumed one Σ and one
optimal portfolio for all time; Phase 2 makes the engine react to a changing
market.

    hmm            GaussianHMM — from-scratch Baum-Welch + Viterbi
    regimes        RegimeDetector — HMM states → named regimes
    adaptive       AdaptiveWeightEngine — regime → weight tilts
    cointegration  Engle-Granger, Johansen, spread z-signals
"""

from signals.hmm import GaussianHMM, HMMParams
from signals.regimes import (
    RegimeDetector,
    RegimeResult,
    LOW_VOL_TRENDING,
    HIGH_VOL_MEANREV,
    CRISIS,
    REGIME_ORDER,
)
from signals.adaptive import AdaptiveWeightEngine, AdaptiveWeights
from signals.cointegration import (
    engle_granger,
    test_all_pairs,
    johansen_rank,
    spread_signal,
    PairTest,
    SpreadSignal,
)

__all__ = [
    "GaussianHMM",
    "HMMParams",
    "RegimeDetector",
    "RegimeResult",
    "LOW_VOL_TRENDING",
    "HIGH_VOL_MEANREV",
    "CRISIS",
    "REGIME_ORDER",
    "AdaptiveWeightEngine",
    "AdaptiveWeights",
    "engle_granger",
    "test_all_pairs",
    "johansen_rank",
    "spread_signal",
    "PairTest",
    "SpreadSignal",
]
