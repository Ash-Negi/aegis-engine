"""
Aegis Engine — Adaptive Weight Engine (Phase 2)
===============================================
Tilt the Phase 1 optimal portfolio according to the detected regime.

This is the bridge between "markets are non-stationary" (regime detection)
and "so the portfolio should change" (this module). The optimizer gives one
risk-optimal set of weights; the adaptive engine nudges them risk-on in calm
trending markets and risk-off in a crisis — leaning into equity/crypto when
the regime rewards it, into gold when it does not.

Mechanism: multiplicative tilts by asset class (from SignalsConfig), applied
to the base weights and then RENORMALISED to sum to 1. Multiplicative +
renormalise is deliberately chosen over additive tilts because it
automatically preserves the long-only, fully-invested constraints — a
positive base weight scaled by a positive multiplier stays positive, and the
renormalisation restores the budget. A zero base weight stays zero (the
engine tilts what you hold; it does not open new positions).
"""

from dataclasses import dataclass

import pandas as pd

from config import SignalsConfig, UNIVERSE

# ticker → asset class, so a class-level tilt maps to individual assets.
_ASSET_CLASS = {a.ticker: a.asset_class for a in UNIVERSE}


@dataclass
class AdaptiveWeights:
    """A tilt result, keeping the before/after and the regime for auditing."""
    weights: pd.Series      # tilted, renormalised weights
    base_weights: pd.Series # the pre-tilt optimal weights
    regime: str             # regime that produced the tilt
    turnover: float         # ½Σ|tilted − base|, the trade the tilt implies


class AdaptiveWeightEngine:
    """Apply regime-conditioned multiplicative tilts to a base portfolio."""

    def __init__(self, config: SignalsConfig | None = None):
        self.config = config or SignalsConfig()

    def tilt(self, base_weights: pd.Series, regime: str) -> AdaptiveWeights:
        """
        Tilt `base_weights` for `regime` and renormalise to sum to 1.

        Unknown regimes (or the neutral one) leave the weights unchanged.
        """
        if regime not in self.config.regime_tilts:
            raise ValueError(f"unknown regime {regime!r}; "
                             f"known: {list(self.config.regime_tilts)}")

        multipliers = self.config.regime_tilts[regime]
        tilted = base_weights.copy().astype(float)
        for ticker in tilted.index:
            asset_class = _ASSET_CLASS.get(ticker)
            tilted[ticker] *= multipliers.get(asset_class, 1.0)

        total = tilted.sum()
        if total <= 0:
            raise ValueError("tilt collapsed all weight to ≤ 0; check tilt config")
        tilted = tilted / total

        turnover = 0.5 * (tilted - base_weights).abs().sum()
        return AdaptiveWeights(
            weights=tilted,
            base_weights=base_weights.copy(),
            regime=regime,
            turnover=float(turnover),
        )
