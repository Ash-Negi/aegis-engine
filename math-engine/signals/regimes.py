"""
Aegis Engine — Regime Detection (Phase 2)
=========================================
Fit a Gaussian HMM to market returns and translate its hidden states into
named market regimes the rest of the engine can act on:

    low_vol_trending   calm, positive drift — the risk-on regime
    high_vol_meanrev   choppy, elevated vol, little drift
    crisis             high vol, negative drift — the risk-off regime

The HMM has no idea what a "regime" is — it finds K Gaussian states that best
explain the return series. The mapping from anonymous states to these names
is done here, AFTER fitting, by ranking the states on their fitted
volatility: the calmest state is trending, the wildest is crisis, the middle
one is mean-reverting. This keeps the statistical model honest (it fits the
data) and the labels interpretable (they mean something to a portfolio).

Phase 2 is where the engine stops assuming markets are stationary. Phase 1's
optimizer produces ONE covariance and ONE optimal portfolio for all time;
regime detection is the admission that the right portfolio in a calm bull
market is not the right portfolio in a crisis.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.hmm import GaussianHMM

LOW_VOL_TRENDING = "low_vol_trending"
HIGH_VOL_MEANREV = "high_vol_meanrev"
CRISIS = "crisis"
# Ordered by ascending volatility — the ranking used to assign the labels.
REGIME_ORDER = [LOW_VOL_TRENDING, HIGH_VOL_MEANREV, CRISIS]


@dataclass
class RegimeResult:
    """Fitted regime assignment plus everything needed to audit it."""
    labels: pd.Series          # regime name per date (Viterbi decoding)
    states: pd.Series          # raw HMM state index per date
    proba: pd.DataFrame        # smoothed P(regime | data), cols = regime names
    state_to_regime: dict      # HMM state index → regime name
    regime_stats: pd.DataFrame # per regime: ann_return, ann_vol, days, frequency
    model: GaussianHMM

    @property
    def current(self) -> str:
        """The most recent day's regime label."""
        return self.labels.iloc[-1]


class RegimeDetector:
    """
    Detect market regimes from a daily return series with a 3-state Gaussian
    HMM. Fit on a market proxy (e.g. an equal-weight portfolio's returns).
    """

    def __init__(self, n_regimes: int = 3, trading_days: int = 252,
                 random_state: int | None = 42):
        if n_regimes != 3:
            raise ValueError("RegimeDetector labels exactly 3 regimes; "
                             f"got n_regimes={n_regimes}")
        self.n_regimes = n_regimes
        self.trading_days = trading_days
        self.random_state = random_state

    def fit(self, market_returns: pd.Series) -> RegimeResult:
        r = market_returns.dropna()
        x = r.to_numpy()

        model = GaussianHMM(self.n_regimes, random_state=self.random_state).fit(x)
        states = model.predict(x)
        proba = model.predict_proba(x)

        # Rank states by fitted volatility → assign the ordered regime names.
        vols = np.sqrt(model.params_.covars.reshape(self.n_regimes, -1)[:, 0])
        order = np.argsort(vols)                       # calmest → wildest
        state_to_regime = {int(s): REGIME_ORDER[rank] for rank, s in enumerate(order)}

        labels = pd.Series([state_to_regime[s] for s in states], index=r.index, name="regime")
        proba_named = pd.DataFrame(
            {REGIME_ORDER[rank]: proba[:, s] for rank, s in enumerate(order)},
            index=r.index,
        )[REGIME_ORDER]

        # Per-regime descriptive stats, from the actual (not model) returns.
        rows = {}
        for regime in REGIME_ORDER:
            mask = labels == regime
            seg = r[mask]
            rows[regime] = {
                "ann_return_pct": round(seg.mean() * self.trading_days * 100, 2) if len(seg) else np.nan,
                "ann_vol_pct": round(seg.std() * np.sqrt(self.trading_days) * 100, 2) if len(seg) else np.nan,
                "days": int(mask.sum()),
                "frequency_pct": round(mask.mean() * 100, 1),
            }
        regime_stats = pd.DataFrame(rows).T.loc[REGIME_ORDER]

        return RegimeResult(
            labels=labels,
            states=pd.Series(states, index=r.index, name="state"),
            proba=proba_named,
            state_to_regime=state_to_regime,
            regime_stats=regime_stats,
            model=model,
        )
