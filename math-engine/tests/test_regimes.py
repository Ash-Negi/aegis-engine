"""
Aegis Engine — Regime Detection & Adaptive Weights Tests
========================================================
Run with: pytest tests/test_regimes.py -v

Two things must hold for the regime layer to be trustworthy: the labels must
match their names (the "crisis" regime really is the high-vol one), and the
adaptive tilts must move risk in the intended direction while staying
long-only and fully invested.

──────────────────────────────────────────────────────────────────────────────
Review Log
──────────────────────────────────────────────────────────────────────────────
2026-07-20 — Phase 2, regimes + adaptive weights

  test_labels_match_volatility_order
      The label assignment is by fitted volatility, so crisis vol ≥
      mean-rev vol ≥ trending vol must hold in the output stats. A mislabel
      would send the adaptive engine the wrong way in a crisis.
  test_risk_on_adds_equity / test_crisis_adds_gold
      The economic contract of the tilts: risk-on lifts equity, crisis lifts
      gold. This is the whole point of the adaptive layer.
  test_tilt_stays_long_only_and_invested / test_zero_weight_stays_zero
      Tilts preserve the portfolio constraints and never open a new position.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import pytest

from config import DataConfig, SignalsConfig, TICKERS
from data.pipeline import DataPipeline
from signals import (
    RegimeDetector,
    AdaptiveWeightEngine,
    LOW_VOL_TRENDING,
    HIGH_VOL_MEANREV,
    CRISIS,
    REGIME_ORDER,
)


@pytest.fixture(scope="module")
def market_returns():
    """Equal-weight portfolio daily returns as the market proxy."""
    pipeline = DataPipeline(DataConfig())
    dataset = pipeline.run(TICKERS, use_cache=True)
    r = pipeline.get_common_period(dataset)["returns"]
    return r.mean(axis=1)


class TestRegimeDetector:
    def test_labels_are_named_regimes(self, market_returns):
        res = RegimeDetector(random_state=42).fit(market_returns)
        assert set(res.labels.unique()).issubset(set(REGIME_ORDER))

    def test_labels_match_volatility_order(self, market_returns):
        res = RegimeDetector(random_state=42).fit(market_returns)
        vols = res.regime_stats["ann_vol_pct"]
        # Present regimes must be ordered trending ≤ mean-rev ≤ crisis in vol.
        present = [r for r in REGIME_ORDER if res.regime_stats.loc[r, "days"] > 0]
        ordered_vols = [vols[r] for r in present]
        assert ordered_vols == sorted(ordered_vols), f"vol order violated: {ordered_vols}"

    def test_proba_normalised(self, market_returns):
        res = RegimeDetector(random_state=42).fit(market_returns)
        np.testing.assert_allclose(res.proba.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
        assert list(res.proba.columns) == REGIME_ORDER

    def test_current_is_last_label(self, market_returns):
        res = RegimeDetector(random_state=42).fit(market_returns)
        assert res.current == res.labels.iloc[-1]

    def test_rejects_non_three(self):
        with pytest.raises(ValueError):
            RegimeDetector(n_regimes=2)


class TestAdaptiveWeights:
    @pytest.fixture
    def base(self):
        # A plausible optimal portfolio: equity (QQQ+VXUS), gold, no crypto.
        return pd.Series({"QQQ": 0.25, "GLDM": 0.30, "FBTC": 0.05, "VXUS": 0.40})

    def test_tilt_stays_long_only_and_invested(self, base):
        eng = AdaptiveWeightEngine(SignalsConfig())
        for regime in REGIME_ORDER:
            out = eng.tilt(base, regime)
            assert out.weights.sum() == pytest.approx(1.0)
            assert (out.weights >= 0).all()

    def test_risk_on_adds_equity(self, base):
        eng = AdaptiveWeightEngine(SignalsConfig())
        out = eng.tilt(base, LOW_VOL_TRENDING)
        base_equity = base["QQQ"] + base["VXUS"]
        tilt_equity = out.weights["QQQ"] + out.weights["VXUS"]
        assert tilt_equity > base_equity

    def test_crisis_adds_gold(self, base):
        eng = AdaptiveWeightEngine(SignalsConfig())
        out = eng.tilt(base, CRISIS)
        assert out.weights["GLDM"] > base["GLDM"]

    def test_neutral_regime_unchanged(self, base):
        eng = AdaptiveWeightEngine(SignalsConfig())
        out = eng.tilt(base, HIGH_VOL_MEANREV)   # all multipliers 1.0
        pd.testing.assert_series_equal(out.weights, base, check_names=False)

    def test_zero_weight_stays_zero(self, base):
        base = base.copy()
        base["FBTC"] = 0.0
        base = base / base.sum()
        eng = AdaptiveWeightEngine(SignalsConfig())
        out = eng.tilt(base, LOW_VOL_TRENDING)
        assert out.weights["FBTC"] == 0.0

    def test_unknown_regime_raises(self, base):
        with pytest.raises(ValueError):
            AdaptiveWeightEngine().tilt(base, "euphoria")
