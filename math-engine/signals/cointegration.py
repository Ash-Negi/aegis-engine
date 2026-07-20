"""
Aegis Engine — Cointegration & Spread Signals (Phase 2)
=======================================================
Statistical-arbitrage plumbing: find pairs of assets whose prices move
together in the long run (are *cointegrated*), and trade the spread when it
strays too far from its mean.

Correlation vs cointegration — the distinction this module is built on:
    Correlation is about co-movement of RETURNS day to day. Cointegration is
    stronger: two non-stationary price series are cointegrated if some linear
    combination of them IS stationary — i.e. they are tied together by an
    equilibrium the spread reverts to. Correlated assets can still drift
    apart forever; cointegrated ones cannot. Only cointegration justifies a
    mean-reversion trade on the spread, because only then is the spread
    guaranteed (statistically) to come back.

Division of labour (mirrors ADR-005): the hard statistical machinery — unit-
root and cointegration tests with their MacKinnon/Osterwald-Lenum critical
values — comes from statsmodels, where the critical-value tables are correct
and battle-tested. The trading logic on top (hedge ratio, spread z-score,
±entry/exit signals) is built here, because that is the part worth owning.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from config import SignalsConfig


@dataclass
class PairTest:
    """Engle-Granger result for one ordered pair (regress y on x)."""
    y: str
    x: str
    hedge_ratio: float     # β in  y = α + β·x  (OLS on log prices)
    eg_pvalue: float       # Engle-Granger cointegration p-value
    adf_pvalue: float      # ADF p-value on the spread residual
    cointegrated: bool     # eg_pvalue < significance


@dataclass
class SpreadSignal:
    """A spread and the mean-reversion signal derived from its z-score."""
    spread: pd.Series      # y − β·x on log prices
    zscore: pd.Series      # standardised spread
    position: pd.Series    # +1 long spread, −1 short spread, 0 flat
    hedge_ratio: float
    entry_z: float
    exit_z: float


def engle_granger(log_prices: pd.DataFrame, y: str, x: str,
                  config: SignalsConfig | None = None) -> PairTest:
    r"""
    Engle-Granger two-step test that `y` and `x` are cointegrated.

    Step 1: OLS `y = α + β·x` on the log prices gives the hedge ratio β and
    the spread (residual) `y − α − β·x`.
    Step 2: test the spread for a unit root. If the spread is stationary
    (rejects the unit-root null), the pair is cointegrated. We report both
    the statsmodels Engle-Granger p-value and a direct ADF p-value on the
    fitted spread — they should agree on the verdict.
    """
    config = config or SignalsConfig()
    Y = log_prices[y].to_numpy()
    X = log_prices[x].to_numpy()

    beta = float(sm.OLS(Y, sm.add_constant(X)).fit().params[1])
    _, eg_pvalue, _ = coint(Y, X)
    spread = Y - beta * X
    adf_pvalue = float(adfuller(spread, autolag="AIC")[1])

    return PairTest(
        y=y, x=x, hedge_ratio=beta,
        eg_pvalue=float(eg_pvalue), adf_pvalue=adf_pvalue,
        cointegrated=eg_pvalue < config.coint_significance,
    )


def test_all_pairs(log_prices: pd.DataFrame,
                   config: SignalsConfig | None = None) -> pd.DataFrame:
    """
    Run Engle-Granger on every ordered pair and return the results sorted by
    cointegration p-value (most cointegrated first). Ordered pairs because
    the hedge ratio from regressing y on x differs from x on y.
    """
    config = config or SignalsConfig()
    tickers = list(log_prices.columns)
    rows = []
    for y in tickers:
        for x in tickers:
            if y == x:
                continue
            t = engle_granger(log_prices, y, x, config)
            rows.append({
                "y": t.y, "x": t.x, "hedge_ratio": round(t.hedge_ratio, 4),
                "eg_pvalue": round(t.eg_pvalue, 4), "adf_pvalue": round(t.adf_pvalue, 4),
                "cointegrated": t.cointegrated,
            })
    return pd.DataFrame(rows).sort_values("eg_pvalue").reset_index(drop=True)


def johansen_rank(log_prices: pd.DataFrame,
                  config: SignalsConfig | None = None) -> pd.DataFrame:
    r"""
    Johansen trace test for the cointegration rank of the whole system.

    Unlike Engle-Granger (one pair at a time), Johansen tests all assets
    jointly and returns how many independent cointegrating relationships
    exist (rank r). We compare each trace statistic to its 95% critical value;
    the rank is the number of statistics that exceed their critical value.
    """
    config = config or SignalsConfig()
    # det_order=0 (constant in the cointegration relation), k_ar_diff=1 lag.
    result = coint_johansen(log_prices.to_numpy(), det_order=0, k_ar_diff=1)
    # critical values columns are [90%, 95%, 99%]; use the 95% column.
    crit_95 = result.cvt[:, 1]
    trace = result.lr1
    rows = []
    rank = 0
    for i, (stat, cv) in enumerate(zip(trace, crit_95)):
        exceeds = stat > cv
        rank += int(exceeds)
        rows.append({
            "hypothesis": f"r <= {i}",
            "trace_stat": round(float(stat), 3),
            "crit_95": round(float(cv), 3),
            "reject": bool(exceeds),
        })
    df = pd.DataFrame(rows)
    df.attrs["rank"] = rank
    return df


def spread_signal(log_prices: pd.DataFrame, pair: PairTest,
                  config: SignalsConfig | None = None,
                  lookback: int | None = None) -> SpreadSignal:
    r"""
    Build the mean-reversion trading signal for a cointegrated pair.

    The spread `s = y − β·x` is standardised to a z-score. A trade opens when
    the spread is stretched beyond ±entry_z (bet on reversion: short the
    spread when z is high, long it when z is low) and closes as it reverts
    inside ±exit_z. With no lookback the z-score uses the full-sample mean/std;
    with a lookback it uses a rolling window (more realistic — you would not
    know the future mean live).
    """
    config = config or SignalsConfig()
    Y = log_prices[pair.y]
    X = log_prices[pair.x]
    spread = Y - pair.hedge_ratio * X

    if lookback:
        mu = spread.rolling(lookback).mean()
        sd = spread.rolling(lookback).std()
    else:
        mu, sd = spread.mean(), spread.std()
    z = (spread - mu) / sd

    # Build the position with entry/exit hysteresis: once in a trade, hold it
    # until the spread reverts inside the exit band, rather than flipping on
    # every threshold cross.
    position = np.zeros(len(z))
    state = 0
    zv = z.to_numpy()
    for i, zi in enumerate(zv):
        if np.isnan(zi):
            position[i] = 0
            continue
        if state == 0:
            if zi > config.spread_entry_z:
                state = -1          # spread too high → short it (expect fall)
            elif zi < -config.spread_entry_z:
                state = 1           # spread too low → long it (expect rise)
        else:
            if abs(zi) < config.spread_exit_z:
                state = 0           # reverted → close
        position[i] = state

    return SpreadSignal(
        spread=spread, zscore=z,
        position=pd.Series(position, index=z.index, name="position"),
        hedge_ratio=pair.hedge_ratio,
        entry_z=config.spread_entry_z, exit_z=config.spread_exit_z,
    )
