"""
Aegis Engine — Performance Metrics
==================================
The standard risk/return summary of a return stream. Kept separate from the
engine so the same metrics apply to any series — a backtest, a single asset,
a benchmark.

Every metric is annualized with the same trading_days convention used
everywhere else in the engine, so numbers are comparable across modules.
"""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Annualized performance summary of a daily return series."""
    total_return: float      # cumulative return over the whole window
    cagr: float              # compound annual growth rate
    ann_return: float        # arithmetic annualized return (mean × 252)
    ann_volatility: float    # annualized standard deviation
    sharpe: float            # (ann_return − rf) / ann_volatility
    max_drawdown: float      # worst peak-to-trough decline (≤ 0)
    calmar: float            # cagr / |max_drawdown|
    win_rate: float          # fraction of positive days

    def as_dict(self) -> dict:
        return asdict(self)


def performance_metrics(
    returns: pd.Series,
    trading_days: int = 252,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Compute the performance summary of a daily (simple) return series.

    max_drawdown uses the compounded equity curve, not the returns directly,
    because a drawdown is a path-dependent quantity — you cannot read the
    worst peak-to-trough loss off the mean and variance.
    """
    r = returns.dropna()
    n = len(r)
    if n == 0:
        raise ValueError("empty return series")

    total_return = float((1.0 + r).prod() - 1.0)
    cagr = float((1.0 + total_return) ** (trading_days / n) - 1.0) if total_return > -1 else -1.0
    ann_return = float(r.mean() * trading_days)
    ann_vol = float(r.std() * np.sqrt(trading_days))
    # Guard with an epsilon, not > 0: a constant series has float-dust
    # variance (~1e-18), which would otherwise explode the Sharpe ratio.
    sharpe = float((ann_return - risk_free_rate) / ann_vol) if ann_vol > 1e-12 else 0.0

    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.inf

    win_rate = float((r > 0).mean())

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        ann_return=ann_return,
        ann_volatility=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        win_rate=win_rate,
    )
