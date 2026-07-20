"""
Aegis Engine — Backtest Harness (Phase 1, Week 4)
=================================================
Simulate a target portfolio through history with realistic frictions, then
measure and attribute the result.

    engine       BacktestEngine — band-rebalancing simulation with costs
    metrics      performance_metrics — CAGR, Sharpe, max drawdown, Calmar…
    attribution  return_attribution — per-asset contribution + cost drag

The harness answers the question every optimizer output raises: "yes it looks
good on paper, but what survives after you actually have to trade it?"
"""

from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import performance_metrics, PerformanceMetrics
from backtest.attribution import return_attribution, Attribution

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "performance_metrics",
    "PerformanceMetrics",
    "return_attribution",
    "Attribution",
]
