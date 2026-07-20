"""
Aegis Engine — Backtest Engine (Phase 1, Week 4)
================================================
Simulate holding a target portfolio through history with realistic frictions:
transaction costs, slippage, and band-triggered rebalancing.

The point of a backtest harness is not to produce a flattering return number
— it is to make the frictions VISIBLE. A frictionless backtest lies: it lets
the strategy rebalance for free every day and reports a Sharpe no live
account will ever see. The three frictions modelled here are the ones that
actually erode a rebalancing strategy:

    transaction cost  commission paid on traded notional
    slippage          the gap between decision price and fill price
    rebalancing       WHEN you trade — trade too often and costs eat you,
                      too rarely and the portfolio drifts off target

Design: the simulation runs in WEIGHT space (not share space), which keeps
it currency-agnostic and makes the drift/rebalance logic transparent. Each
day the held weights drift with returns; a rebalance back to target is
triggered only when some asset's weight has drifted more than the band away
from its target — the "no-trade region" that a real desk uses to avoid
churning on noise.

Cost convention (documented because it drives the net numbers): the cost
charged on a rebalance is  (one-way turnover) × total_cost_bps, where
one-way turnover = ½·Σ|w_new − w_old| is the fraction of the portfolio
actually traded, and total_cost_bps = transaction_cost_bps + slippage_bps
from PortfolioConfig.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import PortfolioConfig


@dataclass
class BacktestResult:
    """
    Everything the simulation produced, gross and net, for metrics and
    attribution to consume.

        equity_curve    portfolio value over time (starts at initial_capital
                        net of the day-0 buy-in cost)
        net_returns     daily returns after costs
        gross_returns   daily returns before costs
        weights         start-of-day weights (rows=dates, cols=tickers)
        turnover        one-way turnover traded each day (0 on no-rebalance days)
        costs           cost drag applied each day (fraction of portfolio)
        rebalance_dates dates a rebalance fired
        target_weights  the target the strategy rebalances toward
    """
    equity_curve: pd.Series
    net_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    rebalance_dates: list
    target_weights: pd.Series


class BacktestEngine:
    """
    Band-rebalancing backtester for a fixed target-weight portfolio.

    Frictions and the rebalancing band come from PortfolioConfig, the single
    source of truth for costs and the ±band.
    """

    def __init__(self, config: PortfolioConfig | None = None):
        self.config = config or PortfolioConfig()

    def run(self, returns: pd.DataFrame, target_weights: pd.Series) -> BacktestResult:
        """
        Simulate the strategy over `returns` (clean SIMPLE daily returns,
        rows=dates, cols=tickers) holding toward `target_weights`.

        Rebalancing rule: after each day's drift, if any asset's weight is
        more than (rebalance_band_pct/100) away from its target, snap the
        whole portfolio back to target and pay the turnover cost.
        """
        assets = list(returns.columns)
        target = target_weights.reindex(assets).to_numpy(dtype=float)
        if not np.isclose(target.sum(), 1.0):
            raise ValueError(f"target_weights must sum to 1, got {target.sum():.6f}")
        if not np.isfinite(returns.to_numpy()).all():
            raise ValueError("returns contains NaN/inf — pass the clean common period")

        R = returns.to_numpy(dtype=float)
        T = R.shape[0]
        band = self.config.rebalance_band_pct / 100.0
        cost_rate = self.config.total_cost_bps / 1e4

        # Day-0 buy-in: go from all-cash to target. One-way turnover = ½Σ|target|.
        buy_in_turnover = 0.5 * np.abs(target).sum()
        value = self.config.initial_capital * (1.0 - buy_in_turnover * cost_rate)

        w = target.copy()
        equity, net_rets, gross_rets, turns, cost_series = [], [], [], [], []
        weights_hist, rebalance_dates = [], []

        for t in range(T):
            weights_hist.append(w.copy())
            rt = R[t]
            gross = float(w @ rt)                       # day's gross return

            # Drift the weights to end of day (they no longer sum-normalize to target).
            w_end = w * (1.0 + rt) / (1.0 + gross)

            # Trigger a rebalance for the next day if drift exceeds the band.
            if np.max(np.abs(w_end - target)) > band:
                turnover = 0.5 * np.abs(target - w_end).sum()
                cost = turnover * cost_rate
                w_next = target.copy()
                rebalance_dates.append(returns.index[t])
            else:
                turnover, cost = 0.0, 0.0
                w_next = w_end

            net = gross - cost
            value *= (1.0 + net)

            gross_rets.append(gross)
            net_rets.append(net)
            turns.append(turnover)
            cost_series.append(cost)
            equity.append(value)
            w = w_next

        idx = returns.index
        return BacktestResult(
            equity_curve=pd.Series(equity, index=idx, name="equity"),
            net_returns=pd.Series(net_rets, index=idx, name="net"),
            gross_returns=pd.Series(gross_rets, index=idx, name="gross"),
            weights=pd.DataFrame(weights_hist, index=idx, columns=assets),
            turnover=pd.Series(turns, index=idx, name="turnover"),
            costs=pd.Series(cost_series, index=idx, name="cost"),
            rebalance_dates=rebalance_dates,
            target_weights=pd.Series(target, index=assets),
        )
