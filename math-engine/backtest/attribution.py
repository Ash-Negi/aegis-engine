"""
Aegis Engine — Performance Attribution
======================================
Decompose the backtest's return into where it came from: which asset earned
what, and how much the frictions cost.

Attribution is done in ARITHMETIC (daily-sum) space, not compounded space,
because only arithmetic contributions are additive — they sum exactly to the
total. The daily contribution of asset i is wₜ,ᵢ · rₜ,ᵢ (its weight times its
return that day); summed over time these partition the total gross return
exactly, since Σᵢ wₜ,ᵢrₜ,ᵢ = the portfolio's gross return on day t. Cost drag
is then subtracted as its own line so the "what the strategy paid to trade"
number is explicit rather than buried in the net figure.

(The compounded total return differs slightly from the arithmetic sum because
of reinvestment; the arithmetic decomposition is the honest way to answer
"who contributed what", and the gap is itself reported.)
"""

from dataclasses import dataclass

import pandas as pd

from backtest.engine import BacktestResult


@dataclass
class Attribution:
    """Where the return came from, in additive arithmetic terms."""
    contribution: pd.Series      # per-asset arithmetic contribution to gross
    gross_return_arith: float    # Σ daily gross = Σ contributions
    total_cost: float            # Σ daily cost drag
    net_return_arith: float      # gross_arith − total_cost
    total_turnover: float        # Σ one-way turnover traded
    n_rebalances: int            # how many times the band fired


def return_attribution(result: BacktestResult, returns: pd.DataFrame) -> Attribution:
    """
    Attribute the backtest's gross return to individual assets and quantify
    cost drag.

    Args:
        result:  a BacktestResult from BacktestEngine.run.
        returns: the SAME simple-return frame the backtest was run on.
    """
    aligned = returns.reindex(index=result.weights.index, columns=result.weights.columns)
    daily_contrib = result.weights * aligned         # wₜ,ᵢ · rₜ,ᵢ
    contribution = daily_contrib.sum(axis=0)         # per asset, summed over time

    gross_arith = float(result.gross_returns.sum())
    total_cost = float(result.costs.sum())

    return Attribution(
        contribution=contribution,
        gross_return_arith=gross_arith,
        total_cost=total_cost,
        net_return_arith=gross_arith - total_cost,
        total_turnover=float(result.turnover.sum()),
        n_rebalances=len(result.rebalance_dates),
    )
