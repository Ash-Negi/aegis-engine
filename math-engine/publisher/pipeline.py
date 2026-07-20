"""
Aegis Engine — Signal Generation Pipeline (Phase 3)
===================================================
The end-to-end path from raw prices to a publishable signal. This is where
Phases 1 and 2 meet the execution layer:

    prices → Ledoit-Wolf Σ → Bayes-Stein μ → max-Sharpe w   (Phase 1)
           → HMM regime → regime tilt on w                  (Phase 2)
           → TargetWeightSignal                             (Phase 3)

Kept separate from `redis_publisher` so the signal can be generated and
inspected without a Redis server running — `generate_signal()` is a pure
function of market data, which is what makes it testable.
"""

import numpy as np
import pandas as pd

from config import (
    DataConfig, OptimizerConfig, PortfolioConfig, SignalsConfig,
    ExecutionConfig, TICKERS,
)
from data.pipeline import DataPipeline
from covariance import ledoit_wolf_covariance
from optimizer import estimate_expected_returns, max_sharpe_portfolio
from signals import RegimeDetector, AdaptiveWeightEngine
from publisher.contract import TargetWeightSignal, build_signal


def generate_signal(
    opt_config: OptimizerConfig | None = None,
    sig_config: SignalsConfig | None = None,
    port_config: PortfolioConfig | None = None,
    exec_config: ExecutionConfig | None = None,
    data_config: DataConfig | None = None,
) -> TargetWeightSignal:
    """Run the full stack on the latest market data and return a signal."""
    opt_config = opt_config or OptimizerConfig()
    sig_config = sig_config or SignalsConfig()
    port_config = port_config or PortfolioConfig()
    exec_config = exec_config or ExecutionConfig()
    data_config = data_config or DataConfig()

    pipeline = DataPipeline(data_config)
    dataset = pipeline.run(TICKERS, use_cache=True)
    common = pipeline.get_common_period(dataset)
    log_returns = common["log_returns"]
    returns = common["returns"]

    # Phase 1 — the risk-optimal base portfolio.
    cov_daily = ledoit_wolf_covariance(log_returns)
    cov = cov_daily * opt_config.trading_days_per_year
    er = estimate_expected_returns(log_returns, cov_daily, opt_config)
    base = max_sharpe_portfolio(
        er.mu, cov, data_config.risk_free_rate, opt_config,
    )

    # Phase 2 — what market are we in, and how should that change the base?
    # Fit on the equal-weight return: a market proxy that does not presume
    # the optimizer's own answer.
    regimes = RegimeDetector(n_regimes=sig_config.n_regimes).fit(returns.mean(axis=1))
    adaptive = AdaptiveWeightEngine(sig_config).tilt(base, regimes.current)

    return build_signal(
        target_weights=adaptive.weights,
        regime=regimes.current,
        regime_confidence=float(regimes.proba[regimes.current].iloc[-1]),
        as_of=returns.index[-1],
        expected_turnover=adaptive.turnover,
        rebalance_band_pct=port_config.rebalance_band_pct,
        config=exec_config,
    )
