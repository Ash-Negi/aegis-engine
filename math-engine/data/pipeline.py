"""
Aegis Engine — Data Pipeline
=============================
Fetches, cleans, transforms, and stores daily price data for the asset universe.

This module is the first thing that runs in the Aegis system. Every downstream
module (covariance estimation, optimizer, signals) depends on clean data.
If this is wrong, everything is wrong.

Design principles:
    1. Raw data is always preserved. Never modify the original fetch.
    2. Every transformation is a separate, testable function.
    3. The pipeline produces three outputs:
       - prices:      clean adjusted close prices (DatetimeIndex × tickers)
       - returns:     simple daily returns
       - log_returns: log daily returns (used by most models)
    4. All data issues are logged, not silently hidden.

Usage:
    from data.pipeline import DataPipeline
    from config import DataConfig, TICKERS

    pipeline = DataPipeline(DataConfig())
    dataset = pipeline.run(TICKERS)

    dataset.prices      # pd.DataFrame of clean prices
    dataset.returns      # pd.DataFrame of simple returns
    dataset.log_returns  # pd.DataFrame of log returns
    dataset.stats        # pd.DataFrame of descriptive statistics
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise ImportError(
        "yfinance is required.  Install it with: pip install yfinance"
    )

from config import DataConfig, DATA_DIR

#Suppress noisy yfinance warnings - they clutter the output
#without providing actionable information
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

logger = logging.getLogger(__name__)

# ─── Data Container ────────────────

@dataclass
class Dataset:
    """
    Immutable container for all pipeline outputs.

    Having a single object that holds prices, returns, and stats means
    downstream modules receive everything they need in one argument.
    This prevents the common bug of passing misaligned DataFrames.
    """
    prices: pd.DataFrame       # Adjusted close prices
    returns: pd.DataFrame      # Simple daily returns: (P_t - P_{t-1}) / P_{t-1}
    log_returns: pd.DataFrame  # Log daily returns: ln(P_t / P_{t-1})
    stats: pd.DataFrame        # Descriptive statistics
    common_start: pd.Timestamp # First date where ALL assets have data
    metadata: dict              # Data quality info for auditing


# ─── Pipeline ──────────────────────────────────────────────────────────

class DataPipeline:
    """
    Fetches and processes daily price data for the Aegis asset universe.

    The pipeline is idempotent: running it twice with the same config
    produces the same output. It can also load from cached CSV files
    to avoid hitting the yfinance API repeatedly during development.
    """

    def __init__(self, config:DataConfig | None = None):
        self.config = config or DataConfig()
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, tickers: list[str], use_cache:bool = True) -> Dataset:
        """
        Execute the full pipeline: fetch → clean → transform → stats.

        Args:
            tickers: List of ticker symbols to fetch.
            use_cache: If True, load from local CSV if available.
            Set to False to force a fresh download.

        Returns:
            Dataset containing prices, returns, log_returns, and stats.
        """

        logger.info(f"Running data pipeline for {tickers}")
        logger.info(f"Period: {self.config.start_date} -> {self.config.end_date}")

        #Step 1: Fetch raw prices
        raw_prices = self._fetch(tickers, use_cache)

        #Step 2: Clean - handle missing data, align dates
        clean_prices, metadata = self._clean(raw_prices, tickers)

        #Step 3: Compute returns
        simple_returns = self._compute_simple_returns(clean_prices)
        log_returns = self._compute_log_returns(clean_prices)

        #Step 4: Descriptive statistics
        stats = self._compute_stats(clean_prices, simple_returns, log_returns)

        #Step 5: Determine common start date (all assets have data)
        common_start = clean_prices.dropna().index[0]

        logger.info(f"Pipeline complete. Common start date: {common_start.date()}")
        logger.info(f"Total trading days: {len(clean_prices.loc[common_start:])}")

        return Dataset(
            prices=clean_prices,
            returns=simple_returns,
            log_returns=log_returns,
            stats=stats,
            common_start=common_start,
            metadata=metadata,
        )

    # ── Step 1: Fetch ─────────────────────────────────────────────────────

    def _fetch(self, tickers: list[str], use_cache:bool) -> pd.DataFrame:
        """
        Fetch adjusted close prices from yfinance or local cache.

        Why adjusted close?
        Raw close prices don't account for dividends and stock splits.
        If a stock pays a $1 dividend, the raw close drops by $1 on the
        ex-date, creating a fake negative return. Adjusted close backs
        out these events so returns reflect actual investor experience.
        """
        cache_path = DATA_DIR / "raw_prices.csv"

        #check cache
        #if it is, check if every requested ticker is in columns
        #if even one ticker is missing, throw the cache away and re-fetch
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            # Verify all requested tickers are in the cache
            missing = set(tickers) - set(prices.columns)
            if not missing:
                return prices
            logger.info(f"Cache missing tickers {missing}, fetching fresh data")

        logger.info(f"Fetching data from yfinance for {tickers}...")
        raw = yf.download(
            tickers,
            start=self.config.start_date,
            end=self.config.end_date,
            # auto_adjust=True applies the CRSP back-adjustment methodology:
            # whenever a dividend is paid or a split occurs, yfinance scales
            # all prior prices down proportionally so the return on the ex-date
            # looks flat. Without this, a dividend payment would appear as a
            # price drop, creating a fake negative return in our series.
            auto_adjust=True,
            progress=False,  # suppress tqdm download bar
        )

        # When multiple tickers are requested, yfinance returns a DataFrame with
        # a two-level (MultiIndex) column structure:
        #
        #               Close                    High         ...
        #   Ticker      FBTC   GLDM   QQQ   VT   FBTC  GLDM  ...
        #   Date
        #   2024-01-11  42.31  35.10  423.5  99.2  ...
        #
        # The outer level is the field (Close, High, Low, Volume, etc.) and the
        # inner level is the ticker. raw["Close"] slices out just the Close field,
        # giving a plain DataFrame with one column per ticker — which is all we need.
        #
        # When a single ticker is requested, yfinance returns flat columns instead
        # (no outer level), so we handle both cases to keep the function general.
        if isinstance(raw.columns, pd.MultiIndex):
            # [tickers] reorders columns to match the input list order,
            # since yfinance may return them alphabetically.
            prices = raw["Close"][tickers].copy()
        else:
            #.copy() ensures downstream stages own their data
            prices = raw[["Close"]].copy()
            prices.columns = tickers

        # .copy() above is intentional: slicing a MultiIndex DataFrame in pandas
        # returns a view, not a new object. Mutating a view can silently corrupt
        # the original or raise a SettingWithCopyWarning. Copying here ensures
        # every downstream stage owns its data independently.

        #Cache for future runs
        prices.to_csv(cache_path)
        logger.info(f"Cached raw prices to {cache_path}")

        return prices

    # ── Step 2: Clean ─────────────────────────────────────────────────────

    def _clean(self, prices: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, dict]:
        """
        Clean raw price data and produce an audit trail.

        Cleaning steps:
        1. Drop rows where ALL tickers are NaN (non-trading days)
        2. Forward-fill small gaps (holidays differ across exchanges)
        3. Log any remaining issues

        Why forward-fill?
        Different exchanges have different holiday calendars. GLDM might
        not trade on a day when QQQ does. Forward-filling means "the last
        known price is still the best estimate." We limit this to avoid
        hiding real data problems (like a delisted ticker).

        Returns:
            Tuple of (cleaned prices DataFrame, metadata dict)
        """

        metadata = {"raw_rows":len(prices), "tickers": tickers}

        # Drop full-NaN rows (weekends are already excluded by yfinance)
        prices = prices.dropna(how="all").copy()
        metadata["after_drop_all_nan"] = len(prices)

        # Report per-ticker data availability
        metadata["availability"] = {}
        for ticker in tickers:
            valid_count = prices[ticker].notna().sum()
            first_valid = prices[ticker].first_valid_index()
            last_valid = prices[ticker].last_valid_index()
            metadata["availability"][ticker] = {
                "valid_days": int(valid_count),
                "total_days": len(prices),
                "coverage_pct" : round(valid_count / len(prices) * 100, 1),
                "first_date": str(first_valid.date()) if first_valid else None,
                "last_date": str(last_valid.date()) if last_valid else None
            }
            logger.info(
                f"  {ticker}: {valid_count}/{len(prices)} days "
                f"({metadata['availability'][ticker]['coverage_pct']}%) "
                f"from {first_valid}"
            )

        # Forward-fill gaps up to the configured limit
        pre_fill_nans = prices.isna().sum().to_dict()
        prices = prices.ffill(limit=self.config.forward_fill_limit)
        post_fill_nans = prices.isna().sum().to_dict()

        metadata["nans_filled"] = {
            ticker: pre_fill_nans[ticker] - post_fill_nans[ticker]
            for ticker in tickers
        }
        metadata["nans_remaining"] = post_fill_nans

        # Log warnings for any remaining NaN values
        for ticker in tickers:
            remaining = post_fill_nans[ticker]
            if remaining > 0:
                logger.warning(
                    f"  {ticker}: {remaining} NaN values remain after "
                    f"forward-fill (limit={self.config.forward_fill_limit}). "
                    f"This is expected for FBTC before its inception date."
                )

        return prices, metadata

    # ── Step 3: Returns ───────────────────────────────────────────────────

    @staticmethod
    def _compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Simple returns: R_t = (P_t - P_{t-1}) / P_{t-1}

        Properties:
        - Intuitive: a return of 0.02 means "up 2%"
        - NOT additive over time: (1+R1)(1+R2) ≠ 1+R1+R2
        - Portfolio return IS the weighted sum of asset returns
          R_portfolio = Σ w_i * R_i  (this is exact, not approximate)

        We keep simple returns because portfolio-level calculations
        require them. You cannot compute portfolio returns from log
        returns without converting back first.
        """
        return prices.pct_change()

    @staticmethod
    def _compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Log returns: r_t = ln(P_t / P_{t-1})

        Properties:
        - Additive over time: r(t1→t3) = r(t1→t2) + r(t2→t3)
        - Approximately normal (better than simple returns)
        - Required for continuous-time models (GBM, Ito calculus)
        - For small daily moves: log_return ≈ simple_return
          The difference only matters for large moves (FBTC)

        Most of the math engine operates on log returns.
        The optimizer converts back to simple returns when needed
        for portfolio-level calculations.
        """
        return np.log(prices / prices.shift(1))

    # ── Step 4: Descriptive Statistics ────────────────────────────────────

    def _compute_stats(
            self,
            prices: pd.DataFrame,
            returns:pd.DataFrame,
            log_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute descriptive statistics for the asset universe.

        These stats serve three purposes:
        1. Sanity check — do the numbers look reasonable?
        2. Baseline — what does each asset look like before optimization?
        3. Input validation — are there data issues hiding in the stats?

        If annualized vol for QQQ comes back as 2%, something is wrong.
        If FBTC's vol is 200%, something might also be wrong. Knowing
        the "expected" range for each stat helps you catch bugs early.
        """
        trading_days = self.config.trading_days_per_year

        # Use the common period where all assets have data
        common_start = prices.dropna().index[0]
        r = returns.loc[common_start:].dropna()
        lr = log_returns.loc[common_start:].dropna()
        p = prices.loc[common_start:]

        stats_records = []
        for ticker in prices.columns:
            daily_r = r[ticker]
            daily_lr = lr[ticker]

            #Annualized return (from log returns - geometrically correct)
            ann_return = daily_lr.mean() * trading_days

            #Annualized volatility
            ann_vol = daily_lr.std() * np.sqrt(trading_days)

            #Sharpe ratio(annualized, excess over risk-free)
            #risk_free_rate now lives in DataConfig - no fallback needed
            rf_daily = self.config.risk_free_rate/trading_days
            excess_daily = daily_r - rf_daily
            sharpe = (
                (excess_daily.mean() / daily_r.std()) * np.sqrt(trading_days)
                if daily_r.std() > 0
                else 0.0
            )

            #Maximum drawdown
            cumulative = (1+daily_r).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()

            # Skewness and kurtosis of log returns
            # Skew < 0 means fat left tail (more large losses than gains)
            # Excess kurtosis > 0 means fatter tails than normal distribution
            # Both matter because MVO assumes normality — these tell you
            # how badly that assumption is violated for each asset
            skew = daily_lr.skew()
            excess_kurt = daily_lr.kurtosis() # pandas returns excess kurtosis

            # Total return over the common period
            total_return = (p[ticker].iloc[-1]/p[ticker].iloc[0]) - 1

            stats_records.append({
                "ticker": ticker,
                "total_return_pct": round(total_return * 100, 2),
                "ann_return_pct": round(ann_return * 100, 2),
                "ann_vol_pct": round(ann_vol * 100, 2),
                "sharpe": round(sharpe, 3),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "skewness": round(skew, 3),
                "excess_kurtosis": round(excess_kurt, 3),
                "daily_win_rate_pct": round((daily_r > 0).mean() * 100, 1),
                "best_day_pct": round(daily_r.max() * 100, 2),
                "worst_day_pct": round(daily_r.min() * 100, 2),
                "trading_days": len(daily_r),
            })

        return pd.DataFrame(stats_records).set_index("ticker")

    # ── Utilities ─────────────────────────────────────────────────────────

    def get_common_period(self, dataset: Dataset) -> dict:
        """
        Return prices, returns, and log_returns for the common period only
        (where all assets have data). Use this for any cross-asset analysis
        like covariance estimation.
        """
        start = dataset.common_start
        return {
            "prices": dataset.prices.loc[start:].dropna(),
            "returns": dataset.returns.loc[start:].dropna(),
            "log_returns": dataset.log_returns.loc[start:].dropna(),
        }

    def get_full_history(self, dataset: Dataset, ticker: str) -> dict:
        """
        Return the full available history for a single asset.
        Useful for per-asset analysis where you want maximum data.
        """
        first_valid = dataset.prices[ticker].first_valid_index()
        return {
            "prices": dataset.prices[ticker].loc[first_valid:].dropna(),
            "returns": dataset.returns[ticker].loc[first_valid:].dropna(),
            "log_returns": dataset.log_returns[ticker].loc[first_valid:].dropna(),
        }











