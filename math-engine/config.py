"""
Aegis Engine - Configuration
============================
Every tunable parameter lives here. No magic numbers buried in logic.

Design Principle: if you need to change a number to test a hypothesis, you should only need to change it in one place.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ─── Paths ───────────────────────────────
#Pipeline uses this to cache CSV files and save plots.
#Pipeline imports DATA_DIR from config to know where to save cached price data and output charts.
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "store"

# ─── Asset Universe ------
#quick way to make classes
@dataclass(frozen=True)
class Asset:
    """Metadata for a single asset in the universe."""
    ticker: str
    name: str
    role: str #what this asset does in the portfolio
    asset_class: str

#The four assets in the Aegis universe
#Each one hedges a different failure mode:
#   QQQ  — captures equity risk premium and tech growth
#   GLDM — hedges monetary debasement and inflation
#   FBTC — hedges institutional/currency capture (high vol, financial repression)
#   VXUS — hedges US-concentration risk through ex-US (developed + emerging) equity

UNIVERSE = [
    Asset("QQQ","Invesco QQQ Trust","Tech Beta","equity"),
    Asset("GLDM", "SPDR Gold MiniShares","Inflation Hedge","commodity"),
    Asset("FBTC", "Fidelity Wise Origin Bitcoin Fund","Financial Repression","crypto"),
    Asset("VXUS", "Vanguard Total International Stock ETF","Ex-US Diversification", "equity")
]

TICKERS = [asset.ticker for asset in UNIVERSE]

# ─── Data Pipeline Config ───────────────────

@dataclass
class DataConfig:
    """
    Parameters for data acquisition and cleaning.

    Notes on date ranges:
        FBTC launched January 11, 2024. Any backtest that includes FBTC
        can only start from that date. We fetch a longer history for the
        other assets so you can run analyses with and without FBTC.

    Notes on log returns:
        We use log returns (not simple returns) throughout the engine because:
        1. They are additive over time: log_ret(t1→t3) = log_ret(t1→t2) + log_ret(t2→t3)
        2. They are approximately normally distributed (a key assumption in MVO)
        3. They make the math cleaner for continuous-time models (GBM, O-U)
        For small daily moves, log returns ≈ simple returns. The difference
        matters over longer horizons and for volatile assets like FBTC.
    """
    #Full history start - captures pre-FBTC data for assets that have it
    start_date: str = "2019-01-01"
    end_date: str = "2026-12-31"

    #FBTC inception - the engine will handle this automatically
    fbtc_inception: str = "2024-01-11"

    #Data cleaning
    forward_fill_limit: int = 5 #Max consecutive NaN days to forward-fill
    #beyond this limit, something is wrong with the data, not just a holiday

    #Annualization factor
    trading_days_per_year: int = 252

    # Risk-free rate (annualized) — for Sharpe ratio calculation
    # Approximate T-bill rate for 2023-2025 regime
    # Lives here (not PortfolioConfig) because the pipeline uses it
    # to compute stats. One source of truth for this number.
    # TODO: update when rate regime changes materially
    risk_free_rate: float = 0.05


# ─── Covariance Estimation Config ─────────────
@dataclass
class CovarianceConfig:
    """
    Parameters for the Week 2 covariance-estimation layer.

    The covariance matrix Σ is the single most important input to
    mean-variance optimization: the optimizer inverts it, so any
    instability in Σ is amplified directly into the weights. Every knob
    that changes how Σ is estimated lives here, one source of truth.
    """

    # Return basis. The whole engine models log returns (see DataConfig),
    # so covariance is estimated on log returns too. Kept explicit so a
    # future experiment can switch to simple returns in exactly one place.
    use_log_returns: bool = True

    # ── Sample covariance ──
    # Bessel's correction: divide by (T-1) instead of T so the estimator
    # is unbiased for the population covariance. This is the textbook
    # baseline and matches np.cov / pandas .cov() defaults.
    sample_ddof: int = 1

    # ── EWMA (RiskMetrics) ──
    # λ = 0.94 is J.P. Morgan RiskMetrics' daily standard. It sets the
    # memory of the estimator: the weight on an observation i days old is
    # ∝ λ^i, a half-life of ln(0.5)/ln(0.94) ≈ 11.2 trading days.
    ewma_lambda: float = 0.94
    # RiskMetrics assumes a zero daily mean and does NOT demean returns
    # before forming the second moment. We follow that convention so the
    # λ=0.94 calibration is used as intended; flip to True for research.
    ewma_demean: bool = False

    # ── Annualization ──
    # Daily covariance scales linearly with horizon under i.i.d.:
    # Σ_annual = 252 · Σ_daily. Because this is a scalar multiple it leaves
    # the CONDITION NUMBER and correlation structure unchanged, so every
    # conditioning diagnostic is identical daily vs. annualized — the
    # factor is purely for human-readable volatilities.
    trading_days_per_year: int = 252

    # ── Diagnostics ──
    # An eigenvalue counts as a "real" risk factor when it explains at
    # least this share of total variance. With four assets, factors below
    # a few percent are noise the optimizer should not chase.
    factor_variance_threshold: float = 0.05


# ─── Optimizer Config (Week 3) ────────────────
@dataclass
class OptimizerConfig:
    """
    Parameters for the Week 3 mean-variance optimization layer.

    MVO solves for weights w ∝ Σ⁻¹μ. Two inputs make or break it, and both
    are landmines flagged in the Week 1 review:
      * Σ is ill-conditioned → the inverse is brittle. Handled upstream by
        the covariance layer (Ledoit-Wolf) and by solving rather than
        explicitly inverting.
      * μ (historical mean returns) is extremely noisy → the optimizer
        chases in-sample flukes and produces concentrated, levered bets.
        Handled here by expected-return shrinkage.
    """

    # Which covariance estimator feeds the optimizer. Ledoit-Wolf by
    # default: it is the best-conditioned, so Σ⁻¹ is safest.
    covariance_estimator: str = "ledoit_wolf"   # sample | ewma | ledoit_wolf

    # ── Expected-return shrinkage ──
    # Vanilla MVO with raw historical means overfits: in this sample every
    # asset had large positive returns, so unconstrained MVO piles into
    # whichever had the best in-sample Sharpe (gold). Shrinkage pulls the
    # noisy mean vector toward a common target so the optimizer stops
    # trusting spurious differences.
    return_shrinkage: str = "bayes_stein"       # none | bayes_stein | grand_mean
    # Only used when return_shrinkage == "grand_mean": a fixed intensity in
    # [0, 1] toward the cross-sectional average return. Bayes-Stein instead
    # derives its intensity from the data.
    grand_mean_intensity: float = 0.5

    # ── Constraints (for the constrained optimizer) ──
    long_only: bool = True                      # no shorting: w_i ≥ 0
    max_weight: float = 1.0                     # per-asset ceiling
    # Sector (asset-class) caps: total weight in a class may not exceed the
    # cap. "equity" groups QQQ + VXUS, which are 0.72-correlated — without a
    # cap the optimizer would happily double up on the same equity beta.
    sector_caps: dict = field(default_factory=lambda: {"equity": 0.65})

    # ── Efficient frontier ──
    frontier_points: int = 50                   # resolution of the frontier sweep

    # ── Reporting basis ──
    annualize: bool = True
    trading_days_per_year: int = 252


# ─── Signals Config (Phase 2) ─────────────────
@dataclass
class SignalsConfig:
    """
    Parameters for the Phase 2 adaptive layer: regime detection, regime-based
    weight tilts, and cointegration signals.
    """

    # ── HMM regime detection ──
    n_regimes: int = 3               # low-vol trending / high-vol mean-rev / crisis
    hmm_random_state: int = 42       # EM is seed-sensitive; pin for reproducibility

    # ── Regime tilts ──
    # Multiplicative tilts applied to the base optimal weights by asset class,
    # per regime. Risk-on (low-vol trending) leans into equity and crypto and
    # trims gold; crisis does the opposite — gold up, equity/crypto down.
    # Weights are renormalized to sum to 1 after tilting, so these are relative.
    regime_tilts: dict = field(default_factory=lambda: {
        "low_vol_trending": {"equity": 1.25, "commodity": 0.85, "crypto": 1.15},
        "high_vol_meanrev": {"equity": 1.00, "commodity": 1.00, "crypto": 1.00},
        "crisis":           {"equity": 0.60, "commodity": 1.50, "crypto": 0.50},
    })

    # ── Cointegration / spread signals ──
    coint_significance: float = 0.05  # p-value / critical-level threshold
    spread_entry_z: float = 2.0       # enter a spread trade beyond ±2σ
    spread_exit_z: float = 0.5        # exit as the spread reverts toward the mean


# ─── Portfolio Config ─────────────────────
@dataclass
class PortfolioConfig:
    """
    Parameters for portfolio construction and rebalancing.

    These are INITIAL values.  Phase 1 will compute optimal weights from the math.  These serve as a baseline for comparison.
    """

    #Starting weights - the "naive" allocation before optimization
    # This is what you compare your optimizer against
    # we have to use field() with default_factory to generate a fresh independent dictionary for each instance
    initial_weights: dict = field(default_factory=lambda: {
        "QQQ":  0.30,
        "GLDM": 0.25,
        "FBTC": 0.15,
        "VXUS": 0.30,
    })

    #Transaction costs(basis points, round_trip)
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0

    # Rebalancing bands (used in tactical layer)
    # A weight must drift this far from target before triggering rebalance
    rebalance_band_pct: float = 5.0 # ±5% of target weight

    # Initial capital for backtest
    initial_capital: float = 100_000.0

    def __post_init__(self) -> None:
        self.validate()

    @property
    def total_cost_bps(self) -> float:
        return self.transaction_cost_bps + self.slippage_bps

    def validate(self) -> None:
        """Verify weights sum to 1 and all tickers are covered"""
        weight_sum = sum(self.initial_weights.values())
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(f"Weights must sum to 1.0, it is currently {weight_sum:.6f}")
        for ticker in TICKERS:
            if ticker not in self.initial_weights:
                raise ValueError(f"Missing weight for {ticker}")




