# ADR-003: Replace VT with VXUS in the asset universe

## Status

Accepted (2026-04-27)

## Context

The Aegis Engine asset universe is built around a thesis: each asset hedges a *different* failure mode, so the four positions span as many independent risk drivers as possible. This is what makes the universe a meaningful starting point for an optimizer — if two assets share the same primary risk driver, the optimizer is effectively choosing between two slightly different versions of the same bet, and the second asset adds noise rather than diversification.

VT (Vanguard Total World Stock ETF) was originally chosen for the "global diversification" leg. The problem surfaced in the Week 1 data review (CLAUDE.md, 2026-04-16):

1. **QQQ-VT correlation = 0.92** in the common 2024-01-11 → 2025-12-30 sample. The covariance matrix is full-rank but ill-conditioned along that axis.
2. The mechanism is structural, not statistical noise. VT is roughly 60% US equity by market cap, of which a large share is the same mega-cap names that dominate QQQ (AAPL, MSFT, NVDA, AMZN, META, GOOGL, etc.). At the security level, QQQ and VT overlap heavily.
3. The "global diversification" role is therefore overstated — VT hedges US risk with more US risk plus some international exposure on top.

This was flagged as a **landmine for Week 2**: naïve `np.linalg.inv(Σ)` on the sample covariance produces brittle weights along the QQQ-VT direction. Ledoit-Wolf shrinkage was the planned mitigation, and it does help, but it treats the symptom rather than the underlying universe-design problem.

VXUS (Vanguard Total International Stock ETF) is explicitly ex-US (developed + emerging markets). Security-level overlap with QQQ drops to zero. Historical correlation between QQQ and VXUS runs in the 0.65–0.75 range — meaningfully lower independent risk content than QQQ-VT's 0.92.

## Decision

Replace VT with VXUS in `UNIVERSE` (`math-engine/config.py`). The asset's role in the thesis becomes "Ex-US diversification" — an honest description of what the security actually does. Initial weight remains 0.30 (no change to the baseline allocation structure).

The four-asset structure is unchanged:

| Ticker | Role |
|--------|------|
| QQQ | Tech beta / equity risk premium |
| GLDM | Inflation / monetary debasement hedge |
| FBTC | Stochastic noise / institutional capture hedge |
| VXUS | Ex-US (developed + emerging) diversification |

## Consequences

### Benefits

- **Cleaner thesis.** The four assets now span four genuinely different risk drivers. The "global diversification" leg actually diversifies away from the US equity risk that QQQ already provides, rather than reinforcing it.
- **Better-conditioned covariance.** Lower QQQ-VXUS correlation (vs. QQQ-VT) means smaller condition number on Σ, which means more stable optimizer weights without leaning as hard on shrinkage as a corrective.
- **Honest naming.** "Global diversification" implied VT diversified globally; in practice it diversified ~40% globally and 60% reinforced US exposure. "Ex-US diversification" is descriptive of what VXUS actually delivers.

### Tradeoffs

- **Loss of total-market benchmark in the optimizable universe.** VT was conveniently a one-line "the whole stock market." VXUS is not — it's the international half. If a future module wants to compare the engine's portfolio against a total-market benchmark, VT (or ACWI, SPY, etc.) will need to be added as a *non-optimizable* benchmark series, separate from `UNIVERSE`.
- **Week 1 review numbers are now historical artifacts.** The sample landmines documented in CLAUDE.md (kurtosis values, 0.92 correlation, tariff-week dominance) were measured on VT data. They remain valuable as the audit trail of the decision but do not describe the current sample. Week 2 will produce a fresh baseline on the new universe.
- **One stale data cache.** `math-engine/data/store/raw_prices.csv` contained VT prices and was deleted as part of this change so the pipeline re-fetches with VXUS on next run.

### Alternatives considered

- **Keep VT, lean harder on shrinkage.** Treats the symptom. The 0.92 correlation is real and structural; shrinkage reduces its impact on optimizer brittleness but doesn't change the fact that two positions are betting on substantially the same thing.
- **Replace VT with ACWI ex-US (IXUS) or VEU.** Functionally similar to VXUS. VXUS chosen for liquidity (highest AUM in this niche) and Vanguard cost structure (0.07% expense ratio).
- **Drop the fourth slot entirely, run a 3-asset universe.** Considered and rejected. Three assets is too few to demonstrate the covariance / optimizer machinery meaningfully, and removing the international leg would over-index the engine on US equity.
