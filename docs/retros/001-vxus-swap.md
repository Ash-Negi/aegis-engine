# Retro-001: VXUS swap

## Date

2026-04-27

## Change

Replaced VT (Vanguard Total World Stock ETF) with VXUS (Vanguard Total International Stock ETF) in the four-asset universe. See [ADR-003](../adr/003-asset-universe-vxus.md) for the decision and alternatives.

## Expected outcome

The structural prediction was that QQQ and VT shared a primary risk driver (US mega-cap tech) at the security level, which would explain the 0.92 sample correlation observed in the Week 1 review. Replacing VT with an explicitly ex-US fund should:

- Drop QQQ-international correlation into the 0.65–0.75 range (historical norm for QQQ vs. ex-US equity).
- Reduce the condition number of Σ, which in Week 1 was being held high by the QQQ-VT axis.
- Preserve the four-asset thesis (tech / inflation / crypto / international) but make the international leg actually diversifying rather than partially redundant.

The Week 1 landmines that were *not* expected to change:

- Gold (GLDM) dominating Sharpe and historical return. Vanilla MVO will still produce a gold-heavy portfolio.
- All four assets having positive in-sample returns. Expected-return shrinkage is still required for the optimizer.
- Sample-window dominance by April 2025 tariff week, since that event affected all assets, not just VT.

## Actual outcome

Pipeline re-run on the new universe over the common 2024-01-11 → 2026-04-27 window (574 trading days):

| Metric | Week 1 (with VT, 494 days) | Now (with VXUS, 574 days) |
|--------|----------------------------|----------------------------|
| QQQ vs. international corr | 0.92 (VT) | **0.72 (VXUS)** |
| International excess kurtosis | +16.63 (VT) | **+8.47 (VXUS)** |
| Condition number κ(Σ) | ~150+ (estimated) | **39.7** |
| Fattest-tail asset | VT (+16.63) | QQQ (+11.35) |

Correlation prediction landed inside the predicted band. Conditioning improved by roughly 4×. The international leg's kurtosis dropped by half — partly the diversification benefit of going ex-US, partly window dilution from the additional 80 days of data. The two effects compound the same direction, so the swap is not solely responsible.

## What we learned

1. **The QQQ-VT correlation was structural, not statistical.** It survived a full year of distinct market events because the underlying holdings overlapped. Picking a more diversified fund (VXUS) defused it cleanly. Lesson for future asset selection: check security-level overlap before relying on naming ("global" vs "ex-US" can mean very different things).

2. **The Week 1 sample-week landmine was largely a VT artifact.** VT's +16.63 excess kurtosis was tariff-week dominated; VXUS at +8.47 has the same event in its history but does not get hijacked by it nearly as severely. Ledoit-Wolf shrinkage in Week 2 will still help but matters less than it would have with VT.

3. **The "fattest tails" baton passed to QQQ, and the data-driven narrative line in `main.py` updated automatically.** This is what auditable-by-design looks like in practice: the previous CLAUDE.md known-issue note about hardcoded narrative was already obsolete because the code computed `fattest_tails = stats["excess_kurtosis"].idxmax()` rather than hardcoding a ticker.

4. **Three of four Week-1 landmines carry over unchanged.** Gold's +36.7% return + 1.58 Sharpe still dominates the historical-mean MVO landscape. All four assets still have positive returns in-sample. FBTC's surprisingly low kurtosis (+1.88) still reflects regime luck, not asset stability. **The asset swap was a covariance fix, not an expected-returns fix.** Week 3 still needs Black-Litterman or James-Stein shrinkage on the means.

5. **A new observation surfaced: GLDM has skewness = -1.17.** Strongly negative-skewed in-sample — gold's downside moves have been chunkier than its upside. That's a separate way the normality assumption breaks (asymmetric, not just fat-tailed) and may need its own treatment in any optimizer that cares about downside risk specifically.

## What carries forward

- **Week 2 (covariance):** the conditioning argument for Ledoit-Wolf shrinkage is now weaker (κ=40 is workable) but the case for shrinkage as protection against unstable individual covariances still stands. The diagnostic comparing sample vs. shrinkage-implied weights remains the right test.
- **Week 3 (optimizer):** expected-return shrinkage is now the *primary* defense against in-sample overfit, not a secondary one. Document this prominently in the optimizer ADR.
- **Week 3 (optimizer):** consider whether downside-skew-aware risk measures (CVaR, semi-variance) deserve a slot given GLDM's -1.17 skew and FBTC's -49% drawdown.
- **Benchmarking:** VT was implicitly the "total-market" benchmark while in the universe. If a future module wants a benchmark series, VT (or ACWI/SPY) needs to be added explicitly, separate from `UNIVERSE`.
- **CLAUDE.md historical record:** the "Sample landmines (Week 1 review)" section now applies to a universe that no longer exists. It is preserved as audit trail, but Week 2 should produce a fresh baseline section for the new universe before relying on those numbers for any modeling decision.
