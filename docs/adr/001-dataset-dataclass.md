# ADR-001: Bundle pipeline outputs into a single Dataset dataclass

## Status

Accepted

## Context

The data pipeline produces several related outputs: adjusted close prices, simple returns, log returns, descriptive statistics, and metadata. These are all derived from the same cleaned data in the same pipeline run.

Without bundling, the alternative looks like this:

```python
prices = get_prices(tickers)
returns = compute_returns(prices)
log_returns = compute_log_returns(prices)
```

This works until someone filters `prices` (e.g. drops a ticker with too many NaNs) but forgets to apply the same filter to `returns` and `log_returns`. Now `prices` has 9 columns and `returns` has 10. The covariance matrix is 10x10 but price-based signals reference 9 assets. Or date ranges drift — `prices` gets trimmed to `common_start` but `returns` still has earlier rows, and a merge produces misaligned data.

These bugs are particularly dangerous because they don't raise exceptions — they produce plausible but wrong results (e.g. slightly off portfolio weights) that may not surface until a backtest fails to reproduce.

## Decision

All pipeline outputs are bundled into a single `Dataset` dataclass (`math-engine/data/pipeline.py`). The pipeline constructs one `Dataset` at the end of its run, and every downstream module receives that object rather than individual DataFrames.

```python
@dataclass
class Dataset:
    prices: pd.DataFrame
    returns: pd.DataFrame
    log_returns: pd.DataFrame
    stats: pd.DataFrame
    common_start: pd.Timestamp
    metadata: dict
```

## Consequences

### Benefits

- **Alignment by construction.** All fields are derived from the same cleaned data in the same run. There is no opportunity to pass returns from one run and prices from another.
- **Single argument passing.** Downstream functions take one `Dataset` instead of three-to-six separate arguments, reducing function signature complexity.
- **Provenance travels with data.** `metadata` and `common_start` are always available for auditing without re-derivation.
- **Named, typed fields.** Typos like `ds.retunrs` are caught by linters at edit time rather than silently returning `None` at runtime (as a dict would).

### Tradeoffs

- **Adding a new output requires updating the dataclass.** Any new pipeline product (e.g. volatility estimates) means modifying `Dataset` and all call sites that construct it. This is minor friction but intentional — it forces explicit decisions about what belongs in the core data contract.
- **Memory footprint.** The object holds all outputs in memory simultaneously. For the current asset universe this is negligible, but if the pipeline scales to tick-level data this may need revisiting.
- **Not truly immutable.** Python dataclasses are mutable by default. The convention signals "don't mutate," but the compiler does not enforce it. `frozen=True` can be added if this becomes an issue.
