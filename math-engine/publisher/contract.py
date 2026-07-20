"""
Aegis Engine — Signal Contract (Phase 3)
========================================
The wire format between the Python math engine and the Java execution
engine. This module is the single source of truth for that payload: if the
shape changes here, `schema_version` bumps and the Java record changes to
match (see `execution-engine/.../TargetWeightSignal.java`).

Two decisions encoded in this schema are worth stating outright.

**The engine publishes target WEIGHTS, not orders.** A signal says "the
portfolio should be 19% QQQ" — it does not say "buy 43 shares of QQQ". The
math engine does not know current positions, account equity, or what is
already working in the market; the execution engine does. Sending desired
*state* rather than imperative *commands* means a replayed or duplicated
signal is harmless (it re-asserts the same target), and the two services can
be reasoned about independently.

**Every signal is idempotent and expiring.** `signal_id` lets the consumer
detect a redelivery and skip it; `valid_until` lets it detect a stale signal
and refuse to trade. Both are required because Redis pub/sub gives at-most-
once delivery with no ordering guarantee across reconnects — the consumer
cannot assume it saw every message, or saw them in order.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import ExecutionConfig


@dataclass(frozen=True)
class TargetWeightSignal:
    """One published rebalance target. Serialises to the Redis payload."""

    schema_version: int
    signal_id: str              # UUID — consumer dedupe key
    generated_at: str           # ISO-8601 UTC, when the engine produced this
    valid_until: str            # ISO-8601 UTC, after which the signal is stale
    as_of: str                  # date of the last market data used (YYYY-MM-DD)
    regime: str                 # detected regime name
    regime_confidence: float    # P(regime | data) for the labelled regime
    target_weights: dict        # ticker → weight, sums to 1
    expected_turnover: float    # ½Σ|new − old|, the trade this implies
    rebalance_band_pct: float   # no-trade band the execution engine should apply

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> TargetWeightSignal:
        return cls(**json.loads(payload))

    def is_stale(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return now > datetime.fromisoformat(self.valid_until)


def build_signal(
    target_weights: pd.Series,
    regime: str,
    regime_confidence: float,
    as_of: pd.Timestamp,
    expected_turnover: float,
    rebalance_band_pct: float,
    config: ExecutionConfig | None = None,
    now: datetime | None = None,
) -> TargetWeightSignal:
    """
    Assemble a signal from math-engine output, validating the invariants the
    execution engine is entitled to assume.

    Validation happens at the boundary, not on the consumer: a malformed
    signal should never reach Redis in the first place. The Java side still
    re-checks, because a queue is an untrusted input, but the authoritative
    check is here where the failure is attributable.
    """
    config = config or ExecutionConfig()
    now = now or datetime.now(timezone.utc)

    weights = {
        t: (0.0 if abs(w) < config.weight_epsilon else round(float(w), 6))
        for t, w in target_weights.items()
    }

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"target weights sum to {total:.6f}, not 1.0")
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"negative weight in a long-only signal: {weights}")
    if not 0.0 <= regime_confidence <= 1.0:
        raise ValueError(f"regime_confidence {regime_confidence} outside [0, 1]")

    return TargetWeightSignal(
        schema_version=config.schema_version,
        signal_id=str(uuid.uuid4()),
        generated_at=now.isoformat(),
        valid_until=(now + timedelta(seconds=config.signal_ttl_seconds)).isoformat(),
        as_of=str(pd.Timestamp(as_of).date()),
        regime=regime,
        regime_confidence=round(float(regime_confidence), 4),
        target_weights=weights,
        expected_turnover=round(float(expected_turnover), 6),
        rebalance_band_pct=float(rebalance_band_pct),
    )
