# Architecture Decision Records

An ADR captures a design decision, the alternatives considered, and the tradeoffs accepted. It answers the question a future reader (or future-you) will ask: *why is this built this way and not the obvious other way?*

## When to write an ADR

Write one when **all** of these are true:

1. There is a real choice — at least two alternatives that a reasonable engineer might pick.
2. The decision will shape future work. Reversing it later would cost effort.
3. The reasoning is not obvious from reading the code. If a well-named function and a single comment answer the "why," a code comment is enough.

Examples that merit an ADR:
- Bundling pipeline outputs into a single `Dataset` vs. returning separate DataFrames (see 001).
- Testing philosophy: contracts vs. market calibration (see 002).
- Choice of covariance estimator family (sample, EWMA, Ledoit-Wolf shrinkage) and when each is applied.
- Choice of optimizer formulation (mean-variance, risk parity, max-diversification).
- Caching format and invalidation policy.

Examples that do **not** merit an ADR:
- Adding a test, fixing a bug, renaming a variable — `git log` is the right record.
- Config values like "forward-fill limit is 5 days" — a comment in `config.py` is enough.
- Style choices already captured in `CLAUDE.md` or `conftest.py` docstrings.

## How to write one

1. Pick the next number (`003-...`). Never renumber existing ADRs.
2. Title: imperative and specific. Not "Testing" — "Pipeline tests validate contracts, not market calibration."
3. Use the four-section template below. Keep each section tight — an ADR is a decision record, not a white paper.
4. Status starts as `Proposed`, becomes `Accepted` when the code lands. If a later ADR reverses this one, mark it `Superseded by ADR-NNN` and link.

### Template

```markdown
# ADR-NNN: <Imperative, specific title>

## Status

Proposed | Accepted | Superseded by ADR-NNN

## Context

What problem or forcing function led to this decision? What alternatives exist? What are the constraints? One to three paragraphs.

## Decision

What did we decide? State it directly. Include enough concreteness (file paths, function names, code snippets) that the reader can verify the code matches the ADR.

## Consequences

### Benefits

- What this buys us.

### Tradeoffs

- What this costs us, and why we accepted that cost.
```

## Current ADRs

- [ADR-001](001-dataset-dataclass.md) — Bundle pipeline outputs into a single `Dataset` dataclass.
- [ADR-002](002-test-contracts-not-calibration.md) — Pipeline tests validate contracts, not market calibration.
- [ADR-003](003-asset-universe-vxus.md) — Replace VT with VXUS in the asset universe.
- [ADR-004](004-covariance-estimator-design.md) — Implement covariance estimators from scratch, shrink toward a scaled identity.
- [ADR-005](005-optimizer-closed-form-and-slsqp.md) — Closed form where it exists, SLSQP where it doesn't; shrink the mean vector.
- [ADR-006](006-hmm-from-scratch-stats-tests-delegated.md) — Build the HMM from scratch; delegate the cointegration critical-value tables.
- [ADR-007](007-publish-target-weights-not-orders.md) — Publish target weights, not orders; make every signal idempotent and expiring.