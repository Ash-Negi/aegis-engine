# Change Retrospectives

A retro captures what was learned from a meaningful change *after* the fact: what was expected, what actually happened, and what carries forward into future work. ADRs and retros pair: the ADR records *why we decided* to do something, the retro records *what we learned by doing it*.

## When to write a retro

Write one when **any** of these are true:

1. The change was non-trivial enough to warrant an ADR, and the outcome produced numbers, behaviors, or surprises worth preserving.
2. The change defused or surfaced a landmine — something the next person working on the codebase should know about even if it's no longer visible in the code.
3. A decision was validated or invalidated by the data after implementation. "We did X expecting Y; we got Z" is exactly the kind of context that rots out of code instantly.

Examples that merit a retro:
- An asset universe change after Week 1 review (see 001).
- Migrating an estimator (e.g. sample → Ledoit-Wolf) and observing how weights actually change.
- Tuning a parameter from one regime-validated value to another and recording the before/after.
- Discovering a real bug in a model that was producing plausible numbers.

Examples that do **not** merit a retro:
- Bug fixes — `git log` is the right record.
- Routine refactors with no quantitative outcome.
- Aesthetic changes (renaming, restructuring).

## How to write one

1. Pick the next number (`001-`, `002-`...). Never renumber.
2. Title: descriptive of the change, not the lesson. "VXUS swap" — not "Diversification matters."
3. Use the template below. Keep it tight — a retro is a debrief, not an essay.
4. If a retro invalidates an earlier decision, link the ADR being walked back and consider whether a new ADR is warranted.

### Template

```markdown
# Retro-NNN: <descriptive change name>

## Date

YYYY-MM-DD

## Change

What was changed, in one or two sentences. Link the ADR if there is one.

## Expected outcome

What did we predict before making the change, and why? Be specific — numbers if they were predicted, mechanisms if they were predicted.

## Actual outcome

What happened. Include the numbers. Compare to the predicted outcome explicitly.

## What we learned

The signals worth preserving. Surprises (good and bad), things confirmed, things invalidated. This is the core of the retro — everything else is scaffolding.

## What carries forward

Open landmines, follow-up work, things downstream modules will need to handle. The point of writing this section is so the future reader doesn't have to re-discover them.
```

## Current Retros

- [Retro-001](001-vxus-swap.md) — Replace VT with VXUS in the asset universe.
