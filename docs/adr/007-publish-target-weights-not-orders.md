# ADR-007: Publish target weights, not orders; make every signal idempotent and expiring

## Status

Accepted (2026-07-20)

## Context

Phase 3 splits the system across two processes and a network. That boundary
has to be designed, not merely crossed, and three questions had real
alternatives.

**1. What crosses the wire?** The math engine could send imperative commands
("buy 43 shares of QQQ") or desired state ("the portfolio should be 19% QQQ").
Commands are more direct — the execution engine becomes a thin relay. State
requires the execution engine to do arithmetic the math engine could have done
itself.

**2. What are the delivery guarantees?** Redis pub/sub is at-most-once and has
no replay: a subscriber that is restarting simply misses whatever was
published. Redis Streams offer consumer groups and replay at the cost of more
moving parts. A queue like RabbitMQ or Kafka offers more still.

**3. How does the consumer avoid acting twice, or too late?** Any transport
that can redeliver can cause double-trading, and any transport that can be
slow can deliver a signal describing a portfolio that made sense last week.

## Decision

**The wire carries target weights — desired state, not commands.**

`math-engine/publisher/contract.py` defines a `TargetWeightSignal` mirrored by
the Java record of the same name. It carries weights, the detected regime and
its confidence, the implied turnover, and the rebalance band. It carries no
share counts, no prices, and no order instructions.

The execution engine turns weights into orders in `RebalanceService`, because
that translation needs three things the math engine does not have: current
positions, current prices, and account equity. Asking the math engine to know
them would mean giving it a broker connection and a position store — which is
the execution engine, rebuilt in the wrong process.

The deeper reason is that **state is idempotent and commands are not**.
Re-applying "target 19% QQQ" is a no-op once the portfolio is already there.
Re-applying "buy 43 shares" doubles the position. Every recovery mechanism
below depends on that property.

**Redis pub/sub, plus a durable key.** Every publish does both:

```
SET     aegis:signals:latest   ← written first, so a crash between the two
PUBLISH aegis:signals              still leaves the signal recoverable
```

The execution engine subscribes for live updates and reads the key on startup
(`SignalSubscriber.recoverOnStartup`), so a redeployed engine learns the
current target immediately rather than trading a stale portfolio until the
next publish. Replaying the key on every boot is safe precisely because the
payload is idempotent target state.

**Every signal is idempotent and expiring.** `signal_id` is a UUID the
consumer dedupes on; `valid_until` is a hard expiry. `ExecutionService`
refuses a signal that is already in the ledger, or stale, or invalid — and
records all three cases with the reason, because a silently dropped signal is
indistinguishable from one that never arrived.

Order-level idempotency is separate and stronger: `clientOrderId` is
`signalId + ":" + symbol`, so a redelivered signal produces the same key and
the `UNIQUE` constraint on `orders.client_order_id` refuses the duplicate.
The application-level check in front of it is an optimisation; the database
constraint is the guarantee.

## Consequences

**Benefits**

- The two services can be reasoned about independently. The math engine has no
  concept of an order; the execution engine has no concept of covariance.
- Recovery is trivial. Restart the execution engine and it re-reads the key;
  the dedupe and the band mean nothing is double-traded and nothing churns.
- The no-trade band is applied with the same rule the Phase 1 backtest used,
  so the cost drag that backtest measured is a real prediction rather than a
  fiction.
- Testing the seam is possible without running both services: the Java
  contract test parses a payload copied verbatim from a real Python run.

**Tradeoffs**

- Rebalance arithmetic is duplicated in spirit between the Python backtest and
  the Java engine. Accepted because they serve different masters (one
  simulates, one executes), but they must not diverge — the band being
  *relative to target* is the specific thing tested on both sides.
- Pub/sub gives no delivery receipt. The publisher logs subscriber count but
  cannot know a signal was acted on; the ledger is the only place that answers
  that. Accepted for now, and the reason `signals` is written before anything
  is traded.
- Validation is duplicated: the producer validates before publishing, and the
  consumer re-validates on receipt. Deliberate. The producer's check
  attributes the bug to the code that caused it; the consumer's check treats
  the queue as the untrusted input it is.

## Alternatives considered

**Publish orders.** Rejected: it puts position tracking and price data in the
math engine, and makes every redelivery a double-trade risk that no amount of
downstream care fully removes.

**Redis Streams instead of pub/sub.** Streams give consumer groups,
acknowledgements, and replay — genuinely better delivery semantics. Rejected
for now because idempotent target state plus a durable key recovers the same
property with one Redis data type and no consumer-group bookkeeping. If the
engine ever needs multiple competing consumers or an audit of what was
delivered (as opposed to what was acted on), this is the decision to revisit.

**A message broker (Kafka/RabbitMQ).** Rejected as disproportionate. The
system publishes one signal per rebalance decision — on the order of one a
day, not thousands a second — and Redis is already required for other state.

**Trusting the broker's status field to decide FILLED.** Rejected in
`OrderStateMachine`: completion is determined by comparing accumulated fill
quantity against the target, because brokers disagree about when to call an
order complete and the quantities are the part that reconciles against a
statement.

## Note on scope

The Alpaca integration is deliberately stubbed. `BrokerClient` is the
interface; `MockBrokerClient` is the implementation, and it reproduces the two
behaviours the engine must be correct about — partial fills and rejections —
rather than always filling instantly, which would let real bugs through. What
is *not* built is credential handling, rate limiting, and reconciliation
against a live account. Those are Phase 5 concerns and are listed as such.
