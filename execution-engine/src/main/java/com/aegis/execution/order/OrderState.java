package com.aegis.execution.order;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

/**
 * The lifecycle of an order, with the legal transitions declared rather than
 * scattered across the code that performs them.
 *
 * <p>Making the transition table explicit is the point. An order's state is
 * the one piece of information reconciliation depends on, and the failure mode
 * that matters is not a rejected order — it is an order that quietly moves
 * from FILLED back to SENT because two brokers' callbacks arrived out of
 * order. That is unrecoverable after the fact: you cannot tell from the
 * database whether you own the shares. So illegal transitions are rejected at
 * the point of attempt, where the offending event is still in hand.
 *
 * <pre>
 *   PENDING ──→ SENT ──→ PARTIALLY_FILLED ──→ FILLED
 *      │         │              │
 *      │         ├──────────────┴──→ CANCELLED
 *      │         └──→ REJECTED
 *      └──→ REJECTED        (never left the building — failed local risk checks)
 * </pre>
 *
 * <p>FILLED, REJECTED and CANCELLED are terminal. PARTIALLY_FILLED is
 * re-entrant: a partial fill followed by another partial fill is normal and
 * must not be treated as a duplicate.
 */
public enum OrderState {

    /** Created locally, not yet sent to the broker. */
    PENDING,

    /** Accepted by the broker, working in the market. */
    SENT,

    /** Some quantity has filled; more is still working. */
    PARTIALLY_FILLED,

    /** Fully filled. Terminal. */
    FILLED,

    /** Refused, either by local risk checks or by the broker. Terminal. */
    REJECTED,

    /** Withdrawn before completing. Terminal. */
    CANCELLED;

    private static final Map<OrderState, Set<OrderState>> LEGAL_TRANSITIONS = Map.of(
            PENDING,          EnumSet.of(SENT, REJECTED, CANCELLED),
            SENT,             EnumSet.of(PARTIALLY_FILLED, FILLED, REJECTED, CANCELLED),
            PARTIALLY_FILLED, EnumSet.of(PARTIALLY_FILLED, FILLED, CANCELLED),
            FILLED,           Collections.emptySet(),
            REJECTED,         Collections.emptySet(),
            CANCELLED,        Collections.emptySet()
    );

    public boolean canTransitionTo(OrderState next) {
        return LEGAL_TRANSITIONS.get(this).contains(next);
    }

    public boolean isTerminal() {
        return LEGAL_TRANSITIONS.get(this).isEmpty();
    }
}
