package com.aegis.execution.order;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.math.MathContext;
import java.time.Instant;

/**
 * The only component permitted to change an order's state.
 *
 * <p>Every mutator on {@link Order} is package-private, so all state changes
 * funnel through here and every one of them is checked against
 * {@link OrderState#canTransitionTo}. Centralising it this way means the
 * invariant "an order never makes an illegal transition" is enforced by the
 * type system rather than by everyone remembering to check.
 *
 * <p>The subtle case is {@link #applyFill}. Whether a fill produces
 * PARTIALLY_FILLED or FILLED is decided by comparing accumulated quantity
 * against the target — <em>not</em> by trusting a status field on the broker's
 * callback. Brokers disagree about when to call an order complete, and the
 * quantities are the thing that is actually reconcilable.
 */
@Component
public class OrderStateMachine {

    private static final Logger log = LoggerFactory.getLogger(OrderStateMachine.class);
    private static final MathContext MC = new MathContext(16);

    /** PENDING → SENT, recording the broker's id for the order. */
    public void markSent(Order order, String brokerOrderId, Instant now) {
        transition(order, OrderState.SENT, now);
        order.setBrokerOrderId(brokerOrderId);
    }

    /** → REJECTED, from either PENDING (local risk check) or SENT (broker). */
    public void markRejected(Order order, String reason, Instant now) {
        transition(order, OrderState.REJECTED, now);
        order.setRejectReason(reason);
    }

    /** → CANCELLED. Legal from any non-terminal state. */
    public void markCancelled(Order order, Instant now) {
        transition(order, OrderState.CANCELLED, now);
    }

    /**
     * Apply a fill, accumulating quantity and the quantity-weighted average
     * price, then move to PARTIALLY_FILLED or FILLED as the totals dictate.
     *
     * @throws IllegalArgumentException if the fill would overfill the order —
     *         a real condition worth failing loudly on, because it means
     *         either a duplicate callback slipped past dedupe or the broker
     *         executed more than was asked for.
     */
    public void applyFill(Order order, Fill fill, Instant now) {
        BigDecimal newFilled = order.getFilledQuantity().add(fill.getQuantity());
        if (newFilled.compareTo(order.getTargetQuantity()) > 0) {
            throw new IllegalArgumentException(
                    "fill %s would overfill order %s: %s + %s > %s".formatted(
                            fill.getBrokerFillId(), order.getClientOrderId(),
                            order.getFilledQuantity(), fill.getQuantity(),
                            order.getTargetQuantity()));
        }

        // Quantity-weighted average price, carried forward across fills.
        BigDecimal priorNotional = order.getAverageFillPrice() == null
                ? BigDecimal.ZERO
                : order.getAverageFillPrice().multiply(order.getFilledQuantity());
        BigDecimal newAverage = priorNotional.add(fill.notional()).divide(newFilled, MC);

        OrderState next = newFilled.compareTo(order.getTargetQuantity()) == 0
                ? OrderState.FILLED
                : OrderState.PARTIALLY_FILLED;

        transition(order, next, now);
        order.setFilledQuantity(newFilled);
        order.setAverageFillPrice(newAverage);
    }

    private void transition(Order order, OrderState next, Instant now) {
        OrderState current = order.getState();
        if (!current.canTransitionTo(next)) {
            log.error("rejected illegal transition for order {}: {} → {}",
                    order.getClientOrderId(), current, next);
            throw new IllegalTransitionException(order.getClientOrderId(), current, next);
        }
        order.setState(next);
        order.setUpdatedAt(now);
        log.debug("order {} {} → {}", order.getClientOrderId(), current, next);
    }
}
