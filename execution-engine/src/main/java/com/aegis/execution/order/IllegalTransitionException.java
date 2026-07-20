package com.aegis.execution.order;

/**
 * Thrown when an event would move an order into a state it cannot legally
 * reach. Carries both states so the log line identifies the offending event
 * without a debugger.
 */
public class IllegalTransitionException extends RuntimeException {

    private final OrderState from;
    private final OrderState to;

    public IllegalTransitionException(String clientOrderId, OrderState from, OrderState to) {
        super("order %s cannot transition %s → %s".formatted(clientOrderId, from, to));
        this.from = from;
        this.to = to;
    }

    public OrderState getFrom() {
        return from;
    }

    public OrderState getTo() {
        return to;
    }
}
