package com.aegis.execution.broker;

import com.aegis.execution.order.Order;

/**
 * The broker boundary.
 *
 * <p>Deliberately narrow: submit, and ask what happened. Everything the
 * execution engine knows about order state it learns through this interface,
 * so swapping {@link MockBrokerClient} for an Alpaca implementation changes
 * nothing above it.
 *
 * <p>{@code submit} takes the order's {@code clientOrderId} to the broker. That
 * is what makes a submission timeout recoverable: if the response is lost, the
 * engine can ask the broker about that id rather than guess whether the order
 * exists — resubmitting blind is how you end up with two positions.
 */
public interface BrokerClient {

    /**
     * Send an order to the broker.
     *
     * @return the broker's acknowledgement, carrying its own order id
     * @throws BrokerException if the broker refused or was unreachable
     */
    BrokerAck submit(Order order);

    /** Ask the broker to cancel a working order. */
    void cancel(String brokerOrderId);
}
