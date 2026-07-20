package com.aegis.execution.broker;

/**
 * The broker was unreachable or returned something unusable.
 *
 * <p>Distinct from a rejection: a rejection is an answer (the order is dead),
 * while this is the absence of one (the order's fate is unknown and must be
 * resolved by querying the broker, never by resubmitting).
 */
public class BrokerException extends RuntimeException {

    public BrokerException(String message) {
        super(message);
    }

    public BrokerException(String message, Throwable cause) {
        super(message, cause);
    }
}
