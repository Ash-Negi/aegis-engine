package com.aegis.execution.broker;

/** The broker's response to a submission. */
public record BrokerAck(String brokerOrderId, boolean accepted, String rejectReason) {

    public static BrokerAck accepted(String brokerOrderId) {
        return new BrokerAck(brokerOrderId, true, null);
    }

    public static BrokerAck rejected(String reason) {
        return new BrokerAck(null, false, reason);
    }
}
