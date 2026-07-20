package com.aegis.execution.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.math.BigDecimal;
import java.util.Map;

/**
 * Configuration for the execution engine, bound from {@code application.yml}.
 *
 * <p>Mirrors the "one source of truth" rule the Python side follows in
 * {@code config.py}: no tunable is hardcoded in a service.
 */
@ConfigurationProperties(prefix = "aegis")
public class AegisProperties {

    private final Signal signal = new Signal();
    private final Broker broker = new Broker();
    private final Account account = new Account();

    public Signal getSignal() {
        return signal;
    }

    public Broker getBroker() {
        return broker;
    }

    public Account getAccount() {
        return account;
    }

    public static class Signal {
        /** Channel the math engine publishes to. */
        private String channel = "aegis:signals";
        /** Durable key holding the last-known-good signal, read at startup. */
        private String latestKey = "aegis:signals:latest";
        /** Payload versions this build understands; anything else is refused. */
        private int schemaVersion = 1;
        /** Subscribe to live signals. Off for replay/reconciliation runs. */
        private boolean subscribeEnabled = true;

        public String getChannel() {
            return channel;
        }

        public void setChannel(String channel) {
            this.channel = channel;
        }

        public String getLatestKey() {
            return latestKey;
        }

        public void setLatestKey(String latestKey) {
            this.latestKey = latestKey;
        }

        public int getSchemaVersion() {
            return schemaVersion;
        }

        public void setSchemaVersion(int schemaVersion) {
            this.schemaVersion = schemaVersion;
        }

        public boolean isSubscribeEnabled() {
            return subscribeEnabled;
        }

        public void setSubscribeEnabled(boolean subscribeEnabled) {
            this.subscribeEnabled = subscribeEnabled;
        }
    }

    public static class Broker {
        /** mock | alpaca. Only "mock" is implemented. */
        private String mode = "mock";
        private double rejectProbability = 0.0;
        private double partialFillProbability = 0.4;
        private long seed = 42L;

        public String getMode() {
            return mode;
        }

        public void setMode(String mode) {
            this.mode = mode;
        }

        public double getRejectProbability() {
            return rejectProbability;
        }

        public void setRejectProbability(double rejectProbability) {
            this.rejectProbability = rejectProbability;
        }

        public double getPartialFillProbability() {
            return partialFillProbability;
        }

        public void setPartialFillProbability(double partialFillProbability) {
            this.partialFillProbability = partialFillProbability;
        }

        public long getSeed() {
            return seed;
        }

        public void setSeed(long seed) {
            this.seed = seed;
        }
    }

    /** Simulated account state, standing in for a real brokerage account. */
    public static class Account {
        private BigDecimal startingCash = new BigDecimal("100000");
        private Map<String, BigDecimal> prices = Map.of();

        public BigDecimal getStartingCash() {
            return startingCash;
        }

        public void setStartingCash(BigDecimal startingCash) {
            this.startingCash = startingCash;
        }

        public Map<String, BigDecimal> getPrices() {
            return prices;
        }

        public void setPrices(Map<String, BigDecimal> prices) {
            this.prices = prices;
        }
    }
}
