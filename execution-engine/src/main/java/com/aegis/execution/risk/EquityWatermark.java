package com.aegis.execution.risk;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.math.BigDecimal;
import java.time.Instant;

/**
 * The highest total equity this account has ever been observed at.
 *
 * <p>Singleton row ({@code id = 1}), enforced by the migration's check
 * constraint. The drawdown circuit breaker in {@code ExecutionService} reads
 * this on every signal and ratchets it up when equity makes a new high, so a
 * restart sees the same peak an in-memory field would have lost.
 */
@Entity
@Table(name = "equity_watermark")
public class EquityWatermark {

    @Id
    private int id = 1;

    @Column(name = "peak_equity", nullable = false)
    private BigDecimal peakEquity;

    @Column(name = "observed_at", nullable = false)
    private Instant observedAt;

    protected EquityWatermark() {
        // for JPA
    }

    public EquityWatermark(BigDecimal peakEquity, Instant observedAt) {
        this.peakEquity = peakEquity;
        this.observedAt = observedAt;
    }

    public BigDecimal getPeakEquity() {
        return peakEquity;
    }

    public Instant getObservedAt() {
        return observedAt;
    }

    void raiseTo(BigDecimal newPeak, Instant now) {
        this.peakEquity = newPeak;
        this.observedAt = now;
    }
}
