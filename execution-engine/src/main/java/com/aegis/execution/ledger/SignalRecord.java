package com.aegis.execution.ledger;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.time.Instant;

/**
 * Every signal the engine received, stored before anything is done with it.
 *
 * <p>This table is the idempotency boundary and the audit root. Recording the
 * signal <em>first</em> means that if the engine crashes mid-rebalance, the
 * restart can see that this signal was already seen and reconcile rather than
 * re-trade it. The raw payload is kept verbatim so a later dispute about what
 * the engine was told is answerable from the ledger rather than from logs.
 *
 * <p>Signals that fail validation are recorded too, with {@code accepted =
 * false} and the reason. A dropped signal that leaves no trace is
 * indistinguishable from one that never arrived.
 */
@Entity
@Table(name = "signals")
public class SignalRecord {

    @Id
    @Column(name = "signal_id", length = 64)
    private String signalId;

    @Column(name = "schema_version", nullable = false)
    private int schemaVersion;

    @Column(nullable = false, length = 32)
    private String regime;

    @Column(name = "regime_confidence", nullable = false)
    private double regimeConfidence;

    @Column(name = "as_of", nullable = false, length = 16)
    private String asOf;

    @Column(name = "generated_at", nullable = false)
    private Instant generatedAt;

    @Column(name = "received_at", nullable = false)
    private Instant receivedAt;

    @Column(nullable = false)
    private boolean accepted;

    @Column(name = "reject_reason", length = 512)
    private String rejectReason;

    /** The payload exactly as it arrived. */
    @Column(name = "raw_payload", nullable = false, columnDefinition = "text")
    private String rawPayload;

    protected SignalRecord() {
        // for JPA
    }

    public SignalRecord(String signalId, int schemaVersion, String regime,
                        double regimeConfidence, String asOf, Instant generatedAt,
                        Instant receivedAt, boolean accepted, String rejectReason,
                        String rawPayload) {
        this.signalId = signalId;
        this.schemaVersion = schemaVersion;
        this.regime = regime;
        this.regimeConfidence = regimeConfidence;
        this.asOf = asOf;
        this.generatedAt = generatedAt;
        this.receivedAt = receivedAt;
        this.accepted = accepted;
        this.rejectReason = rejectReason;
        this.rawPayload = rawPayload;
    }

    public String getSignalId() {
        return signalId;
    }

    public int getSchemaVersion() {
        return schemaVersion;
    }

    public String getRegime() {
        return regime;
    }

    public double getRegimeConfidence() {
        return regimeConfidence;
    }

    public String getAsOf() {
        return asOf;
    }

    public Instant getGeneratedAt() {
        return generatedAt;
    }

    public Instant getReceivedAt() {
        return receivedAt;
    }

    public boolean isAccepted() {
        return accepted;
    }

    public String getRejectReason() {
        return rejectReason;
    }

    public String getRawPayload() {
        return rawPayload;
    }
}
