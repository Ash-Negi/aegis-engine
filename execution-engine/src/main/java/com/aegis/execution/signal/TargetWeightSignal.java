package com.aegis.execution.signal;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.Map;

/**
 * The wire format published by the Python math engine. Mirrors
 * {@code math-engine/publisher/contract.py} — if one changes, both change and
 * {@code schemaVersion} is bumped.
 *
 * <p>{@code @JsonIgnoreProperties(ignoreUnknown = true)} is deliberate. The
 * producer must be able to add a field without simultaneously deploying a new
 * consumer; unknown fields are additive and safe. Removing or retyping a
 * field is not, and that is what the version number is for.
 *
 * <p>The signal carries target <em>weights</em>, not orders. Translating a
 * desired portfolio into share quantities requires current positions and
 * account equity, which only this service knows.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record TargetWeightSignal(
        @JsonProperty("schema_version") int schemaVersion,
        @JsonProperty("signal_id") String signalId,
        @JsonProperty("generated_at") Instant generatedAt,
        @JsonProperty("valid_until") Instant validUntil,
        @JsonProperty("as_of") String asOf,
        String regime,
        @JsonProperty("regime_confidence") double regimeConfidence,
        @JsonProperty("target_weights") Map<String, BigDecimal> targetWeights,
        @JsonProperty("expected_turnover") BigDecimal expectedTurnover,
        @JsonProperty("rebalance_band_pct") BigDecimal rebalanceBandPct
) {

    /** Weights are allowed to miss 1.0 by this much before the signal is refused. */
    private static final BigDecimal WEIGHT_SUM_TOLERANCE = new BigDecimal("0.0001");

    public boolean isStale(Instant now) {
        return now.isAfter(validUntil);
    }

    /**
     * Re-check the invariants the producer already validated.
     *
     * <p>Duplicating the producer's validation is intentional: a message queue
     * is an untrusted input. The producer's check attributes the bug; this one
     * stops a malformed signal from becoming orders. A signal that fails here
     * is dropped, not corrected.
     */
    public void validate(int expectedSchemaVersion) {
        if (schemaVersion != expectedSchemaVersion) {
            throw new InvalidSignalException(
                    "unsupported schema version %d (expected %d)"
                            .formatted(schemaVersion, expectedSchemaVersion));
        }
        if (targetWeights == null || targetWeights.isEmpty()) {
            throw new InvalidSignalException("signal carries no target weights");
        }
        BigDecimal sum = BigDecimal.ZERO;
        for (Map.Entry<String, BigDecimal> entry : targetWeights.entrySet()) {
            if (entry.getValue().signum() < 0) {
                throw new InvalidSignalException(
                        "negative weight for " + entry.getKey() + ": " + entry.getValue());
            }
            sum = sum.add(entry.getValue());
        }
        if (sum.subtract(BigDecimal.ONE).abs().compareTo(WEIGHT_SUM_TOLERANCE) > 0) {
            throw new InvalidSignalException("target weights sum to " + sum + ", not 1.0");
        }
        if (regimeConfidence < 0.0 || regimeConfidence > 1.0) {
            throw new InvalidSignalException("regime confidence outside [0,1]: " + regimeConfidence);
        }
    }
}
