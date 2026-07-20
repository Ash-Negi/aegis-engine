package com.aegis.execution.signal;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.assertThatCode;

/**
 * Tests for the Python↔Java wire contract.
 *
 * <p>The fixture below is a <em>real</em> payload, copied verbatim from a run
 * of {@code python -m publisher.report}. That matters: a hand-written fixture
 * only proves this record can parse what its author imagined, whereas the
 * genuine article catches the mismatches that actually happen — snake_case
 * field names, Python's {@code +00:00} offset spelling instead of {@code Z},
 * and a float that renders as {@code 0.0} rather than {@code 0}.
 *
 * <p>Whenever {@code contract.py} changes, this fixture should be regenerated
 * rather than edited by hand.
 */
class TargetWeightSignalTest {

    /** Verbatim output of `python -m publisher.report`, 2026-07-20. */
    private static final String PYTHON_PAYLOAD = """
            {"schema_version":1,"signal_id":"86f01d60-1d7e-400d-abc7-726187f6d406",\
            "generated_at":"2026-07-20T21:36:41.290554+00:00",\
            "valid_until":"2026-07-21T21:36:41.290554+00:00","as_of":"2026-04-27",\
            "regime":"high_vol_meanrev","regime_confidence":0.9825,\
            "target_weights":{"QQQ":0.191153,"GLDM":0.361865,"FBTC":0.0,"VXUS":0.446982},\
            "expected_turnover":0.0,"rebalance_band_pct":5.0}""";

    private final ObjectMapper mapper = new ObjectMapper().registerModule(new JavaTimeModule());

    @Nested
    @DisplayName("parsing what Python actually publishes")
    class Parsing {

        @Test
        void parsesRealPayload() throws Exception {
            TargetWeightSignal signal =
                    mapper.readValue(PYTHON_PAYLOAD, TargetWeightSignal.class);

            assertThat(signal.schemaVersion()).isEqualTo(1);
            assertThat(signal.signalId()).isEqualTo("86f01d60-1d7e-400d-abc7-726187f6d406");
            assertThat(signal.regime()).isEqualTo("high_vol_meanrev");
            assertThat(signal.regimeConfidence()).isEqualTo(0.9825);
            assertThat(signal.asOf()).isEqualTo("2026-04-27");
        }

        @Test
        @DisplayName("Python's +00:00 offset parses to the right instant")
        void parsesPythonTimestampFormat() throws Exception {
            TargetWeightSignal signal =
                    mapper.readValue(PYTHON_PAYLOAD, TargetWeightSignal.class);

            assertThat(signal.generatedAt())
                    .isEqualTo(Instant.parse("2026-07-20T21:36:41.290554Z"));
            assertThat(signal.validUntil())
                    .isEqualTo(Instant.parse("2026-07-21T21:36:41.290554Z"));
        }

        @Test
        @DisplayName("weights survive the trip as exact decimals")
        void weightsAreExact() throws Exception {
            Map<String, BigDecimal> weights =
                    mapper.readValue(PYTHON_PAYLOAD, TargetWeightSignal.class).targetWeights();

            assertThat(weights).hasSize(4);
            assertThat(weights.get("QQQ")).isEqualByComparingTo("0.191153");
            assertThat(weights.get("FBTC")).isEqualByComparingTo("0");
        }

        @Test
        @DisplayName("the real payload passes validation")
        void realPayloadIsValid() throws Exception {
            TargetWeightSignal signal =
                    mapper.readValue(PYTHON_PAYLOAD, TargetWeightSignal.class);
            assertThatCode(() -> signal.validate(1)).doesNotThrowAnyException();
        }

        @Test
        @DisplayName("an unknown field does not break the consumer")
        void toleratesNewFields() throws Exception {
            String withExtra = PYTHON_PAYLOAD.replaceFirst(
                    "\\{", "{\"some_future_field\":\"whatever\",");
            assertThatCode(() -> mapper.readValue(withExtra, TargetWeightSignal.class))
                    .doesNotThrowAnyException();
        }
    }

    @Nested
    @DisplayName("validation")
    class Validation {

        private TargetWeightSignal with(Map<String, BigDecimal> weights,
                                        int version, double confidence) {
            return new TargetWeightSignal(
                    version, "sig-1", Instant.EPOCH, Instant.EPOCH.plusSeconds(3600),
                    "2026-04-27", "crisis", confidence, weights,
                    BigDecimal.ZERO, new BigDecimal("5"));
        }

        @Test
        void rejectsUnknownSchemaVersion() {
            assertThatThrownBy(() -> with(Map.of("QQQ", BigDecimal.ONE), 99, 0.9).validate(1))
                    .isInstanceOf(InvalidSignalException.class)
                    .hasMessageContaining("schema version");
        }

        @Test
        void rejectsWeightsThatDoNotSumToOne() {
            assertThatThrownBy(() ->
                    with(Map.of("QQQ", new BigDecimal("0.5")), 1, 0.9).validate(1))
                    .isInstanceOf(InvalidSignalException.class)
                    .hasMessageContaining("sum to");
        }

        @Test
        void rejectsNegativeWeight() {
            assertThatThrownBy(() -> with(Map.of(
                    "QQQ", new BigDecimal("1.5"),
                    "GLDM", new BigDecimal("-0.5")), 1, 0.9).validate(1))
                    .isInstanceOf(InvalidSignalException.class)
                    .hasMessageContaining("negative weight");
        }

        @Test
        void rejectsEmptyWeights() {
            assertThatThrownBy(() -> with(Map.of(), 1, 0.9).validate(1))
                    .isInstanceOf(InvalidSignalException.class);
        }

        @Test
        void rejectsConfidenceOutsideUnitInterval() {
            assertThatThrownBy(() -> with(Map.of("QQQ", BigDecimal.ONE), 1, 1.4).validate(1))
                    .isInstanceOf(InvalidSignalException.class)
                    .hasMessageContaining("confidence");
        }

        @Test
        @DisplayName("rounding dust within tolerance is accepted")
        void toleratesRoundingDust() {
            // Six-decimal weights will not always sum to exactly 1.
            assertThatCode(() -> with(Map.of(
                    "QQQ", new BigDecimal("0.333333"),
                    "GLDM", new BigDecimal("0.333333"),
                    "VXUS", new BigDecimal("0.333334")), 1, 0.9).validate(1))
                    .doesNotThrowAnyException();
        }
    }

    @Nested
    @DisplayName("staleness")
    class Staleness {

        private TargetWeightSignal validUntil(Instant t) {
            return new TargetWeightSignal(
                    1, "sig-1", Instant.EPOCH, t, "2026-04-27", "crisis", 0.9,
                    Map.of("QQQ", BigDecimal.ONE), BigDecimal.ZERO, new BigDecimal("5"));
        }

        @Test
        void freshSignalIsNotStale() {
            Instant now = Instant.parse("2026-07-20T14:00:00Z");
            assertThat(validUntil(now.plusSeconds(60)).isStale(now)).isFalse();
        }

        @Test
        void expiredSignalIsStale() {
            Instant now = Instant.parse("2026-07-20T14:00:00Z");
            assertThat(validUntil(now.minusSeconds(1)).isStale(now)).isTrue();
        }
    }
}
