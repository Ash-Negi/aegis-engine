package com.aegis.execution.execution;

import com.aegis.execution.broker.MockBrokerClient;
import com.aegis.execution.config.AegisProperties;
import com.aegis.execution.ledger.SignalRecord;
import com.aegis.execution.ledger.SignalRecordRepository;
import com.aegis.execution.order.Fill;
import com.aegis.execution.order.FillRepository;
import com.aegis.execution.order.Order;
import com.aegis.execution.order.OrderRepository;
import com.aegis.execution.order.OrderState;
import com.aegis.execution.order.OrderStateMachine;
import com.aegis.execution.portfolio.PortfolioSnapshot;
import com.aegis.execution.rebalance.RebalanceService;
import com.aegis.execution.risk.RiskGuard;
import com.aegis.execution.signal.TargetWeightSignal;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * End-to-end tests for signal intake, against the real Flyway-migrated schema.
 *
 * <p>These run on H2 in PostgreSQL mode rather than a mocked repository layer,
 * because the things being tested here — the unique constraint on
 * {@code client_order_id}, the JPA mappings matching the migration, the
 * ordering of ledger writes — are precisely the things a mock cannot verify.
 *
 * <p>The clock is injected and fixed, so staleness is tested by constructing a
 * signal that expired rather than by sleeping.
 */
@SpringBootTest
@ActiveProfiles("test")
@Transactional
class ExecutionServiceTest {

    private static final Instant NOW = Instant.parse("2026-07-20T14:00:00Z");

    @Autowired
    private RebalanceService rebalanceService;
    @Autowired
    private OrderStateMachine stateMachine;
    @Autowired
    private OrderRepository orders;
    @Autowired
    private FillRepository fills;
    @Autowired
    private SignalRecordRepository signals;
    @Autowired
    private RiskGuard riskGuard;
    @Autowired
    private AegisProperties properties;

    private ExecutionService service;
    private MockBrokerClient broker;

    @BeforeEach
    void setUp() {
        broker = new MockBrokerClient(
                Map.of("QQQ", new BigDecimal("100"), "GLDM", new BigDecimal("50")),
                0.0, 0.0, 42L);
        service = new ExecutionService(
                rebalanceService, stateMachine, broker, orders, fills, signals, riskGuard,
                Clock.fixed(NOW, ZoneOffset.UTC), properties);
    }

    private TargetWeightSignal signal(String id, Instant validUntil) {
        return new TargetWeightSignal(
                1, id, NOW.minusSeconds(60), validUntil, "2026-04-27",
                "high_vol_meanrev", 0.98,
                Map.of("QQQ", new BigDecimal("0.5"), "GLDM", new BigDecimal("0.5")),
                BigDecimal.ZERO, new BigDecimal("5"));
    }

    private PortfolioSnapshot cashAccount() {
        return new PortfolioSnapshot(new BigDecimal("10000"), List.of());
    }

    private Map<String, BigDecimal> prices() {
        return Map.of("QQQ", new BigDecimal("100"), "GLDM", new BigDecimal("50"));
    }

    @Test
    @DisplayName("a valid signal is recorded and produces sent orders")
    void happyPath() {
        List<Order> planned = service.onSignal(
                signal("sig-1", NOW.plusSeconds(3600)), "{}", cashAccount(), prices());

        assertThat(planned).hasSize(2);
        assertThat(orders.findBySignalId("sig-1"))
                .hasSize(2)
                .allMatch(o -> o.getState() == OrderState.SENT)
                .allMatch(o -> o.getBrokerOrderId() != null);

        SignalRecord record = signals.findById("sig-1").orElseThrow();
        assertThat(record.isAccepted()).isTrue();
        assertThat(record.getRegime()).isEqualTo("high_vol_meanrev");
    }

    @Test
    @DisplayName("a redelivered signal is ignored, not re-traded")
    void redeliveryIsIdempotent() {
        TargetWeightSignal s = signal("sig-dup", NOW.plusSeconds(3600));

        assertThat(service.onSignal(s, "{}", cashAccount(), prices())).hasSize(2);
        assertThat(service.onSignal(s, "{}", cashAccount(), prices())).isEmpty();

        assertThat(orders.findBySignalId("sig-dup")).hasSize(2);
    }

    @Test
    @DisplayName("a stale signal is refused but still recorded, with the reason")
    void staleSignalRefused() {
        service.onSignal(signal("sig-old", NOW.minusSeconds(1)), "{}",
                cashAccount(), prices());

        assertThat(orders.findBySignalId("sig-old")).isEmpty();
        SignalRecord record = signals.findById("sig-old").orElseThrow();
        assertThat(record.isAccepted()).isFalse();
        assertThat(record.getRejectReason()).contains("expired");
    }

    @Test
    @DisplayName("an unknown schema version is refused rather than guessed at")
    void wrongSchemaVersionRefused() {
        TargetWeightSignal future = new TargetWeightSignal(
                99, "sig-v99", NOW.minusSeconds(60), NOW.plusSeconds(3600), "2026-04-27",
                "crisis", 0.9, Map.of("QQQ", new BigDecimal("0.5"), "GLDM", new BigDecimal("0.5")),
                BigDecimal.ZERO, new BigDecimal("5"));

        service.onSignal(future, "{}", cashAccount(), prices());

        assertThat(orders.findBySignalId("sig-v99")).isEmpty();
        assertThat(signals.findById("sig-v99").orElseThrow().getRejectReason())
                .contains("schema version");
    }

    @Test
    @DisplayName("weights that do not sum to 1 never become orders")
    void malformedWeightsRefused() {
        TargetWeightSignal bad = new TargetWeightSignal(
                1, "sig-bad", NOW.minusSeconds(60), NOW.plusSeconds(3600), "2026-04-27",
                "crisis", 0.9,
                Map.of("QQQ", new BigDecimal("0.3"), "GLDM", new BigDecimal("0.3")),
                BigDecimal.ZERO, new BigDecimal("5"));

        service.onSignal(bad, "{}", cashAccount(), prices());

        assertThat(orders.findBySignalId("sig-bad")).isEmpty();
        assertThat(signals.findById("sig-bad").orElseThrow().getRejectReason())
                .contains("sum to");
    }

    @Test
    @DisplayName("a signal over the per-position concentration ceiling never becomes orders")
    void concentrationBreachRefused() {
        TargetWeightSignal overConcentrated = new TargetWeightSignal(
                1, "sig-conc", NOW.minusSeconds(60), NOW.plusSeconds(3600), "2026-04-27",
                "crisis", 0.9,
                Map.of("QQQ", new BigDecimal("0.75"), "GLDM", new BigDecimal("0.25")),
                BigDecimal.ZERO, new BigDecimal("5"));

        service.onSignal(overConcentrated, "{}", cashAccount(), prices());

        assertThat(orders.findBySignalId("sig-conc")).isEmpty();
        assertThat(signals.findById("sig-conc").orElseThrow().getRejectReason())
                .contains("concentration breach")
                .contains("QQQ");
    }

    @Test
    @DisplayName("trading halts once equity has fallen too far below its peak")
    void drawdownBreachRefused() {
        service.onSignal(signal("sig-peak", NOW.plusSeconds(3600)),
                "{}", new PortfolioSnapshot(new BigDecimal("10000"), List.of()), prices());

        TargetWeightSignal afterDrop = signal("sig-drop", NOW.plusSeconds(3600));
        service.onSignal(afterDrop, "{}",
                new PortfolioSnapshot(new BigDecimal("7000"), List.of()), prices());

        assertThat(orders.findBySignalId("sig-drop")).isEmpty();
        assertThat(signals.findById("sig-drop").orElseThrow().getRejectReason())
                .contains("drawdown breach");
    }

    @Test
    @DisplayName("a drop that stays inside the drawdown limit still trades")
    void drawdownWithinLimitAllowed() {
        service.onSignal(signal("sig-peak2", NOW.plusSeconds(3600)),
                "{}", new PortfolioSnapshot(new BigDecimal("10000"), List.of()), prices());

        List<Order> planned = service.onSignal(signal("sig-smalldrop", NOW.plusSeconds(3600)),
                "{}", new PortfolioSnapshot(new BigDecimal("9000"), List.of()), prices());

        assertThat(planned).isNotEmpty();
        assertThat(signals.findById("sig-smalldrop").orElseThrow().isAccepted()).isTrue();
    }

    @Test
    @DisplayName("fills accumulate through to FILLED and persist")
    void fillsDriveOrderToCompletion() {
        service.onSignal(signal("sig-fill", NOW.plusSeconds(3600)), "{}",
                cashAccount(), prices());
        Order order = orders.findByClientOrderId("sig-fill:QQQ").orElseThrow();

        for (Fill fill : broker.generateFills(order, NOW)) {
            service.onFill(order, fill);
        }

        Order reloaded = orders.findByClientOrderId("sig-fill:QQQ").orElseThrow();
        assertThat(reloaded.getState()).isEqualTo(OrderState.FILLED);
        assertThat(reloaded.getFilledQuantity())
                .isEqualByComparingTo(reloaded.getTargetQuantity());
        assertThat(fills.findByOrderId(order.getId())).isNotEmpty();
    }

    @Test
    @DisplayName("a redelivered fill callback does not double-count")
    void duplicateFillIgnored() {
        service.onSignal(signal("sig-dupfill", NOW.plusSeconds(3600)), "{}",
                cashAccount(), prices());
        Order order = orders.findByClientOrderId("sig-dupfill:QQQ").orElseThrow();

        Fill fill = new Fill(order.getId(), "BROKER-FILL-1", new BigDecimal("10"),
                new BigDecimal("100"), NOW);
        service.onFill(order, fill);
        service.onFill(order, fill);

        assertThat(orders.findByClientOrderId("sig-dupfill:QQQ").orElseThrow()
                .getFilledQuantity()).isEqualByComparingTo("10");
        assertThat(fills.findByOrderId(order.getId())).hasSize(1);
    }

    @Test
    @DisplayName("a broker rejection is recorded on the order, not swallowed")
    void brokerRejectionRecorded() {
        ExecutionService rejecting = new ExecutionService(
                rebalanceService, stateMachine,
                new MockBrokerClient(Map.of(), 1.0, 0.0, 1L),
                orders, fills, signals, riskGuard,
                Clock.fixed(NOW, ZoneOffset.UTC), properties);

        rejecting.onSignal(signal("sig-rej", NOW.plusSeconds(3600)), "{}",
                cashAccount(), prices());

        assertThat(orders.findBySignalId("sig-rej"))
                .isNotEmpty()
                .allMatch(o -> o.getState() == OrderState.REJECTED)
                .allMatch(o -> o.getRejectReason() != null);
    }
}
