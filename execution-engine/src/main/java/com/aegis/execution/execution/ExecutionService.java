package com.aegis.execution.execution;

import com.aegis.execution.broker.BrokerAck;
import com.aegis.execution.broker.BrokerClient;
import com.aegis.execution.broker.BrokerException;
import com.aegis.execution.config.AegisProperties;
import com.aegis.execution.ledger.SignalRecord;
import com.aegis.execution.ledger.SignalRecordRepository;
import com.aegis.execution.order.Fill;
import com.aegis.execution.order.FillRepository;
import com.aegis.execution.order.Order;
import com.aegis.execution.order.OrderRepository;
import com.aegis.execution.order.OrderStateMachine;
import com.aegis.execution.portfolio.PortfolioSnapshot;
import com.aegis.execution.rebalance.RebalanceService;
import com.aegis.execution.risk.RiskGuard;
import com.aegis.execution.signal.InvalidSignalException;
import com.aegis.execution.signal.TargetWeightSignal;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.Clock;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Orchestrates one signal end to end: record it, validate it, plan orders,
 * submit them, and persist everything that happens.
 *
 * <p>The ordering of the first two steps is the important part. The signal is
 * written to the ledger <em>before</em> it is acted on, so a crash between
 * "received" and "traded" leaves evidence that the signal arrived. Recovery
 * then reconciles from the ledger instead of replaying blind.
 *
 * <p>Five guards decide whether a signal is acted on at all, and each has a
 * different failure mode behind it:
 *
 * <ul>
 *   <li><b>Already seen</b> — Redis pub/sub can redeliver, and the durable
 *       key is re-read on every restart. Without a dedupe on signal id, a
 *       restart loop would re-trade the same target repeatedly.</li>
 *   <li><b>Drawdown breach</b> — {@link RiskGuard} compares observed equity
 *       against its all-time high-water mark. Past a configured loss from
 *       peak, the account stops trading regardless of what the new signal
 *       says, the same way a real risk desk would rather sit in cash than
 *       trust a model mid-drawdown.</li>
 *   <li><b>Stale</b> — a signal older than its TTL describes a portfolio that
 *       made sense against last week's prices. Acting on it is worse than
 *       doing nothing.</li>
 *   <li><b>Concentration breach</b> — {@link RiskGuard} refuses a signal that
 *       asks for more than the configured ceiling in one symbol. This is a
 *       backstop independent of the optimizer's own constraints upstream, so
 *       a bad regime call or a bug in that layer can't bet the book on one
 *       asset.</li>
 *   <li><b>Invalid</b> — malformed weights become malformed orders.</li>
 * </ul>
 *
 * <p>A rejected signal is still recorded, with the reason. Silence is not an
 * acceptable audit trail.
 */
@Service
public class ExecutionService {

    private static final Logger log = LoggerFactory.getLogger(ExecutionService.class);

    private final RebalanceService rebalanceService;
    private final OrderStateMachine stateMachine;
    private final BrokerClient broker;
    private final OrderRepository orders;
    private final FillRepository fills;
    private final SignalRecordRepository signals;
    private final RiskGuard riskGuard;
    private final Clock clock;
    private final int expectedSchemaVersion;

    public ExecutionService(RebalanceService rebalanceService,
                            OrderStateMachine stateMachine,
                            BrokerClient broker,
                            OrderRepository orders,
                            FillRepository fills,
                            SignalRecordRepository signals,
                            RiskGuard riskGuard,
                            Clock clock,
                            AegisProperties properties) {
        this.rebalanceService = rebalanceService;
        this.stateMachine = stateMachine;
        this.broker = broker;
        this.orders = orders;
        this.fills = fills;
        this.signals = signals;
        this.riskGuard = riskGuard;
        this.clock = clock;
        this.expectedSchemaVersion = properties.getSignal().getSchemaVersion();
    }

    /**
     * Process one signal.
     *
     * @return the orders submitted; empty if the signal was refused or
     *         everything was already inside the no-trade band
     */
    @Transactional
    public List<Order> onSignal(TargetWeightSignal signal, String rawPayload,
                                PortfolioSnapshot snapshot, Map<String, BigDecimal> prices) {
        Instant now = clock.instant();

        if (signals.existsById(signal.signalId())) {
            log.info("signal {} already processed, ignoring redelivery", signal.signalId());
            return List.of();
        }

        Optional<String> drawdownBreach = riskGuard.checkDrawdown(snapshot, now);
        if (drawdownBreach.isPresent()) {
            reject(signal, rawPayload, now, drawdownBreach.get());
            return List.of();
        }

        if (signal.isStale(now)) {
            reject(signal, rawPayload, now, "signal expired at " + signal.validUntil());
            return List.of();
        }

        Optional<String> concentrationBreach = riskGuard.checkConcentration(signal);
        if (concentrationBreach.isPresent()) {
            reject(signal, rawPayload, now, concentrationBreach.get());
            return List.of();
        }

        try {
            signal.validate(expectedSchemaVersion);
        } catch (InvalidSignalException e) {
            reject(signal, rawPayload, now, e.getMessage());
            return List.of();
        }

        signals.save(record(signal, rawPayload, now, true, null));

        List<Order> planned = rebalanceService.planOrders(signal, snapshot, prices, now);
        planned.forEach(order -> submit(order, now));
        return planned;
    }

    /**
     * Send one order and record the outcome.
     *
     * <p>A {@link BrokerException} leaves the order in PENDING deliberately.
     * The submission may or may not have reached the broker, so the only safe
     * states are "unknown" (PENDING, to be reconciled by querying the broker
     * for the client order id) — marking it REJECTED would assert something
     * not known to be true, and resubmitting could double the position.
     */
    private void submit(Order order, Instant now) {
        if (orders.existsByClientOrderId(order.getClientOrderId())) {
            log.info("order {} already exists, skipping duplicate",
                    order.getClientOrderId());
            return;
        }
        orders.save(order);

        BrokerAck ack;
        try {
            ack = broker.submit(order);
        } catch (BrokerException e) {
            log.error("broker unreachable for {}; leaving PENDING for reconciliation",
                    order.getClientOrderId(), e);
            return;
        }

        if (ack.accepted()) {
            stateMachine.markSent(order, ack.brokerOrderId(), now);
        } else {
            stateMachine.markRejected(order, ack.rejectReason(), now);
        }
        orders.save(order);
    }

    /**
     * Apply a fill callback from the broker.
     *
     * <p>Dedupe is on the broker's fill id, not on quantity or timestamp:
     * brokers legitimately send two fills of the same size at the same price
     * milliseconds apart, and collapsing those would lose real quantity.
     */
    @Transactional
    public void onFill(Order order, Fill fill) {
        if (fills.existsByBrokerFillId(fill.getBrokerFillId())) {
            log.info("fill {} already applied, ignoring redelivery", fill.getBrokerFillId());
            return;
        }
        stateMachine.applyFill(order, fill, clock.instant());
        fills.save(fill);
        orders.save(order);
    }

    private void reject(TargetWeightSignal signal, String rawPayload,
                        Instant now, String reason) {
        log.warn("rejecting signal {}: {}", signal.signalId(), reason);
        signals.save(record(signal, rawPayload, now, false, reason));
    }

    private SignalRecord record(TargetWeightSignal signal, String rawPayload,
                                Instant now, boolean accepted, String reason) {
        return new SignalRecord(
                signal.signalId(), signal.schemaVersion(), signal.regime(),
                signal.regimeConfidence(), signal.asOf(), signal.generatedAt(),
                now, accepted, reason, rawPayload);
    }
}
