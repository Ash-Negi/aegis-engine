package com.aegis.execution.broker;

import com.aegis.execution.order.Fill;
import com.aegis.execution.order.Order;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * An in-memory broker that accepts orders and produces fills, standing in for
 * Alpaca until the real integration lands.
 *
 * <p>It is not a no-op stub. It reproduces the two broker behaviours the
 * execution engine has to be correct about, because a mock that only ever
 * fills orders completely and instantly would let genuine bugs through:
 *
 * <ul>
 *   <li><b>Partial fills.</b> A large order comes back in pieces, over several
 *       callbacks. Code that assumes one fill per order is wrong, and this is
 *       the only way to find that out before production.</li>
 *   <li><b>Rejections.</b> Some fraction of orders are refused, so the
 *       rejection path is exercised rather than merely written.</li>
 * </ul>
 *
 * <p>Seeded, so a demo run is reproducible.
 */
@Component
@ConditionalOnProperty(name = "aegis.broker.mode", havingValue = "mock", matchIfMissing = true)
public class MockBrokerClient implements BrokerClient {

    private static final Logger log = LoggerFactory.getLogger(MockBrokerClient.class);

    private final Map<String, BigDecimal> prices;
    private final double rejectProbability;
    private final double partialFillProbability;
    private final Random random;
    private final AtomicLong sequence = new AtomicLong();
    private final Map<String, Order> working = new ConcurrentHashMap<>();

    public MockBrokerClient() {
        this(Map.of(), 0.0, 0.4, 42L);
    }

    public MockBrokerClient(Map<String, BigDecimal> prices, double rejectProbability,
                            double partialFillProbability, long seed) {
        this.prices = new ConcurrentHashMap<>(prices);
        this.rejectProbability = rejectProbability;
        this.partialFillProbability = partialFillProbability;
        this.random = new Random(seed);
    }

    public void setPrice(String symbol, BigDecimal price) {
        prices.put(symbol, price);
    }

    @Override
    public BrokerAck submit(Order order) {
        if (random.nextDouble() < rejectProbability) {
            log.info("mock broker rejected {}", order.getClientOrderId());
            return BrokerAck.rejected("insufficient buying power (simulated)");
        }
        String brokerOrderId = "MOCK-" + sequence.incrementAndGet();
        working.put(brokerOrderId, order);
        return BrokerAck.accepted(brokerOrderId);
    }

    @Override
    public void cancel(String brokerOrderId) {
        working.remove(brokerOrderId);
    }

    /**
     * Produce the fills for an order.
     *
     * <p>Returns either one complete fill or a sequence of partials that sum
     * exactly to the target quantity. The sum is exact by construction — the
     * last slice is the remainder rather than another random draw — so a test
     * asserting that an order reaches FILLED is testing the state machine, not
     * the mock's arithmetic.
     */
    public List<Fill> generateFills(Order order, Instant now) {
        BigDecimal price = prices.getOrDefault(order.getSymbol(), BigDecimal.ONE);
        BigDecimal target = order.getTargetQuantity();
        List<Fill> fills = new ArrayList<>();

        if (target.compareTo(BigDecimal.ONE) <= 0
                || random.nextDouble() >= partialFillProbability) {
            fills.add(new Fill(order.getId(), nextFillId(), target, price, now));
            return fills;
        }

        BigDecimal first = target.multiply(new BigDecimal("0.6"))
                .setScale(0, RoundingMode.DOWN)
                .max(BigDecimal.ONE);
        BigDecimal second = target.subtract(first);

        fills.add(new Fill(order.getId(), nextFillId(), first, price, now));
        if (second.signum() > 0) {
            // A later fill prints slightly worse — enough to make the
            // average-price arithmetic in the state machine observable.
            BigDecimal slipped = price.multiply(new BigDecimal("1.0002"))
                    .setScale(8, RoundingMode.HALF_UP);
            fills.add(new Fill(order.getId(), nextFillId(), second, slipped,
                    now.plusMillis(250)));
        }
        return fills;
    }

    private String nextFillId() {
        return "MOCKFILL-" + sequence.incrementAndGet();
    }
}
