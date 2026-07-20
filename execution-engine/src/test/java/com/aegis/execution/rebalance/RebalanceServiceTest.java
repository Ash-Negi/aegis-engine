package com.aegis.execution.rebalance;

import com.aegis.execution.order.Order;
import com.aegis.execution.order.OrderSide;
import com.aegis.execution.portfolio.PortfolioSnapshot;
import com.aegis.execution.portfolio.Position;
import com.aegis.execution.signal.TargetWeightSignal;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Tests for target weights → orders.
 *
 * <p>The two behaviours worth protecting are the no-trade band (without it,
 * every signal churns the account for pennies) and the idempotency key
 * (without it, a redelivered signal double-trades). Both are cheap to break
 * accidentally and expensive to notice in production.
 */
class RebalanceServiceTest {

    private static final Instant NOW = Instant.parse("2026-07-20T14:00:00Z");
    private static final Instant VALID_UNTIL = NOW.plusSeconds(86_400);

    private final RebalanceService service = new RebalanceService();

    private TargetWeightSignal signal(Map<String, String> weights, String bandPct) {
        Map<String, BigDecimal> parsed = weights.entrySet().stream()
                .collect(java.util.stream.Collectors.toMap(
                        Map.Entry::getKey, e -> new BigDecimal(e.getValue())));
        return new TargetWeightSignal(
                1, "sig-1", NOW.minusSeconds(60), VALID_UNTIL, "2026-04-27",
                "high_vol_meanrev", 0.98, parsed,
                BigDecimal.ZERO, new BigDecimal(bandPct));
    }

    private Map<String, BigDecimal> prices() {
        return Map.of(
                "QQQ", new BigDecimal("100"),
                "GLDM", new BigDecimal("50"),
                "VXUS", new BigDecimal("25"));
    }

    @Nested
    @DisplayName("order generation")
    class Generation {

        @Test
        @DisplayName("an all-cash account buys into every target")
        void fromCash() {
            PortfolioSnapshot snapshot =
                    new PortfolioSnapshot(new BigDecimal("10000"), List.of());
            TargetWeightSignal s = signal(
                    Map.of("QQQ", "0.5", "GLDM", "0.3", "VXUS", "0.2"), "5");

            List<Order> orders = service.planOrders(s, snapshot, prices(), NOW);

            assertThat(orders).hasSize(3);
            assertThat(orders).allMatch(o -> o.getSide() == OrderSide.BUY);
            // 50% of 10,000 = 5,000 at 100 = 50 shares.
            assertThat(orders).filteredOn(o -> o.getSymbol().equals("QQQ"))
                    .singleElement()
                    .extracting(Order::getTargetQuantity)
                    .isEqualTo(new BigDecimal("50"));
        }

        @Test
        @DisplayName("an overweight position is sold down")
        void sellsOverweight() {
            // 100 QQQ @ 100 = 10,000 of a 10,000 account: 100% in QQQ.
            PortfolioSnapshot snapshot = new PortfolioSnapshot(
                    BigDecimal.ZERO,
                    List.of(new Position("QQQ", new BigDecimal("100"), new BigDecimal("100"))));
            TargetWeightSignal s = signal(Map.of("QQQ", "1.0", "GLDM", "0.0"), "5");

            // Target is still 100% QQQ, so nothing should trade.
            assertThat(service.planOrders(s, snapshot, prices(), NOW)).isEmpty();

            TargetWeightSignal halved = signal(Map.of("QQQ", "0.5", "GLDM", "0.5"), "5");
            List<Order> orders = service.planOrders(halved, snapshot, prices(), NOW);

            assertThat(orders).filteredOn(o -> o.getSymbol().equals("QQQ"))
                    .singleElement()
                    .satisfies(o -> {
                        assertThat(o.getSide()).isEqualTo(OrderSide.SELL);
                        assertThat(o.getTargetQuantity()).isEqualByComparingTo("50");
                    });
        }

        @Test
        @DisplayName("a zero target liquidates the whole position")
        void zeroTargetLiquidates() {
            PortfolioSnapshot snapshot = new PortfolioSnapshot(
                    new BigDecimal("5000"),
                    List.of(new Position("QQQ", new BigDecimal("50"), new BigDecimal("100"))));
            TargetWeightSignal s = signal(Map.of("QQQ", "0.0", "GLDM", "1.0"), "5");

            List<Order> orders = service.planOrders(s, snapshot, prices(), NOW);

            assertThat(orders).filteredOn(o -> o.getSymbol().equals("QQQ"))
                    .singleElement()
                    .satisfies(o -> {
                        assertThat(o.getSide()).isEqualTo(OrderSide.SELL);
                        assertThat(o.getTargetQuantity()).isEqualByComparingTo("50");
                    });
        }

        @Test
        @DisplayName("quantities are whole shares, and a sub-share order is dropped")
        void wholeSharesOnly() {
            PortfolioSnapshot snapshot =
                    new PortfolioSnapshot(new BigDecimal("150"), List.of());
            TargetWeightSignal s = signal(Map.of("QQQ", "1.0"), "5");

            List<Order> orders = service.planOrders(s, snapshot, prices(), NOW);

            // 150 / 100 = 1.5 → 1 share, not 1.5.
            assertThat(orders).singleElement()
                    .extracting(Order::getTargetQuantity)
                    .isEqualTo(new BigDecimal("1"));
        }

        @Test
        void skipsSymbolsWithNoPrice() {
            PortfolioSnapshot snapshot =
                    new PortfolioSnapshot(new BigDecimal("10000"), List.of());
            TargetWeightSignal s = signal(Map.of("QQQ", "0.5", "MISSING", "0.5"), "5");

            List<Order> orders = service.planOrders(s, snapshot, prices(), NOW);

            assertThat(orders).singleElement()
                    .extracting(Order::getSymbol).isEqualTo("QQQ");
        }

        @Test
        void refusesToTradeAnEmptyAccount() {
            PortfolioSnapshot snapshot = new PortfolioSnapshot(BigDecimal.ZERO, List.of());
            assertThat(service.planOrders(signal(Map.of("QQQ", "1.0"), "5"),
                    snapshot, prices(), NOW)).isEmpty();
        }
    }

    @Nested
    @DisplayName("no-trade band")
    class Band {

        /** 10,000 account: 49 QQQ @ 100 = 4,900 (49%), rest cash. */
        private PortfolioSnapshot slightlyOff() {
            return new PortfolioSnapshot(
                    new BigDecimal("5100"),
                    List.of(new Position("QQQ", new BigDecimal("49"), new BigDecimal("100"))));
        }

        @Test
        @DisplayName("drift inside the band raises no order")
        void insideBandDoesNothing() {
            // 49% vs 50% target is 2% relative drift — inside a 5% band.
            TargetWeightSignal s = signal(Map.of("QQQ", "0.5", "GLDM", "0.5"), "5");
            List<Order> orders = service.planOrders(s, slightlyOff(), prices(), NOW);

            assertThat(orders).noneMatch(o -> o.getSymbol().equals("QQQ"));
        }

        @Test
        @DisplayName("the band is relative to target, not absolute")
        void bandIsRelative() {
            // 49% vs 50% is 2% relative. A 1% band is tighter than that, so
            // the same drift now trades — proving the band is not ±5pp.
            TargetWeightSignal tight = signal(Map.of("QQQ", "0.5", "GLDM", "0.5"), "1");
            assertThat(service.planOrders(tight, slightlyOff(), prices(), NOW))
                    .anyMatch(o -> o.getSymbol().equals("QQQ"));
        }

        @Test
        @DisplayName("a portfolio already on target trades nothing at all")
        void onTargetIsQuiet() {
            PortfolioSnapshot onTarget = new PortfolioSnapshot(
                    BigDecimal.ZERO,
                    List.of(
                            new Position("QQQ", new BigDecimal("50"), new BigDecimal("100")),
                            new Position("GLDM", new BigDecimal("100"), new BigDecimal("50"))));
            TargetWeightSignal s = signal(Map.of("QQQ", "0.5", "GLDM", "0.5"), "5");

            assertThat(service.planOrders(s, onTarget, prices(), NOW)).isEmpty();
        }
    }

    @Nested
    @DisplayName("idempotency")
    class Idempotency {

        @Test
        @DisplayName("the same signal produces the same client order ids")
        void deterministicKeys() {
            PortfolioSnapshot snapshot =
                    new PortfolioSnapshot(new BigDecimal("10000"), List.of());
            TargetWeightSignal s = signal(Map.of("QQQ", "1.0"), "5");

            String first = service.planOrders(s, snapshot, prices(), NOW)
                    .getFirst().getClientOrderId();
            String second = service.planOrders(s, snapshot, prices(), NOW.plusSeconds(30))
                    .getFirst().getClientOrderId();

            assertThat(first).isEqualTo(second).isEqualTo("sig-1:QQQ");
        }

        @Test
        void ordersCarrySignalLineage() {
            PortfolioSnapshot snapshot =
                    new PortfolioSnapshot(new BigDecimal("10000"), List.of());
            List<Order> orders = service.planOrders(
                    signal(Map.of("QQQ", "1.0"), "5"), snapshot, prices(), NOW);

            assertThat(orders).allMatch(o -> o.getSignalId().equals("sig-1"));
        }
    }
}
