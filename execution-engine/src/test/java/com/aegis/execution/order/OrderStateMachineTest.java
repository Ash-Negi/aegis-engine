package com.aegis.execution.order;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Contract tests for the order lifecycle.
 *
 * <p>The tests that matter here are the negative ones. An order that fills
 * correctly is the easy path; the failure this component exists to prevent is
 * an order moving backwards out of a terminal state because two callbacks
 * arrived out of order, leaving the ledger unable to say whether the shares
 * are owned.
 */
class OrderStateMachineTest {

    private static final Instant NOW = Instant.parse("2026-07-20T14:00:00Z");

    private final OrderStateMachine machine = new OrderStateMachine();

    private Order order(String qty) {
        return new Order("sig-1:QQQ", "sig-1", "QQQ", OrderSide.BUY,
                new BigDecimal(qty), NOW);
    }

    private Fill fill(Order order, String id, String qty, String price) {
        return new Fill(order.getId(), id, new BigDecimal(qty), new BigDecimal(price), NOW);
    }

    @Nested
    @DisplayName("legal transitions")
    class Legal {

        @Test
        void newOrderStartsPending() {
            assertThat(order("100").getState()).isEqualTo(OrderState.PENDING);
        }

        @Test
        void pendingToSentRecordsBrokerId() {
            Order o = order("100");
            machine.markSent(o, "MOCK-1", NOW);
            assertThat(o.getState()).isEqualTo(OrderState.SENT);
            assertThat(o.getBrokerOrderId()).isEqualTo("MOCK-1");
        }

        @Test
        void partialFillLeavesOrderWorking() {
            Order o = order("100");
            machine.markSent(o, "MOCK-1", NOW);
            machine.applyFill(o, fill(o, "F1", "60", "485.00"), NOW);

            assertThat(o.getState()).isEqualTo(OrderState.PARTIALLY_FILLED);
            assertThat(o.getFilledQuantity()).isEqualByComparingTo("60");
            assertThat(o.remainingQuantity()).isEqualByComparingTo("40");
        }

        @Test
        void successivePartialsAreAllowed() {
            Order o = order("100");
            machine.markSent(o, "MOCK-1", NOW);
            machine.applyFill(o, fill(o, "F1", "30", "485.00"), NOW);
            machine.applyFill(o, fill(o, "F2", "30", "485.00"), NOW);

            assertThat(o.getState()).isEqualTo(OrderState.PARTIALLY_FILLED);
            assertThat(o.getFilledQuantity()).isEqualByComparingTo("60");
        }

        @Test
        void fillingTheRemainderCompletesTheOrder() {
            Order o = order("100");
            machine.markSent(o, "MOCK-1", NOW);
            machine.applyFill(o, fill(o, "F1", "60", "485.00"), NOW);
            machine.applyFill(o, fill(o, "F2", "40", "486.00"), NOW);

            assertThat(o.getState()).isEqualTo(OrderState.FILLED);
            assertThat(o.remainingQuantity()).isEqualByComparingTo("0");
        }

        @Test
        @DisplayName("average price is quantity-weighted, not a simple mean")
        void averagePriceIsQuantityWeighted() {
            Order o = order("100");
            machine.markSent(o, "MOCK-1", NOW);
            machine.applyFill(o, fill(o, "F1", "90", "100.00"), NOW);
            machine.applyFill(o, fill(o, "F2", "10", "200.00"), NOW);

            // (90×100 + 10×200)/100 = 110, not the unweighted mean of 150.
            assertThat(o.getAverageFillPrice()).isEqualByComparingTo("110");
        }

        @Test
        void rejectionFromPendingRecordsReason() {
            Order o = order("100");
            machine.markRejected(o, "insufficient buying power", NOW);
            assertThat(o.getState()).isEqualTo(OrderState.REJECTED);
            assertThat(o.getRejectReason()).isEqualTo("insufficient buying power");
        }

        @Test
        void partiallyFilledOrderCanBeCancelled() {
            Order o = order("100");
            machine.markSent(o, "MOCK-1", NOW);
            machine.applyFill(o, fill(o, "F1", "60", "485.00"), NOW);
            machine.markCancelled(o, NOW);
            assertThat(o.getState()).isEqualTo(OrderState.CANCELLED);
        }
    }

    @Nested
    @DisplayName("illegal transitions are refused")
    class Illegal {

        @Test
        @DisplayName("a filled order cannot go back to working")
        void filledCannotRegress() {
            Order o = order("100");
            machine.markSent(o, "MOCK-1", NOW);
            machine.applyFill(o, fill(o, "F1", "100", "485.00"), NOW);

            assertThatThrownBy(() -> machine.markCancelled(o, NOW))
                    .isInstanceOf(IllegalTransitionException.class);
            assertThat(o.getState()).isEqualTo(OrderState.FILLED);
        }

        @Test
        void rejectedIsTerminal() {
            Order o = order("100");
            machine.markRejected(o, "nope", NOW);
            assertThatThrownBy(() -> machine.markSent(o, "MOCK-2", NOW))
                    .isInstanceOf(IllegalTransitionException.class);
        }

        @Test
        @DisplayName("a fill on a PENDING order is refused — it was never sent")
        void cannotFillUnsentOrder() {
            Order o = order("100");
            assertThatThrownBy(() -> machine.applyFill(o, fill(o, "F1", "100", "485.00"), NOW))
                    .isInstanceOf(IllegalTransitionException.class);
        }

        @Test
        @DisplayName("overfilling fails loudly rather than silently capping")
        void cannotOverfill() {
            Order o = order("100");
            machine.markSent(o, "MOCK-1", NOW);
            machine.applyFill(o, fill(o, "F1", "60", "485.00"), NOW);

            assertThatThrownBy(() -> machine.applyFill(o, fill(o, "F2", "50", "485.00"), NOW))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("overfill");
            assertThat(o.getFilledQuantity()).isEqualByComparingTo("60");
        }

        @Test
        void terminalStatesReportThemselvesAsTerminal() {
            assertThat(OrderState.FILLED.isTerminal()).isTrue();
            assertThat(OrderState.REJECTED.isTerminal()).isTrue();
            assertThat(OrderState.CANCELLED.isTerminal()).isTrue();
            assertThat(OrderState.PENDING.isTerminal()).isFalse();
            assertThat(OrderState.SENT.isTerminal()).isFalse();
            assertThat(OrderState.PARTIALLY_FILLED.isTerminal()).isFalse();
        }
    }

    @Nested
    @DisplayName("construction guards")
    class Construction {

        @Test
        void rejectsNonPositiveQuantity() {
            assertThatThrownBy(() -> order("0"))
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        void rejectsNonPositiveFillPrice() {
            Order o = order("100");
            assertThatThrownBy(() ->
                    new Fill(o.getId(), "F1", new BigDecimal("10"), BigDecimal.ZERO, NOW))
                    .isInstanceOf(IllegalArgumentException.class);
        }
    }
}
