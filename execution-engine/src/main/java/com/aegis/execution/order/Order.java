package com.aegis.execution.order;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Version;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

/**
 * A single order and its current state.
 *
 * <p>Quantities and prices are {@link BigDecimal}, never {@code double}.
 * Binary floating point cannot represent 0.1 exactly, so accumulating fills as
 * doubles leaves a filled order a few ulps short of its target quantity and it
 * never reaches FILLED. Money and share counts are decimal quantities and are
 * stored as such.
 *
 * <p>{@code clientOrderId} is the idempotency key. It is derived from the
 * signal that produced the order plus the symbol, so re-processing a redelivered
 * signal produces the same key and the unique constraint refuses the duplicate.
 * The broker is also given this id, so a network timeout on submission can be
 * resolved by asking the broker about that id rather than by guessing.
 */
@Entity
@Table(name = "orders")
public class Order {

    @Id
    private UUID id;

    /** Idempotency key — unique per (signal, symbol). */
    @Column(name = "client_order_id", nullable = false, unique = true, length = 128)
    private String clientOrderId;

    /** The signal this order was raised from, for audit lineage. */
    @Column(name = "signal_id", nullable = false, length = 64)
    private String signalId;

    @Column(nullable = false, length = 16)
    private String symbol;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 8)
    private OrderSide side;

    @Column(name = "target_quantity", nullable = false, precision = 20, scale = 8)
    private BigDecimal targetQuantity;

    @Column(name = "filled_quantity", nullable = false, precision = 20, scale = 8)
    private BigDecimal filledQuantity = BigDecimal.ZERO;

    /** Quantity-weighted average price of the fills received so far. */
    @Column(name = "average_fill_price", precision = 20, scale = 8)
    private BigDecimal averageFillPrice;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 24)
    private OrderState state = OrderState.PENDING;

    @Column(name = "broker_order_id", length = 128)
    private String brokerOrderId;

    @Column(name = "reject_reason", length = 512)
    private String rejectReason;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    /**
     * Optimistic locking. Two fill callbacks for the same order can land on
     * different threads; without this, one read-modify-write silently
     * overwrites the other and quantity goes missing.
     */
    @Version
    private Long version;

    protected Order() {
        // for JPA
    }

    public Order(String clientOrderId, String signalId, String symbol,
                 OrderSide side, BigDecimal targetQuantity, Instant now) {
        if (targetQuantity.signum() <= 0) {
            throw new IllegalArgumentException(
                    "order quantity must be positive, got " + targetQuantity);
        }
        this.id = UUID.randomUUID();
        this.clientOrderId = clientOrderId;
        this.signalId = signalId;
        this.symbol = symbol;
        this.side = side;
        this.targetQuantity = targetQuantity;
        this.createdAt = now;
        this.updatedAt = now;
    }

    public BigDecimal remainingQuantity() {
        return targetQuantity.subtract(filledQuantity);
    }

    public UUID getId() {
        return id;
    }

    public String getClientOrderId() {
        return clientOrderId;
    }

    public String getSignalId() {
        return signalId;
    }

    public String getSymbol() {
        return symbol;
    }

    public OrderSide getSide() {
        return side;
    }

    public BigDecimal getTargetQuantity() {
        return targetQuantity;
    }

    public BigDecimal getFilledQuantity() {
        return filledQuantity;
    }

    public BigDecimal getAverageFillPrice() {
        return averageFillPrice;
    }

    public OrderState getState() {
        return state;
    }

    public String getBrokerOrderId() {
        return brokerOrderId;
    }

    public String getRejectReason() {
        return rejectReason;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    // Mutation is package-private: only OrderStateMachine may change an
    // order, so every state change goes through the transition check.
    void setState(OrderState state) {
        this.state = state;
    }

    void setFilledQuantity(BigDecimal filledQuantity) {
        this.filledQuantity = filledQuantity;
    }

    void setAverageFillPrice(BigDecimal averageFillPrice) {
        this.averageFillPrice = averageFillPrice;
    }

    void setBrokerOrderId(String brokerOrderId) {
        this.brokerOrderId = brokerOrderId;
    }

    void setRejectReason(String rejectReason) {
        this.rejectReason = rejectReason;
    }

    void setUpdatedAt(Instant updatedAt) {
        this.updatedAt = updatedAt;
    }
}
