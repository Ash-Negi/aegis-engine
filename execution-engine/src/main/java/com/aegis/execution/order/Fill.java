package com.aegis.execution.order;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

/**
 * One execution against an order — immutable, append-only.
 *
 * <p>Fills are never updated or deleted. The order's {@code filledQuantity} is
 * a running total that can be re-derived by summing these rows, which is what
 * makes reconciliation against a broker statement possible: if the derived
 * total and the stored total disagree, there is a bug, and the fills are the
 * side that is trusted.
 *
 * <p>{@code brokerFillId} is unique so a redelivered broker callback cannot
 * double-count. Brokers retry; that is normal, not exceptional.
 */
@Entity
@Table(name = "fills")
public class Fill {

    @Id
    private UUID id;

    @Column(name = "order_id", nullable = false)
    private UUID orderId;

    /** Broker's id for this execution — the dedupe key for callbacks. */
    @Column(name = "broker_fill_id", nullable = false, unique = true, length = 128)
    private String brokerFillId;

    @Column(nullable = false, precision = 20, scale = 8)
    private BigDecimal quantity;

    @Column(nullable = false, precision = 20, scale = 8)
    private BigDecimal price;

    @Column(name = "filled_at", nullable = false)
    private Instant filledAt;

    protected Fill() {
        // for JPA
    }

    public Fill(UUID orderId, String brokerFillId, BigDecimal quantity,
                BigDecimal price, Instant filledAt) {
        if (quantity.signum() <= 0) {
            throw new IllegalArgumentException("fill quantity must be positive, got " + quantity);
        }
        if (price.signum() <= 0) {
            throw new IllegalArgumentException("fill price must be positive, got " + price);
        }
        this.id = UUID.randomUUID();
        this.orderId = orderId;
        this.brokerFillId = brokerFillId;
        this.quantity = quantity;
        this.price = price;
        this.filledAt = filledAt;
    }

    public UUID getId() {
        return id;
    }

    public UUID getOrderId() {
        return orderId;
    }

    public String getBrokerFillId() {
        return brokerFillId;
    }

    public BigDecimal getQuantity() {
        return quantity;
    }

    public BigDecimal getPrice() {
        return price;
    }

    public Instant getFilledAt() {
        return filledAt;
    }

    public BigDecimal notional() {
        return quantity.multiply(price);
    }
}
