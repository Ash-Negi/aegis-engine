package com.aegis.execution.portfolio;

import com.aegis.execution.config.AegisProperties;
import com.aegis.execution.order.Fill;
import com.aegis.execution.order.Order;
import com.aegis.execution.order.OrderSide;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The engine's view of the account: cash, positions, and last prices.
 *
 * <p>In-memory, seeded from configuration — the stand-in for a real brokerage
 * account query while the Alpaca integration is stubbed. The interface is what
 * matters: {@link #snapshot()} and {@link #prices()} are what
 * {@code RebalanceService} consumes, and swapping this for a client that calls
 * a real account endpoint changes nothing above it.
 *
 * <p>Positions are updated from fills, not from submitted orders. An order
 * that has been sent has not moved anything; only an execution has. Updating
 * on submission would make the next rebalance compute against a position that
 * does not exist yet.
 */
@Service
public class PortfolioService {

    private final Map<String, BigDecimal> quantities = new ConcurrentHashMap<>();
    private final Map<String, BigDecimal> prices = new ConcurrentHashMap<>();
    private volatile BigDecimal cash;

    public PortfolioService(AegisProperties properties) {
        this.cash = properties.getAccount().getStartingCash();
        this.prices.putAll(properties.getAccount().getPrices());
    }

    public PortfolioSnapshot snapshot() {
        List<Position> positions = new ArrayList<>();
        quantities.forEach((symbol, quantity) -> {
            if (quantity.signum() != 0) {
                positions.add(new Position(
                        symbol, quantity, prices.getOrDefault(symbol, BigDecimal.ZERO)));
            }
        });
        return new PortfolioSnapshot(cash, positions);
    }

    public Map<String, BigDecimal> prices() {
        return Map.copyOf(prices);
    }

    public void setPrice(String symbol, BigDecimal price) {
        prices.put(symbol, price);
    }

    /** Apply an execution: shares move one way, cash the other. */
    public synchronized void applyFill(Order order, Fill fill) {
        BigDecimal signed = order.getSide() == OrderSide.BUY
                ? fill.getQuantity()
                : fill.getQuantity().negate();
        quantities.merge(order.getSymbol(), signed, BigDecimal::add);
        cash = order.getSide() == OrderSide.BUY
                ? cash.subtract(fill.notional())
                : cash.add(fill.notional());
        prices.put(order.getSymbol(), fill.getPrice());
    }
}
