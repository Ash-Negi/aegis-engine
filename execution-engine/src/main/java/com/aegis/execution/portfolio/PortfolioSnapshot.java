package com.aegis.execution.portfolio;

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The account as the execution engine currently understands it: cash plus
 * positions, at a point in time.
 *
 * <p>Total equity includes cash, because target weights are fractions of the
 * whole account. Ignoring cash would make the weights add to more than the
 * money available and every rebalance would try to buy slightly too much.
 */
public record PortfolioSnapshot(BigDecimal cash, List<Position> positions) {

    private static final MathContext MC = new MathContext(16);

    public BigDecimal totalEquity() {
        BigDecimal total = cash;
        for (Position position : positions) {
            total = total.add(position.marketValue());
        }
        return total;
    }

    /** Current weights by symbol, as fractions of total equity. */
    public Map<String, BigDecimal> currentWeights() {
        BigDecimal equity = totalEquity();
        Map<String, BigDecimal> weights = new HashMap<>();
        if (equity.signum() <= 0) {
            return weights;
        }
        for (Position position : positions) {
            weights.put(position.symbol(), position.marketValue().divide(equity, MC));
        }
        return weights;
    }

    public Map<String, Position> positionsBySymbol() {
        Map<String, Position> bySymbol = new HashMap<>();
        for (Position position : positions) {
            bySymbol.put(position.symbol(), position);
        }
        return bySymbol;
    }
}
