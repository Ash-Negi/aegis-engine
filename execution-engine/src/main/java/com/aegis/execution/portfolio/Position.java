package com.aegis.execution.portfolio;

import java.math.BigDecimal;

/** A held quantity of a symbol and its last known mark. */
public record Position(String symbol, BigDecimal quantity, BigDecimal lastPrice) {

    public BigDecimal marketValue() {
        return quantity.multiply(lastPrice);
    }
}
