package com.aegis.execution.rebalance;

import com.aegis.execution.order.Order;
import com.aegis.execution.order.OrderSide;
import com.aegis.execution.portfolio.PortfolioSnapshot;
import com.aegis.execution.portfolio.Position;
import com.aegis.execution.signal.TargetWeightSignal;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Turns a target-weight signal into the orders that would achieve it.
 *
 * <p>This is the service that makes the "weights, not orders" split work. It
 * needs three things the math engine does not have: current positions, current
 * prices, and account equity. Given those, the arithmetic is
 * {@code targetShares = weight × equity / price}, and the order is the
 * difference from what is already held.
 *
 * <p>Two things stop this from churning the account.
 *
 * <p><b>The no-trade band.</b> An order is only raised when the weight has
 * drifted more than {@code rebalanceBandPct} (relative) from target. Prices
 * move every day; without a band, every signal would produce four orders for
 * a few dollars each and the transaction costs would swamp the benefit. The
 * band comes from the signal, so the Python backtest that measured cost drag
 * and the live engine apply the same rule — a divergence there would make the
 * backtest a fiction.
 *
 * <p><b>Whole shares.</b> Quantities are floored to integers, and an order
 * that rounds to zero shares is dropped rather than submitted.
 */
@Service
public class RebalanceService {

    private static final Logger log = LoggerFactory.getLogger(RebalanceService.class);
    private static final MathContext MC = new MathContext(16);
    private static final BigDecimal HUNDRED = new BigDecimal("100");

    /**
     * Compute the orders needed to move {@code snapshot} to the signal's
     * target weights, given current {@code prices}.
     *
     * @return orders in PENDING state; possibly empty if everything is inside
     *         the no-trade band, which is the normal steady-state outcome.
     */
    public List<Order> planOrders(TargetWeightSignal signal,
                                  PortfolioSnapshot snapshot,
                                  Map<String, BigDecimal> prices,
                                  Instant now) {
        BigDecimal equity = snapshot.totalEquity();
        if (equity.signum() <= 0) {
            log.warn("skipping rebalance: account equity is {}", equity);
            return List.of();
        }

        Map<String, Position> held = snapshot.positionsBySymbol();
        Map<String, BigDecimal> currentWeights = snapshot.currentWeights();
        BigDecimal band = signal.rebalanceBandPct().divide(HUNDRED, MC);

        List<Order> orders = new ArrayList<>();
        for (Map.Entry<String, BigDecimal> entry : signal.targetWeights().entrySet()) {
            String symbol = entry.getKey();
            BigDecimal targetWeight = entry.getValue();

            BigDecimal price = prices.get(symbol);
            if (price == null || price.signum() <= 0) {
                log.warn("no usable price for {}, skipping", symbol);
                continue;
            }

            BigDecimal currentWeight = currentWeights.getOrDefault(symbol, BigDecimal.ZERO);
            if (withinBand(currentWeight, targetWeight, band)) {
                continue;
            }

            BigDecimal targetShares = targetWeight.multiply(equity)
                    .divide(price, 0, RoundingMode.DOWN);
            BigDecimal heldShares = held.containsKey(symbol)
                    ? held.get(symbol).quantity()
                    : BigDecimal.ZERO;
            BigDecimal delta = targetShares.subtract(heldShares);

            if (delta.signum() == 0) {
                continue;
            }

            OrderSide side = delta.signum() > 0 ? OrderSide.BUY : OrderSide.SELL;
            orders.add(new Order(
                    clientOrderId(signal.signalId(), symbol),
                    signal.signalId(),
                    symbol,
                    side,
                    delta.abs(),
                    now));
        }

        log.info("signal {} ({}): planned {} order(s) against equity {}",
                signal.signalId(), signal.regime(), orders.size(), equity);
        return orders;
    }

    /**
     * Is the current weight close enough to target to leave alone?
     *
     * <p>The band is <em>relative</em> to the target — a 5% band on a 40%
     * target is ±2 percentage points, not ±5. This matches the Phase 1
     * backtest engine, which is the point: the cost drag it measured is only
     * meaningful if the live rule is the same rule.
     *
     * <p>A target of exactly zero has no relative band to speak of, so any
     * nonzero holding is out of band and gets liquidated.
     */
    private boolean withinBand(BigDecimal current, BigDecimal target, BigDecimal band) {
        if (target.signum() == 0) {
            return current.signum() == 0;
        }
        BigDecimal drift = current.subtract(target).abs().divide(target, MC);
        return drift.compareTo(band) <= 0;
    }

    /**
     * The idempotency key. Deterministic in (signal, symbol), so re-processing
     * a redelivered signal produces the same key and the unique constraint on
     * the orders table refuses the duplicate rather than double-trading.
     */
    static String clientOrderId(String signalId, String symbol) {
        return signalId + ":" + symbol;
    }
}
