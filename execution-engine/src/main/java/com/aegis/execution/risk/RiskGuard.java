package com.aegis.execution.risk;

import com.aegis.execution.config.AegisProperties;
import com.aegis.execution.portfolio.PortfolioSnapshot;
import com.aegis.execution.signal.TargetWeightSignal;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.time.Instant;
import java.util.Map;
import java.util.Optional;

/**
 * Runtime safety ceilings that gate a signal independent of what the math
 * engine decided. Only this class mutates {@link EquityWatermark}, the same
 * way only {@code OrderStateMachine} mutates an {@code Order} — the
 * high-water mark is state other code should observe, not adjust.
 *
 * <p>Two checks, each answering a different question:
 *
 * <ul>
 *   <li><b>Drawdown</b> — is the account itself in a state where it should
 *       keep trading at all, regardless of what this signal proposes?</li>
 *   <li><b>Concentration</b> — does this specific signal's own weights
 *       violate a hard per-asset ceiling? This is a backstop independent of
 *       whatever constraint the optimizer was supposed to enforce upstream —
 *       a bad regime call or a bug in that layer should not be able to bet
 *       the book on one symbol.</li>
 * </ul>
 */
@Component
public class RiskGuard {

    private static final MathContext MC = new MathContext(16);
    private static final BigDecimal HUNDRED = new BigDecimal("100");

    private final EquityWatermarkRepository watermarks;
    private final AegisProperties properties;

    public RiskGuard(EquityWatermarkRepository watermarks, AegisProperties properties) {
        this.watermarks = watermarks;
        this.properties = properties;
    }

    /**
     * Record the observed equity against the high-water mark, ratcheting it
     * up on a new peak.
     *
     * @return the reject reason if the resulting drawdown breaches the
     *         configured limit, or empty if trading may proceed
     */
    public Optional<String> checkDrawdown(PortfolioSnapshot snapshot, Instant now) {
        BigDecimal equity = snapshot.totalEquity();
        EquityWatermark mark = watermarks.findById(1).orElse(null);

        if (mark == null) {
            watermarks.save(new EquityWatermark(equity, now));
            return Optional.empty();
        }
        if (equity.compareTo(mark.getPeakEquity()) >= 0) {
            mark.raiseTo(equity, now);
            watermarks.save(mark);
            return Optional.empty();
        }

        BigDecimal peak = mark.getPeakEquity();
        BigDecimal drawdown = peak.subtract(equity).divide(peak, MC);
        BigDecimal limit = properties.getRisk().getMaxDrawdownPct();
        if (drawdown.compareTo(limit) > 0) {
            return Optional.of(
                    "drawdown breach: equity %s is down %s from peak %s (limit %s)"
                            .formatted(equity, pct(drawdown), peak, pct(limit)));
        }
        return Optional.empty();
    }

    /**
     * @return the reject reason if any symbol in {@code signal} exceeds the
     *         per-position weight ceiling, or empty otherwise
     */
    public Optional<String> checkConcentration(TargetWeightSignal signal) {
        BigDecimal limit = properties.getRisk().getMaxPositionWeight();
        for (Map.Entry<String, BigDecimal> entry : signal.targetWeights().entrySet()) {
            if (entry.getValue().compareTo(limit) > 0) {
                return Optional.of(
                        "concentration breach: %s target weight %s exceeds limit %s"
                                .formatted(entry.getKey(), entry.getValue(), limit));
            }
        }
        return Optional.empty();
    }

    private static String pct(BigDecimal fraction) {
        return fraction.multiply(HUNDRED).setScale(2, RoundingMode.HALF_UP) + "%";
    }
}
