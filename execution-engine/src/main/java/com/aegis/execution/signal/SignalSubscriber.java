package com.aegis.execution.signal;

import com.aegis.execution.config.AegisProperties;
import com.aegis.execution.execution.ExecutionService;
import com.aegis.execution.portfolio.PortfolioService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.data.redis.connection.Message;
import org.springframework.data.redis.connection.MessageListener;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;

/**
 * Receives target-weight signals from Redis and hands them to the execution
 * service.
 *
 * <p>Two intake paths, and both are needed. {@link #onMessage} is the live
 * pub/sub subscription — low latency, but fire-and-forget: anything published
 * while this service was restarting is simply gone. {@link #recoverOnStartup}
 * closes that gap by reading the durable {@code latestKey} once the context is
 * ready, so a redeployed engine learns the current target immediately instead
 * of trading yesterday's portfolio until the next publish.
 *
 * <p>Replaying the key on every startup is safe only because signals are
 * idempotent target state — {@code ExecutionService} dedupes on signal id, and
 * even without that, re-asserting the same target produces no orders once the
 * portfolio is inside the band.
 *
 * <p>A message that cannot even be parsed is logged and dropped. There is
 * nothing useful to do with it, and throwing would take down the listener
 * container for every subsequent message.
 */
@Component
public class SignalSubscriber implements MessageListener {

    private static final Logger log = LoggerFactory.getLogger(SignalSubscriber.class);

    private final ObjectMapper objectMapper;
    private final ExecutionService executionService;
    private final PortfolioService portfolioService;
    private final StringRedisTemplate redis;
    private final AegisProperties properties;

    public SignalSubscriber(ObjectMapper objectMapper,
                            ExecutionService executionService,
                            PortfolioService portfolioService,
                            StringRedisTemplate redis,
                            AegisProperties properties) {
        this.objectMapper = objectMapper;
        this.executionService = executionService;
        this.portfolioService = portfolioService;
        this.redis = redis;
        this.properties = properties;
    }

    @Override
    public void onMessage(Message message, byte[] pattern) {
        handle(new String(message.getBody(), StandardCharsets.UTF_8), "pubsub");
    }

    /**
     * A Redis that is down at boot must not stop the engine from starting.
     * The ledger is the source of truth for what has already been traded, and
     * the subscription reconnects on its own — refusing to start would turn a
     * transient dependency outage into an outage of the engine itself.
     */
    @EventListener(ApplicationReadyEvent.class)
    public void recoverOnStartup() {
        String payload;
        try {
            payload = redis.opsForValue().get(properties.getSignal().getLatestKey());
        } catch (Exception e) {
            log.warn("could not read {} at startup; continuing without recovery",
                    properties.getSignal().getLatestKey(), e);
            return;
        }
        if (payload == null) {
            log.info("no signal in {} at startup; waiting for the next publish",
                    properties.getSignal().getLatestKey());
            return;
        }
        log.info("recovering last-known-good signal from {}",
                properties.getSignal().getLatestKey());
        handle(payload, "startup-recovery");
    }

    private void handle(String payload, String source) {
        TargetWeightSignal signal;
        try {
            signal = objectMapper.readValue(payload, TargetWeightSignal.class);
        } catch (Exception e) {
            log.error("unparseable signal from {}, dropping: {}", source, payload, e);
            return;
        }
        try {
            executionService.onSignal(
                    signal, payload,
                    portfolioService.snapshot(),
                    portfolioService.prices());
        } catch (Exception e) {
            log.error("failed to process signal {} from {}", signal.signalId(), source, e);
        }
    }
}
