package com.aegis.execution.config;

import com.aegis.execution.signal.SignalSubscriber;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.listener.ChannelTopic;
import org.springframework.data.redis.listener.RedisMessageListenerContainer;

import java.util.concurrent.Executors;

/**
 * Wires the Redis subscription for target-weight signals.
 *
 * <p>The listener container runs on virtual threads. Signal handling is
 * mostly waiting — on Postgres, on the broker — and a virtual thread parked on
 * I/O costs nothing, so the engine can hold many in-flight handlers without
 * sizing a platform-thread pool against them. This is the Java 21 feature the
 * execution layer was designed around.
 *
 * <p>The subscription can be switched off. Reconciliation and replay runs
 * work against the ledger alone and have no business consuming live signals;
 * the test suite uses the same switch rather than requiring a Redis server.
 */
@Configuration
@ConditionalOnProperty(name = "aegis.signal.subscribe-enabled",
        havingValue = "true", matchIfMissing = true)
public class RedisConfig {

    @Bean
    public RedisMessageListenerContainer signalListenerContainer(
            RedisConnectionFactory connectionFactory,
            SignalSubscriber subscriber,
            AegisProperties properties) {
        RedisMessageListenerContainer container = new RedisMessageListenerContainer();
        container.setConnectionFactory(connectionFactory);
        container.setTaskExecutor(Executors.newVirtualThreadPerTaskExecutor());
        container.addMessageListener(
                subscriber, new ChannelTopic(properties.getSignal().getChannel()));
        return container;
    }
}
