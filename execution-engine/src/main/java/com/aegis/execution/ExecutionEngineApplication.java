package com.aegis.execution;

import com.aegis.execution.config.AegisProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

import java.time.Clock;

@SpringBootApplication
@EnableConfigurationProperties(AegisProperties.class)
public class ExecutionEngineApplication {

    public static void main(String[] args) {
        SpringApplication.run(ExecutionEngineApplication.class, args);
    }

    /**
     * Injected rather than called statically, so tests can pin time and assert
     * on staleness without sleeping.
     */
    @Bean
    public Clock clock() {
        return Clock.systemUTC();
    }

    @Bean
    public ObjectMapper objectMapper() {
        return new ObjectMapper().registerModule(new JavaTimeModule());
    }
}
