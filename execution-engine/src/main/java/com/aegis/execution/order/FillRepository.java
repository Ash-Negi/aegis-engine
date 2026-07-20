package com.aegis.execution.order;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

@Repository
public interface FillRepository extends JpaRepository<Fill, UUID> {

    List<Fill> findByOrderId(UUID orderId);

    /** Dedupe key for redelivered broker callbacks. */
    boolean existsByBrokerFillId(String brokerFillId);
}
