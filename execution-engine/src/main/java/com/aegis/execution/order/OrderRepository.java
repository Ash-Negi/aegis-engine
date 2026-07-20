package com.aegis.execution.order;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Repository
public interface OrderRepository extends JpaRepository<Order, UUID> {

    /** Idempotency lookup — has this (signal, symbol) already been ordered? */
    Optional<Order> findByClientOrderId(String clientOrderId);

    boolean existsByClientOrderId(String clientOrderId);

    /** Audit lineage: every order raised from one signal. */
    List<Order> findBySignalId(String signalId);

    List<Order> findByStateIn(List<OrderState> states);
}
