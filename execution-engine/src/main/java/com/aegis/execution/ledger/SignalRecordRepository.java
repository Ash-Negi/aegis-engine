package com.aegis.execution.ledger;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface SignalRecordRepository extends JpaRepository<SignalRecord, String> {
}
