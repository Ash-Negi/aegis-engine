package com.aegis.execution.risk;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface EquityWatermarkRepository extends JpaRepository<EquityWatermark, Integer> {
}
