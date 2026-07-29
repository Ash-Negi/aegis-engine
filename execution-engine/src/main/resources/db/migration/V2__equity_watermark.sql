-- Aegis Engine — equity high-water mark
--
-- One row, tracking the highest total equity this account has ever been
-- observed at. The drawdown circuit breaker compares current equity against
-- this peak rather than against starting cash, so a genuinely profitable
-- account that gives back a slice of its gains still halts — the breaker is
-- about protecting what was earned, not just the original deposit.

CREATE TABLE equity_watermark (
    id           INTEGER PRIMARY KEY,
    peak_equity  NUMERIC(20, 8) NOT NULL CHECK (peak_equity > 0),
    observed_at  TIMESTAMP WITH TIME ZONE NOT NULL,

    CONSTRAINT chk_watermark_singleton CHECK (id = 1)
);
