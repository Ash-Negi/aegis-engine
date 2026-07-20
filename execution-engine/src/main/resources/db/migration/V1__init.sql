-- Aegis Engine — Execution Ledger, initial schema
--
-- Three tables, in the order events actually happen: a signal arrives, it
-- raises orders, orders receive fills. Every row is evidence, so the schema
-- favours append-only history over mutable current state — `fills` is never
-- updated, and an order's running totals can always be re-derived by summing
-- its fills. When the derived total and the stored total disagree, the fills
-- are the side that is trusted.

CREATE TABLE signals (
    signal_id         VARCHAR(64) PRIMARY KEY,
    schema_version    INTEGER      NOT NULL,
    regime            VARCHAR(32)  NOT NULL,
    regime_confidence DOUBLE PRECISION NOT NULL,
    as_of             VARCHAR(16)  NOT NULL,
    generated_at      TIMESTAMP WITH TIME ZONE  NOT NULL,
    received_at       TIMESTAMP WITH TIME ZONE  NOT NULL,
    -- Rejected signals are recorded too. A dropped signal that leaves no
    -- trace is indistinguishable from one that never arrived.
    accepted          BOOLEAN      NOT NULL,
    reject_reason     VARCHAR(512),
    raw_payload       TEXT         NOT NULL
);

CREATE INDEX idx_signals_received_at ON signals (received_at DESC);

CREATE TABLE orders (
    id                UUID PRIMARY KEY,
    -- Idempotency key, deterministic in (signal, symbol). The UNIQUE
    -- constraint is what actually stops a redelivered signal from
    -- double-trading; the application check in front of it is an
    -- optimisation, not the guarantee.
    client_order_id   VARCHAR(128) NOT NULL UNIQUE,
    signal_id         VARCHAR(64)  NOT NULL REFERENCES signals (signal_id),
    symbol            VARCHAR(16)  NOT NULL,
    side              VARCHAR(8)   NOT NULL,
    -- NUMERIC, never DOUBLE PRECISION. Binary floating point cannot represent
    -- 0.1 exactly, so accumulating fills as doubles leaves a filled order a
    -- few ulps short of its target and it never reaches FILLED.
    target_quantity   NUMERIC(20, 8) NOT NULL CHECK (target_quantity > 0),
    filled_quantity   NUMERIC(20, 8) NOT NULL DEFAULT 0 CHECK (filled_quantity >= 0),
    average_fill_price NUMERIC(20, 8),
    state             VARCHAR(24)  NOT NULL,
    broker_order_id   VARCHAR(128),
    reject_reason     VARCHAR(512),
    created_at        TIMESTAMP WITH TIME ZONE  NOT NULL,
    updated_at        TIMESTAMP WITH TIME ZONE  NOT NULL,
    -- Optimistic lock. Two fill callbacks for one order can land on different
    -- threads; without this, one read-modify-write silently overwrites the
    -- other and quantity goes missing.
    version           BIGINT,

    CONSTRAINT chk_not_overfilled CHECK (filled_quantity <= target_quantity)
);

CREATE INDEX idx_orders_signal_id ON orders (signal_id);
CREATE INDEX idx_orders_state ON orders (state);

CREATE TABLE fills (
    id             UUID PRIMARY KEY,
    order_id       UUID NOT NULL REFERENCES orders (id),
    -- Brokers retry callbacks; that is normal, not exceptional. Dedupe is on
    -- the broker's own fill id, never on (quantity, price, timestamp) — two
    -- genuine fills of the same size at the same price milliseconds apart are
    -- real, and collapsing them would lose quantity.
    broker_fill_id VARCHAR(128) NOT NULL UNIQUE,
    quantity       NUMERIC(20, 8) NOT NULL CHECK (quantity > 0),
    price          NUMERIC(20, 8) NOT NULL CHECK (price > 0),
    filled_at      TIMESTAMP WITH TIME ZONE  NOT NULL
);

CREATE INDEX idx_fills_order_id ON fills (order_id);
