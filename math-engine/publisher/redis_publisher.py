"""
Aegis Engine — Redis Signal Publisher (Phase 3)
===============================================
Transmits a `TargetWeightSignal` to the execution engine.

Every publish does two things, and both are necessary:

    PUBLISH aegis:signals        → wakes live subscribers immediately
    SET     aegis:signals:latest → the last-known-good signal, durable

Pub/sub alone is fire-and-forget: a subscriber that is restarting, deploying,
or briefly disconnected simply misses the message, and there is no replay. A
key alone has no push — the consumer would have to poll. Doing both gives
low-latency delivery *and* a recovery path: the execution engine reads the
key on startup to learn the current target, then subscribes for updates. The
signal being idempotent target state (not an order) is what makes this safe —
re-reading the key just re-asserts the same target.
"""

import logging

import redis

from config import ExecutionConfig
from publisher.contract import TargetWeightSignal

log = logging.getLogger(__name__)


class SignalPublisher:
    """Publishes target-weight signals to Redis."""

    def __init__(self, config: ExecutionConfig | None = None, client=None):
        self.config = config or ExecutionConfig()
        # An injected client keeps this testable against fakeredis and lets
        # a caller share a connection pool.
        self.client = client or redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            decode_responses=True,
        )

    def publish(self, signal: TargetWeightSignal) -> int:
        """
        Publish a signal. Returns the number of subscribers that received it.

        A return of 0 is NOT an error — it means nothing was listening at
        this instant. The signal is still in `signal_latest_key`, so a
        consumer that starts later picks it up. Callers should log it, not
        retry.
        """
        payload = signal.to_json()

        # The key is written FIRST. If the process dies between the two
        # commands, a subscriber that missed the publish can still recover
        # from the key; the reverse order would leave a published signal with
        # no durable copy behind it.
        self.client.set(
            self.config.signal_latest_key, payload,
            ex=self.config.signal_ttl_seconds,
        )
        n = self.client.publish(self.config.signal_channel, payload)

        log.info("published signal %s (regime=%s, %d subscribers)",
                 signal.signal_id, signal.regime, n)
        return n

    def read_latest(self) -> TargetWeightSignal | None:
        """The last published signal, or None if the key expired or is unset."""
        payload = self.client.get(self.config.signal_latest_key)
        return TargetWeightSignal.from_json(payload) if payload else None
