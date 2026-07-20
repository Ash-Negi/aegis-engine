"""
Aegis Engine — Signal Publisher Tests
=====================================
Run with: pytest tests/test_publisher.py -v

The publisher is a service boundary, so the tests here are about the CONTRACT
rather than about numbers. A malformed signal reaching Redis is a defect the
Java side cannot fix, so the validation is tested as hard as the transport.

Redis is faked (`fakeredis`) rather than mocked: a fake speaks the real
protocol, so a wrong key name or a missing TTL still fails the test. A mock
would happily accept any call and prove nothing.

──────────────────────────────────────────────────────────────────────────────
Review Log
──────────────────────────────────────────────────────────────────────────────
2026-07-20 — Phase 3, signal publisher

  test_rejects_* (unnormalised / negative / bad confidence)
      Validation happens at the boundary. These are the invariants the Java
      consumer is entitled to assume, so they must be impossible to publish.
  test_roundtrip
      to_json → from_json must be lossless, since the Java record is written
      against exactly this shape.
  test_publish_writes_durable_key / test_key_has_ttl
      The publisher does PUBLISH *and* SET. A late-starting consumer recovers
      from the key, so the key must exist and must expire.
  test_dust_zeroed
      Optimizer output has floating-point dust (FBTC at 1e-17). Publishing it
      would generate an order for a fraction of a share.
──────────────────────────────────────────────────────────────────────────────
"""

from datetime import datetime, timedelta, timezone

import fakeredis
import pandas as pd
import pytest

from config import ExecutionConfig
from publisher.contract import TargetWeightSignal, build_signal
from publisher.redis_publisher import SignalPublisher


def _weights(**kw):
    base = {"QQQ": 0.25, "GLDM": 0.25, "FBTC": 0.25, "VXUS": 0.25}
    base.update(kw)
    return pd.Series(base)


def _build(weights=None, **kw):
    args = dict(
        target_weights=weights if weights is not None else _weights(),
        regime="high_vol_meanrev",
        regime_confidence=0.98,
        as_of=pd.Timestamp("2026-04-27"),
        expected_turnover=0.0,
        rebalance_band_pct=5.0,
    )
    args.update(kw)
    return build_signal(**args)


class TestContract:
    def test_roundtrip(self):
        signal = _build()
        assert TargetWeightSignal.from_json(signal.to_json()) == signal

    def test_rejects_unnormalised_weights(self):
        with pytest.raises(ValueError, match="sum to"):
            _build(_weights(QQQ=0.50))

    def test_rejects_negative_weight(self):
        with pytest.raises(ValueError, match="negative weight"):
            _build(_weights(QQQ=-0.25, GLDM=0.75))

    def test_rejects_bad_confidence(self):
        with pytest.raises(ValueError, match="regime_confidence"):
            _build(regime_confidence=1.4)

    def test_dust_zeroed(self):
        """Floating-point dust from the optimizer must publish as exactly 0."""
        signal = _build(_weights(FBTC=1e-17, VXUS=0.5))
        assert signal.target_weights["FBTC"] == 0.0

    def test_signal_ids_unique(self):
        assert _build().signal_id != _build().signal_id

    def test_valid_until_respects_ttl(self):
        config = ExecutionConfig(signal_ttl_seconds=3600)
        signal = _build(config=config)
        span = (datetime.fromisoformat(signal.valid_until)
                - datetime.fromisoformat(signal.generated_at))
        assert span == timedelta(seconds=3600)

    def test_is_stale(self):
        past = datetime.now(timezone.utc) - timedelta(days=2)
        signal = _build(now=past, config=ExecutionConfig(signal_ttl_seconds=3600))
        assert signal.is_stale()
        assert not _build().is_stale()


class TestPublisher:
    @pytest.fixture
    def publisher(self):
        return SignalPublisher(
            ExecutionConfig(), client=fakeredis.FakeStrictRedis(decode_responses=True)
        )

    def test_publish_writes_durable_key(self, publisher):
        """A consumer that was not subscribed must still be able to recover."""
        signal = _build()
        publisher.publish(signal)
        assert publisher.read_latest() == signal

    def test_key_has_ttl(self, publisher):
        publisher.publish(_build())
        ttl = publisher.client.ttl(publisher.config.signal_latest_key)
        assert 0 < ttl <= publisher.config.signal_ttl_seconds

    def test_read_latest_empty(self, publisher):
        assert publisher.read_latest() is None

    def test_publish_with_no_subscribers_is_not_an_error(self, publisher):
        assert publisher.publish(_build()) == 0

    def test_latest_key_holds_most_recent(self, publisher):
        first, second = _build(), _build(regime="crisis")
        publisher.publish(first)
        publisher.publish(second)
        assert publisher.read_latest().regime == "crisis"
