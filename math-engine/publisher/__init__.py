"""
Aegis Engine — Signal Publisher (Phase 3)
=========================================
The math engine's outbound edge: turn Phase 1+2 output into a versioned,
idempotent, expiring signal and put it on Redis for the execution engine.

    contract         TargetWeightSignal — the Python↔Java wire format
    pipeline         prices → optimizer → regime tilt → signal
    redis_publisher  PUBLISH + SET, so late consumers can still recover
"""

from publisher.contract import TargetWeightSignal, build_signal
from publisher.pipeline import generate_signal
from publisher.redis_publisher import SignalPublisher

__all__ = [
    "TargetWeightSignal",
    "build_signal",
    "generate_signal",
    "SignalPublisher",
]
