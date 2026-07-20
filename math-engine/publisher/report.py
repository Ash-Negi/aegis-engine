"""
Aegis Engine — Phase 3: Signal Publisher Demo
=============================================
Generates a live signal from the full Phase 1+2 stack and publishes it.

Usage (from math-engine/):
    python -m publisher.report            # generate + print, no Redis needed
    python -m publisher.report --publish  # also publish to Redis
"""

import argparse
import json

from config import ExecutionConfig
from publisher import generate_signal, SignalPublisher


def _header(text):
    print(f"\n{'═' * 70}\n  {text}\n{'═' * 70}")


def publisher_report(publish: bool = False, config: ExecutionConfig | None = None):
    config = config or ExecutionConfig()
    _header("AEGIS ENGINE — Phase 3: Signal Publisher")

    signal = generate_signal(exec_config=config)

    print("\n  Generated signal:\n")
    print("    " + json.dumps(json.loads(signal.to_json()), indent=2).replace("\n", "\n    "))

    print(f"\n  Payload size: {len(signal.to_json())} bytes")
    print(f"  Stale: {signal.is_stale()}")

    if not publish:
        print(f"\n  Not published (pass --publish to send to "
              f"{config.redis_host}:{config.redis_port}).\n")
        return signal

    n = SignalPublisher(config).publish(signal)
    print(f"\n  Published to {config.signal_channel} — {n} subscriber(s) received it.")
    print(f"  Durable copy at {config.signal_latest_key} "
          f"(TTL {config.signal_ttl_seconds}s).\n")
    return signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and optionally publish a signal.")
    parser.add_argument("--publish", action="store_true", help="publish to Redis")
    args = parser.parse_args()
    publisher_report(publish=args.publish)
