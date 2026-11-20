import unittest

from core.autopilot import Autopilot


class AutopilotEventFreshnessTests(unittest.TestCase):
    def test_live_websocket_uses_local_receipt_time_when_server_clock_is_offset(self):
        now_ms = 2_000_000
        candle = {
            "source": "binance_websocket",
            "event_time": now_ms - 120_000,
            "close_time": now_ms - 119_999,
            "received_at_ms": now_ms - 750,
        }

        age_ms = Autopilot._candle_event_age_ms(
            candle, received_at_ms=candle["received_at_ms"], now_ms=now_ms
        )

        self.assertEqual(age_ms, 750)

    def test_cache_reconciliation_keeps_original_candle_age(self):
        now_ms = 2_000_000
        candle = {
            "source": "binance_cache_reconciliation",
            "event_time": now_ms - 120_000,
            "received_at_ms": now_ms - 750,
        }

        age_ms = Autopilot._candle_event_age_ms(
            candle, received_at_ms=candle["received_at_ms"], now_ms=now_ms
        )

        self.assertEqual(age_ms, 120_000)

    def test_missing_timestamps_fail_closed_as_stale(self):
        self.assertEqual(
            Autopilot._candle_event_age_ms({}, now_ms=2_000_000),
            10**12,
        )


if __name__ == "__main__":
    unittest.main()
