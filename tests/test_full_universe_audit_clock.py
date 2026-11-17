import tempfile
import time
import unittest

import pandas as pd

from core.autopilot import Autopilot
from core.binance_websocket import BinanceKlineStream


class FakeFeed:
    pass


class FakeCoordinator:
    mode = "DEMO"
    demo_positions = {}
    last_signals = {"spot": None, "futures": None}
    learning = type("Learning", (), {"allows": staticmethod(lambda _key: True)})()

    def _agent_enabled(self, _domain):
        return True


def recent_frame(rows=60, last_open_ms=None):
    interval_ms = 300_000
    now_ms = int(time.time() * 1000)
    floor_ms = (now_ms // interval_ms) * interval_ms
    last_open_ms = last_open_ms if last_open_ms is not None else floor_ms - 2 * interval_ms
    opens = pd.to_datetime(
        [last_open_ms - (rows - 1 - i) * interval_ms for i in range(rows)],
        unit="ms", utc=True,
    )
    frame = pd.DataFrame({
        "open_time": opens,
        "open": [100.0] * rows,
        "high": [101.0] * rows,
        "low": [99.0] * rows,
        "close": [100.5] * rows,
        "volume": [10.0] * rows,
        "close_time": opens + pd.Timedelta(milliseconds=interval_ms - 1),
        "quote_volume": [1000.0] * rows,
        "trades": [10] * rows,
        "taker_buy_base": [5.0] * rows,
        "taker_buy_quote": [500.0] * rows,
        "ignore": [0.0] * rows,
        "is_closed": [True] * rows,
    })
    frame.attrs.update({"timeframe": "5m", "interval": "5m"})
    return frame


class FullUniverseAuditClockTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.stream = BinanceKlineStream(
            FakeFeed(),
            lambda market, event: self.events.append((market, dict(event))),
            max_symbols=0,
        )

    def test_quiet_symbol_gets_zero_volume_closed_candle_for_each_cycle(self):
        interval_ms = 300_000
        now_ms = int(time.time() * 1000)
        floor_ms = (now_ms // interval_ms) * interval_ms
        target_open = floor_ms - interval_ms
        previous_open = target_open - interval_ms
        frame = recent_frame(rows=60, last_open_ms=previous_open)
        self.stream._merge_frame("spot", "QUIETUSDT", "5m", frame)

        result = self.stream._emit_universe_close(
            "spot", "QUIETUSDT", target_open, floor_ms + 1000
        )

        self.assertEqual(result, "synthetic")
        self.assertEqual(len(self.events), 1)
        event = self.events[0][1]
        self.assertTrue(event["is_closed"])
        self.assertEqual(event["open_time"], target_open)
        self.assertEqual(event["volume"], 0.0)
        self.assertEqual(event["close"], 100.5)
        self.assertTrue(event["cache_ready"])
        self.assertEqual(len(event["market_data"]["5m"]), 61)

    def test_partial_websocket_candle_is_finalized_not_flattened(self):
        interval_ms = 300_000
        now_ms = int(time.time() * 1000)
        floor_ms = (now_ms // interval_ms) * interval_ms
        target_open = floor_ms - interval_ms
        self.stream._latest_candles[("futures", "ACTIVEUSDT")] = {
            "symbol": "ACTIVEUSDT", "interval": "5m",
            "open_time": target_open, "open": 10.0, "high": 12.0,
            "low": 9.5, "close": 11.5, "volume": 123.0,
            "is_closed": False, "event_time": target_open + 200_000,
        }

        result = self.stream._emit_universe_close(
            "futures", "ACTIVEUSDT", target_open, floor_ms + 1000
        )

        self.assertEqual(result, "emitted")
        event = self.events[0][1]
        self.assertEqual(event["open"], 10.0)
        self.assertEqual(event["high"], 12.0)
        self.assertEqual(event["low"], 9.5)
        self.assertEqual(event["close"], 11.5)
        self.assertEqual(event["volume"], 123.0)
        self.assertEqual(event["source"], "binance_5m_finalized_partial")

    def test_each_symbol_close_is_emitted_only_once_per_cycle(self):
        interval_ms = 300_000
        now_ms = int(time.time() * 1000)
        floor_ms = (now_ms // interval_ms) * interval_ms
        target_open = floor_ms - interval_ms

        first = self.stream._emit_universe_close(
            "futures", "QUIETUSDT", target_open, floor_ms + 1000
        )
        second = self.stream._emit_universe_close(
            "futures", "QUIETUSDT", target_open, floor_ms + 2000
        )

        self.assertEqual(first, "synthetic")
        self.assertEqual(second, "already_seen")
        self.assertEqual(len(self.events), 1)

    def test_recovered_close_marks_late_cycle_as_audit_only(self):
        interval_ms = 300_000
        now_ms = int(time.time() * 1000)
        floor_ms = (now_ms // interval_ms) * interval_ms
        target_open = floor_ms - 2 * interval_ms

        self.stream._emit_universe_close(
            "spot", "LATEUSDT", target_open, floor_ms + 1000
        )

        self.assertTrue(self.events[0][1]["audit_only"])


class CacheReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.autopilot = Autopilot(
            FakeCoordinator(), config_path=self.temp.name + "/autopilot.json"
        )
        self.autopilot.submit_analysis = lambda *_args: True

    def tearDown(self):
        self.autopilot._event_executor.shutdown(wait=True)
        for pool in self.autopilot._analysis_pools.values():
            pool.shutdown(wait=True)
        self.temp.cleanup()

    def test_short_history_close_waits_until_sixty_contiguous_bars_are_ready(self):
        now_ms = int(time.time() * 1000)
        interval_ms = 300_000
        floor_ms = now_ms // interval_ms * interval_ms
        target_open = floor_ms - interval_ms
        event = {
            "symbol": "NEWUSDT", "interval": "5m",
            "open_time": target_open, "close_time": target_open + interval_ms - 1,
            "event_time": target_open + interval_ms,
            "is_closed": True, "cache_ready": False,
            "market_data": {"5m": recent_frame(rows=25)},
        }

        self.autopilot.on_market_kline("spot", event)

        self.assertIn(("spot", "NEWUSDT"), self.autopilot._waiting_cache_events)
        self.assertNotIn(("spot", "NEWUSDT"), self.autopilot._analysis_pending)

    def test_cache_ready_callback_replays_retained_close_without_double_counting_event(self):
        now_ms = int(time.time() * 1000)
        interval_ms = 300_000
        floor_ms = now_ms // interval_ms * interval_ms
        target_open = floor_ms - interval_ms
        event = {
            "symbol": "BTCUSDT", "interval": "5m",
            "open_time": target_open, "close_time": target_open + interval_ms - 1,
            "event_time": target_open + interval_ms,
            "is_closed": True, "cache_ready": False,
            "market_data": None,
        }
        self.autopilot.on_market_kline("futures", event)
        before = self.autopilot.symbol_auditor_status()["futures"]["events_received"]

        accepted = self.autopilot.on_symbol_cache_ready(
            "futures", "BTCUSDT", {"5m": recent_frame(rows=60)}
        )

        self.assertTrue(accepted)
        self.assertNotIn(("futures", "BTCUSDT"), self.autopilot._waiting_cache_events)
        self.assertEqual(
            self.autopilot.symbol_auditor_status()["futures"]["events_received"],
            before,
        )
        self.assertEqual(
            self.autopilot.symbol_auditor_status()["futures"]["cache_replayed"], 1
        )


if __name__ == "__main__":
    unittest.main()
