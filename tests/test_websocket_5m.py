import json
import tempfile
import unittest

import pandas as pd

from core.autopilot import Autopilot
from core.binance_websocket import BinanceKlineStream
from core.scalping_v3 import FuturesScalpingStrategyV3


class FakeFeed:
    pass


class FakeApp:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeCoordinator:
    mode = "DEMO"
    demo_positions = {}
    learning = type("Learning", (), {"allows": staticmethod(lambda _key: True)})()

    def _agent_enabled(self, _domain):
        return True


def candle_frame(interval, rows=60, closed=True):
    index = pd.date_range("2026-01-01", periods=rows, freq=interval, tz="UTC")
    frame = pd.DataFrame({
        "open_time": index,
        "open": [100.0] * rows,
        "high": [101.0] * rows,
        "low": [99.0] * rows,
        "close": [100.5] * rows,
        "volume": [10.0] * rows,
        "close_time": index + pd.Timedelta(interval) - pd.Timedelta(milliseconds=1),
        "is_closed": [True] * rows,
    })
    frame.iloc[-1, frame.columns.get_loc("is_closed")] = closed
    frame.attrs.update({"timeframe": interval, "interval": interval})
    return frame


class ClosedCandleGuardTests(unittest.TestCase):
    def test_5m_requires_explicit_timeframe_and_closed_flag(self):
        frame = candle_frame("5min")
        frame.attrs["timeframe"] = "1m"
        self.assertFalse(FuturesScalpingStrategyV3.frame_is_closed_5m(frame))

        frame = candle_frame("5min")
        frame.attrs["timeframe"] = "5m"
        frame["is_closed"] = True
        self.assertTrue(FuturesScalpingStrategyV3.frame_is_closed_5m(frame))

        frame.iloc[-1, frame.columns.get_loc("is_closed")] = False
        self.assertFalse(FuturesScalpingStrategyV3.frame_is_closed_5m(frame))

    def test_higher_timeframe_must_match_and_be_closed(self):
        frame = candle_frame("15min")
        frame.attrs["timeframe"] = "15m"
        self.assertTrue(FuturesScalpingStrategyV3.frame_is_closed(frame, "15m"))
        self.assertFalse(FuturesScalpingStrategyV3.frame_is_closed(frame, "1h"))
        frame.iloc[-1, frame.columns.get_loc("is_closed")] = False
        self.assertFalse(FuturesScalpingStrategyV3.frame_is_closed(frame, "15m"))


class BinanceCandleCacheTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.stream = BinanceKlineStream(FakeFeed(), lambda market, event: self.events.append((market, event)))

    def test_snapshot_requires_all_closed_context_frames(self):
        for interval in ("5m", "15m", "1h"):
            frame = candle_frame("5min" if interval == "5m" else "15min" if interval == "15m" else "1h")
            frame.attrs["timeframe"] = interval
            self.stream._merge_frame("futures", "BTCUSDT", interval, frame)
        self.assertIsNotNone(self.stream._snapshot("futures", "BTCUSDT"))

        open_frame = candle_frame("5min")
        open_frame.attrs["timeframe"] = "5m"
        open_frame.iloc[-1, open_frame.columns.get_loc("is_closed")] = False
        self.stream._merge_frame("futures", "BTCUSDT", "5m", open_frame)
        self.assertIsNone(self.stream._snapshot("futures", "BTCUSDT"))

    def test_first_update_only_counts_near_real_candle_open(self):
        opening = 1_800_000_000_000
        self.stream._last_open[("futures", "BTCUSDT")] = opening - 300_000
        base = {
            "e": "kline", "s": "BTCUSDT", "E": opening + 2_000,
            "k": {"i": "5m", "t": opening, "T": opening + 299_999,
                  "o": "100", "h": "101", "l": "99", "c": "100.5",
                  "v": "10", "x": False},
        }
        self.stream._handle_message("futures", 0, 1, FakeApp(), json.dumps(base))
        self.assertEqual(len(self.events), 1)
        self.assertTrue(self.events[0][1]["first_update"])

        self.events.clear()
        self.stream._last_open[("futures", "BTCUSDT")] = opening - 300_000
        base["E"] = opening + 15_000
        self.stream._handle_message("futures", 0, 1, FakeApp(), json.dumps(base))
        self.assertEqual(self.events, [])

    def test_closed_candle_is_delivered_once(self):
        opening = 1_800_000_000_000
        self.stream._last_open[("futures", "BTCUSDT")] = opening
        packet = {
            "e": "kline", "s": "BTCUSDT", "E": opening + 300_001,
            "k": {"i": "5m", "t": opening, "T": opening + 299_999,
                  "o": "100", "h": "101", "l": "99", "c": "100.5",
                  "v": "10", "q": "1000", "n": 20, "V": "5", "Q": "500",
                  "x": True},
        }
        app = FakeApp()
        self.stream._handle_message("futures", 0, 1, app, json.dumps(packet))
        self.stream._handle_message("futures", 0, 1, app, json.dumps(packet))
        self.assertEqual(len(self.events), 1)
        self.assertTrue(self.events[0][1]["is_closed"])
        self.assertFalse(self.events[0][1]["cache_ready"])

    def test_subscription_ack_controls_connected_status(self):
        self.stream._status["futures"]["total_chunks"] = 1
        self.stream._handle_message("futures", 0, 1, FakeApp(), json.dumps({"result": None, "id": 1}))
        self.assertTrue(self.stream.status()["futures"]["connected"])


class ManualEntryQueueTests(unittest.TestCase):
    def test_stale_agent_signal_is_rejected(self):
        now = 1_800_000_000
        with tempfile.TemporaryDirectory() as directory:
            autopilot = Autopilot(
                FakeCoordinator(), config_path=directory + "/autopilot.json",
                clock=lambda: now,
            )
            signal = {
                "symbol": "BTCUSDT", "direction": "LONG",
                "strategy_version": "v3", "signal_generated_at_ms": (now - 60) * 1000,
                "trade_plan": {"success": True, "strategy_version": "v3", "timeframe": "5m", "direction": "LONG"},
            }
            result = autopilot.queue_manual_signal("futures", "BTCUSDT", signal)
            self.assertFalse(result["success"])
            self.assertIn("stale", result["reason"].lower())
            autopilot._event_executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
