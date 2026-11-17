import tempfile
import time
import unittest
from unittest.mock import patch

from core.autopilot import Autopilot
from core.binance_websocket import BinanceKlineStream


class FakeFeed:
    pass


class FakeCoordinator:
    mode = "DEMO"
    demo_positions = {}
    learning = type("Learning", (), {"allows": staticmethod(lambda _key: True)})()

    def _agent_enabled(self, _domain):
        return True


class CacheWarmupReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.autopilot = Autopilot(
            FakeCoordinator(),
            config_path=self.temp.name + "/autopilot.json",
        )
        # Keep jobs pending so assertions can inspect exactly what was queued.
        self.autopilot.submit_analysis = lambda *_args: True

    def tearDown(self):
        self.autopilot._event_executor.shutdown(wait=True)
        for pool in self.autopilot._analysis_pools.values():
            pool.shutdown(wait=True)
        self.temp.cleanup()

    def test_closed_candle_is_retained_when_cache_is_not_ready(self):
        now = int(time.time() * 1000)
        open_time = (now // 300000) * 300000 - 300000
        event = {
            "symbol": "BTCUSDT", "interval": "5m",
            "open_time": open_time, "close_time": open_time + 299999,
            "event_time": open_time + 300001, "is_closed": True,
            "market_data": None, "cache_ready": False,
        }

        self.autopilot.on_market_kline("futures", event)

        self.assertIn(("futures", "BTCUSDT"), self.autopilot._waiting_cache_events)
        self.assertNotIn(("futures", "BTCUSDT"), self.autopilot._analysis_pending)
        status = self.autopilot.symbol_auditor_status()["futures"]
        self.assertEqual(status["events_received"], 1)
        self.assertEqual(status["waiting_cache"], 0) if "waiting_cache" in status else None

    def test_cache_ready_replays_retained_close_once_without_double_count(self):
        now = int(time.time() * 1000)
        open_time = (now // 300000) * 300000 - 300000
        event = {
            "symbol": "ETHUSDT", "interval": "5m",
            "open_time": open_time, "close_time": open_time + 299999,
            "event_time": open_time + 300001, "is_closed": True,
            "market_data": None, "cache_ready": False,
        }
        self.autopilot.on_market_kline("futures", event)
        frames = {"5m": object()}

        self.assertTrue(self.autopilot.on_symbol_cache_ready("futures", "ETHUSDT", frames))
        self.assertFalse(self.autopilot.on_symbol_cache_ready("futures", "ETHUSDT", frames))
        queued = self.autopilot._analysis_pending[("futures", "ETHUSDT")]
        self.assertEqual(queued["market_data"], frames)
        self.assertTrue(queued["cache_ready"])
        self.assertEqual(
            self.autopilot.symbol_auditor_status()["futures"]["events_received"], 1
        )

    def test_delayed_cache_replay_is_audit_only_and_cannot_be_traded(self):
        old_open = int(time.time() * 1000) - 120000
        old_open = (old_open // 300000) * 300000
        event = {
            "symbol": "SOLUSDT", "interval": "5m",
            "open_time": old_open, "close_time": old_open + 299999,
            "event_time": old_open + 300001, "is_closed": True,
            "market_data": None, "cache_ready": False,
        }
        self.autopilot.on_market_kline("spot", event)

        self.assertTrue(self.autopilot.on_symbol_cache_ready(
            "spot", "SOLUSDT", {"5m": object()}
        ))
        queued = self.autopilot._analysis_pending[("spot", "SOLUSDT")]
        self.assertTrue(queued["audit_only"])

    def test_stream_calls_auditor_when_symbol_cache_becomes_ready(self):
        calls = []
        stream = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None,
            on_cache_ready=lambda market, symbol, frames: calls.append(
                (market, symbol, frames)
            ),
        )
        stream._active_intervals = lambda _market: []
        snapshot = {"5m": object()}
        with patch.object(stream, "_snapshot", return_value=snapshot):
            self.assertTrue(stream._warm_symbol("futures", "BTCUSDT"))
        self.assertEqual(calls, [("futures", "BTCUSDT", snapshot)])


if __name__ == "__main__":
    unittest.main()
