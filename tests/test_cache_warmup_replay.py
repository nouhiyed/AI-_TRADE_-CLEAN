import tempfile
import time
import unittest
from unittest.mock import patch

import pandas as pd

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
        open_time = (now // 300000) * 300000 - 600000
        event = {
            "symbol": "BTCUSDT", "interval": "5m",
            "open_time": open_time, "close_time": open_time + 299999,
            "event_time": open_time + 300001, "is_closed": True,
            "market_data": None, "cache_ready": False,
        }

        self.autopilot.on_market_kline("futures", event)

        self.assertIn(("futures", "BTCUSDT"), self.autopilot._waiting_cache_events)
        self.assertNotIn(("futures", "BTCUSDT"), self.autopilot._analysis_pending)
        self.assertEqual(
            self.autopilot.seen_symbols_for_cycle("futures", open_time, ["BTCUSDT"]),
            {"BTCUSDT"},
        )
        self.assertEqual(
            self.autopilot.waiting_cache_symbols_for_cycle(
                "futures", open_time, ["BTCUSDT"]
            ),
            {"BTCUSDT"},
        )
        status = self.autopilot.symbol_auditor_status()["futures"]
        self.assertEqual(status["events_received"], 1)
        self.assertEqual(status["symbols_waiting_for_cache"], 1)

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
        self.assertEqual(
            self.autopilot.seen_symbols_for_cycle("futures", open_time, ["ETHUSDT"]),
            {"ETHUSDT"},
        )
        self.assertEqual(
            self.autopilot.waiting_cache_symbols_for_cycle(
                "futures", open_time, ["ETHUSDT"]
            ),
            set(),
        )
        queued = self.autopilot._analysis_pending[("futures", "ETHUSDT")]
        self.assertEqual(queued["market_data"], frames)
        self.assertTrue(queued["cache_ready"])
        self.assertEqual(
            self.autopilot.symbol_auditor_status()["futures"]["events_received"], 1
        )

    def test_delayed_cache_replay_over_ten_minutes_is_retained_audit_only(self):
        old_open = (int(time.time() * 1000) // 300000) * 300000 - 1500000
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
        self.assertEqual(
            self.autopilot.symbol_auditor_status()["spot"]["stale"], 0
        )

    def test_replay_over_ten_minutes_completes_audit_only_without_strategy(self):
        now_ms = int(time.time() * 1000)
        old_open = (now_ms // 300000) * 300000 - 1500000
        event = {
            "symbol": "SOLUSDT", "interval": "5m",
            "open_time": old_open, "close_time": old_open + 299999,
            "event_time": old_open + 300000, "is_closed": True,
            "market_data": {"5m": object()}, "cache_ready": True,
            "audit_only": True,
        }

        self.autopilot._analyze_ws_close(
            "spot", "SOLUSDT", event, received_at_ms=now_ms
        )

        record = self.autopilot._symbol_audits["spot"]["SOLUSDT"]
        self.assertEqual(record["state"], "audit_only_delayed")
        self.assertFalse(record["analysis_completed_error"])
        self.assertEqual(record["analysis_completed_open_time_ms"], old_open)
        self.assertNotIn(("spot", "SOLUSDT"), self.autopilot._pending_entries)


    def test_optional_spot_4h_does_not_block_5m_cache_readiness(self):
        stream = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None,
        )
        stream.feed.fetch_candles = lambda *_args, **_kwargs: object()
        stream._merge_frame = lambda *_args, **_kwargs: None
        stream._rebuild_derived_frames = lambda *_args, **_kwargs: None
        snapshot = {"5m": object()}
        with patch.object(stream, "_snapshot", return_value=snapshot):
            with patch.object(stream, "_refresh_spot_4h_if_due") as refresh:
                self.assertTrue(stream._warm_symbol("spot", "BTCUSDT"))
                refresh.assert_not_called()

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


    def test_missing_exact_cycle_is_recovered_from_rest_once(self):
        stream = BinanceKlineStream(FakeFeed(), lambda _market, _event: None)
        now_ms = int(time.time() * 1000)
        expected_open = (now_ms // 300000) * 300000 - 300000
        previous_open = expected_open - 300000

        def rows(opens):
            return pd.DataFrame([
                {
                    "open_time": pd.to_datetime(open_time, unit="ms", utc=True),
                    "close_time": pd.to_datetime(open_time + 299999, unit="ms", utc=True),
                    "open": 10.0, "high": 11.0, "low": 9.0,
                    "close": 10.5, "volume": 100.0, "is_closed": True,
                }
                for open_time in opens
            ])

        old_frames = {"5m": rows([previous_open])}
        ready_frames = {"5m": rows([previous_open, expected_open])}
        stream._market_symbols["futures"] = {"BTCUSDT"}
        stream.feed.fetch_candles = lambda *args, **kwargs: ready_frames["5m"]
        stream._merge_frame = lambda *_args, **_kwargs: None

        with patch.object(
            stream, "_snapshot", side_effect=[old_frames, ready_frames]
        ):
            result = stream.latest_closed_events(
                "futures", expected_open, ["BTCUSDT"]
            )

        self.assertEqual(result["available"], 1)
        self.assertEqual(result["missing_symbols"], [])
        self.assertEqual(result["events"][0]["open_time"], expected_open)
        status = stream.status()["futures"]
        self.assertEqual(status["cache_reconcile_rest_attempts"], 1)
        self.assertEqual(status["cache_reconcile_rest_successes"], 1)

    def test_missing_exact_cycle_rest_recovery_is_cooldown_limited(self):
        stream = BinanceKlineStream(FakeFeed(), lambda _market, _event: None)
        now_ms = int(time.time() * 1000)
        expected_open = (now_ms // 300000) * 300000 - 300000
        old_frames = {"5m": pd.DataFrame([{
            "open_time": pd.to_datetime(expected_open - 300000, unit="ms", utc=True),
            "close_time": pd.to_datetime(expected_open - 1, unit="ms", utc=True),
            "open": 10.0, "high": 11.0, "low": 9.0,
            "close": 10.5, "volume": 100.0, "is_closed": True,
        }])}
        calls = []
        stream._market_symbols["futures"] = {"BTCUSDT"}

        def fail_fetch(*args, **kwargs):
            calls.append((args, kwargs))
            raise RuntimeError("temporary REST failure")

        stream.feed.fetch_candles = fail_fetch
        with patch.object(stream, "_snapshot", return_value=old_frames):
            first = stream.latest_closed_events("futures", expected_open, ["BTCUSDT"])
            second = stream.latest_closed_events("futures", expected_open, ["BTCUSDT"])

        self.assertEqual(first["missing_symbols"], ["BTCUSDT"])
        self.assertEqual(second["missing_symbols"], ["BTCUSDT"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            stream.status()["futures"]["cache_reconcile_rest_attempts"], 1
        )

    def test_dispatch_race_rechecks_cache_after_waiting_event_is_retained(self):
        now = int(time.time() * 1000)
        open_time = (now // 300000) * 300000 - 300000
        event = {
            "symbol": "BTCUSDT", "interval": "5m",
            "open_time": open_time, "close_time": open_time + 299999,
            "event_time": open_time + 300001, "is_closed": True,
            "market_data": None, "cache_ready": False,
        }
        frames = {"5m": object()}
        stream = BinanceKlineStream(
            FakeFeed(),
            self.autopilot.on_market_kline,
            on_cache_ready=self.autopilot.on_symbol_cache_ready,
        )

        # Simulate the initial WebSocket snapshot missing the cache, followed
        # by the warmup worker making it ready just after the event is retained.
        self.autopilot.on_market_kline("futures", event)
        with patch.object(stream, "_snapshot", return_value=frames):
            self.assertTrue(stream._replay_waiting_event_after_dispatch(
                "futures", "BTCUSDT", None
            ))

        self.assertNotIn(
            ("futures", "BTCUSDT"), self.autopilot._waiting_cache_events
        )
        queued = self.autopilot._analysis_pending[("futures", "BTCUSDT")]
        self.assertEqual(queued["market_data"], frames)
        self.assertTrue(queued["cache_ready"])
        self.assertEqual(
            self.autopilot.waiting_cache_symbols_for_cycle(
                "futures", open_time, ["BTCUSDT"]
            ),
            set(),
        )


if __name__ == "__main__":
    unittest.main()
