import json
import tempfile
import time
import threading
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
    delta = pd.Timedelta(interval)
    end = pd.Timestamp.now(tz="UTC").floor(interval) - delta
    index = pd.date_range(end=end, periods=rows, freq=interval)
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



class CacheWarmupWorkerTests(unittest.TestCase):
    def test_reconnect_does_not_start_duplicate_warmup_worker(self):
        class FakeThread:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.started = False

            def is_alive(self):
                return self.started

            def start(self):
                self.started = True

        stream = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None, max_symbols=8
        )
        stream._thread_factory = FakeThread
        try:
            self.assertTrue(stream._ensure_warmup_worker("spot"))
            first = stream._warmup_threads["spot"]
            self.assertFalse(stream._ensure_warmup_worker("spot"))
            self.assertIs(stream._warmup_threads["spot"], first)
            self.assertEqual(
                stream.status()["spot"]["cache_warmup_worker_starts"], 1
            )
            self.assertTrue(
                stream.status()["spot"]["cache_warmup_worker_alive"]
            )
        finally:
            stream.stop()

class BinanceCandleCacheTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.stream = BinanceKlineStream(
            FakeFeed(), lambda market, event: self.events.append((market, event)),
            max_symbols=8,
        )

    def test_explicit_stream_limit_is_honored_and_zero_means_all(self):
        limited = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None, max_symbols=8)
        full = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None, max_symbols=0)
        self.assertEqual(limited.max_symbols, 8)
        self.assertFalse(limited.full_universe)
        self.assertEqual(full.max_symbols, 0)
        self.assertTrue(full.full_universe)
        self.assertEqual(full._active_intervals("futures"), ["5m"])
        self.assertEqual(full._active_intervals("spot"), ["5m"])

    def test_full_universe_snapshot_requires_derived_closed_context(self):
        stream = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None, max_symbols=0)
        frame = candle_frame("5min", rows=800)
        frame.attrs["timeframe"] = "5m"
        stream._merge_frame("futures", "BTCUSDT", "5m", frame)
        snapshot = stream._snapshot("futures", "BTCUSDT")
        self.assertIsNotNone(snapshot)
        self.assertEqual(set(snapshot), {"5m", "15m", "1h"})

    def test_800_cached_5m_bars_build_closed_15m_and_1h_without_rest(self):
        stream = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None, max_symbols=250)
        self.assertEqual(stream.CACHE_LIMITS["futures"]["5m"], 800)
        self.assertEqual(stream._active_intervals("futures"), ["5m"])
        frame = candle_frame("5min", rows=800)
        frame.attrs["timeframe"] = "5m"
        stream._merge_frame("futures", "BTCUSDT", "5m", frame)
        snapshot = stream._snapshot("futures", "BTCUSDT")
        self.assertEqual(set(snapshot), {"5m", "15m", "1h"})
        self.assertGreaterEqual(len(snapshot["15m"]), 60)
        self.assertGreaterEqual(len(snapshot["1h"]), 60)
        self.assertTrue(snapshot["15m"]["is_closed"].all())
        self.assertTrue(snapshot["1h"]["is_closed"].all())
        self.assertEqual(snapshot["15m"].iloc[-1]["close"], frame.iloc[-1]["close"])
        self.assertEqual(snapshot["1h"].iloc[-1]["close"], frame.iloc[-1]["close"])

    def test_spot_keeps_seeded_210_hour_history_and_includes_optional_4h(self):
        stream = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None, max_symbols=30)
        five = candle_frame("5min", rows=800)
        five.attrs["timeframe"] = "5m"
        one_hour = candle_frame("1h", rows=210)
        one_hour.attrs["timeframe"] = "1h"
        four_hour = candle_frame("4h", rows=210)
        four_hour.attrs["timeframe"] = "4h"
        stream._merge_frame("spot", "BTCUSDT", "5m", five)
        stream._merge_frame("spot", "BTCUSDT", "1h", one_hour)
        stream._merge_frame("spot", "BTCUSDT", "4h", four_hour)
        snapshot = stream._snapshot("spot", "BTCUSDT")
        self.assertEqual(set(snapshot), {"5m", "15m", "1h", "4h"})
        self.assertGreaterEqual(len(snapshot["1h"]), 210)

    def test_historical_cache_can_be_ready_when_latest_trade_is_old(self):
        stream = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None, max_symbols=0
        )
        frame = candle_frame("5min", rows=800)
        frame.attrs["timeframe"] = "5m"
        frame["open_time"] -= pd.Timedelta(minutes=5)
        frame["close_time"] -= pd.Timedelta(minutes=5)
        stream._merge_frame("futures", "STALEUSDT", "5m", frame)
        self.assertIsNotNone(stream._snapshot("futures", "STALEUSDT"))
        stream._market_symbols["futures"] = {"STALEUSDT"}
        expected_open = int(
            (pd.Timestamp.now(tz="UTC").floor("5min") - pd.Timedelta(minutes=5))
            .timestamp() * 1000
        )
        batch = stream.latest_closed_events("futures", expected_open)
        self.assertEqual(batch["events"], [])
        self.assertEqual(batch["missing_symbols"], ["STALEUSDT"])

    def test_cache_becoming_ready_after_warmup_replays_retained_close(self):
        callbacks = []
        self.stream.on_cache_ready = (
            lambda market, symbol, frames: callbacks.append((market, symbol, frames))
        )
        self.stream._snapshot = lambda _market, _symbol: {"5m": "healthy-frame"}

        class StopAfterOnePass:
            def is_set(self):
                return False

            def wait(self, _seconds):
                return True

        self.stream._stop = StopAfterOnePass()
        self.stream._warm_market("futures", ["BTCUSDT"])

        self.assertEqual(len(callbacks), 1)
        self.assertEqual(callbacks[0][0:2], ("futures", "BTCUSDT"))
        self.assertEqual(callbacks[0][2], {"5m": "healthy-frame"})

    def test_client_ping_timer_is_disabled_to_avoid_false_pong_timeouts(self):
        self.assertEqual(BinanceKlineStream.CLIENT_PING_INTERVAL_SECONDS, 0)

    def test_spot_uses_official_market_data_only_websocket_endpoint(self):
        self.assertEqual(
            BinanceKlineStream.BASE_URLS["spot"],
            "wss://data-stream.binance.vision:443/ws",
        )

    def test_futures_uses_current_binance_market_websocket_endpoint(self):
        self.assertEqual(
            BinanceKlineStream.BASE_URLS["futures"],
            "wss://fstream.binance.com/market/stream",
        )

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

    def test_spot_cache_does_not_require_optional_4h_history(self):
        for interval, freq, rows in (
            ("5m", "5min", 60), ("15m", "15min", 60), ("1h", "1h", 210),
        ):
            frame = candle_frame(freq, rows=rows)
            frame.attrs["timeframe"] = interval
            self.stream._merge_frame("spot", "NEWUSDT", interval, frame)
        snapshot = self.stream._snapshot("spot", "NEWUSDT")
        self.assertIsNotNone(snapshot)
        self.assertNotIn("4h", snapshot)

    def test_sparse_historical_candles_are_valid_but_not_current_cycle_events(self):
        stream = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None, max_symbols=0
        )
        gap = candle_frame("5min", rows=800)
        gap.attrs["timeframe"] = "5m"
        # Binance can omit no-trade intervals. Keep only actual returned bars;
        # no synthetic rows are inserted to fill the time gap.
        gap = gap.drop(gap.index[-2]).reset_index(drop=True)
        stream._cache[("futures", "ETHUSDT", "5m")] = gap
        self.assertIsNone(
            stream._frame_health_reason("futures", "5m", gap)
        )
        self.assertIsNotNone(stream._snapshot("futures", "ETHUSDT"))

        short = candle_frame("5min", rows=20)
        short.attrs["timeframe"] = "5m"
        self.assertEqual(
            stream._frame_health_reason("futures", "5m", short),
            "insufficient_rows",
        )

        stale = candle_frame("5min", rows=800)
        stale.attrs["timeframe"] = "5m"
        stale["open_time"] -= pd.Timedelta(hours=1)
        stale["close_time"] -= pd.Timedelta(hours=1)
        stream._cache[("futures", "ETHUSDT", "5m")] = stale
        self.assertIsNotNone(stream._snapshot("futures", "ETHUSDT"))
        stream._market_symbols["futures"] = {"ETHUSDT"}
        expected_open = int(
            (pd.Timestamp.now(tz="UTC").floor("5min") - pd.Timedelta(minutes=5))
            .timestamp() * 1000
        )
        batch = stream.latest_closed_events("futures", expected_open)
        self.assertEqual(batch["events"], [])
        self.assertEqual(batch["missing_symbols"], ["ETHUSDT"])

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


class V3PrefilterTests(unittest.TestCase):
    def _candidate_frame(self):
        frame = candle_frame("5min", rows=60)
        frame.attrs["timeframe"] = "5m"
        frame["EMA_9"] = 100.0
        frame["EMA_20"] = 100.0
        frame["ATR"] = 1.0
        frame["RSI"] = 55.0
        frame["MACD_hist"] = 0.0
        frame.iloc[-2, frame.columns.get_loc("MACD_hist")] = -0.1
        frame["BB_lower"] = 98.0
        frame["BB_upper"] = 102.0
        frame.iloc[-1, frame.columns.get_loc("open")] = 100.0
        frame.iloc[-1, frame.columns.get_loc("high")] = 101.0
        frame.iloc[-1, frame.columns.get_loc("low")] = 99.0
        frame.iloc[-1, frame.columns.get_loc("close")] = 101.0
        frame.iloc[-1, frame.columns.get_loc("volume")] = 10.0
        return frame

    def test_prefilter_passes_5m_pullback_candidate_for_both_markets(self):
        frame = self._candidate_frame()
        futures = FuturesScalpingStrategyV3.prefilter_5m(frame)
        from core.scalping_v3 import SpotScalpingStrategyV3
        spot = SpotScalpingStrategyV3.prefilter_5m(frame)
        self.assertTrue(futures["candidate"])
        self.assertTrue(spot["candidate"])
        self.assertIn("PULLBACK_LONG", futures["setups"])
        self.assertIn("PULLBACK_LONG", spot["setups"])

    def test_prefilter_never_passes_unclosed_or_wrong_timeframe(self):
        frame = self._candidate_frame()
        frame.iloc[-1, frame.columns.get_loc("is_closed")] = False
        self.assertFalse(FuturesScalpingStrategyV3.prefilter_5m(frame)["candidate"])
        frame.iloc[-1, frame.columns.get_loc("is_closed")] = True
        frame.attrs["timeframe"] = "1m"
        self.assertFalse(FuturesScalpingStrategyV3.prefilter_5m(frame)["candidate"])


class ManualEntryQueueTests(unittest.TestCase):
    def test_agent_signal_older_than_90_seconds_is_rejected(self):
        now = 1_800_000_000
        with tempfile.TemporaryDirectory() as directory:
            autopilot = Autopilot(
                FakeCoordinator(), config_path=directory + "/autopilot.json",
                clock=lambda: now,
            )
            signal = {
                "symbol": "BTCUSDT", "direction": "LONG",
                "strategy_version": "v3", "signal_generated_at_ms": (now - 91) * 1000,
                "trade_plan": {"success": True, "strategy_version": "v3", "timeframe": "5m", "direction": "LONG"},
            }
            result = autopilot.queue_manual_signal("futures", "BTCUSDT", signal)
            self.assertFalse(result["success"])
            self.assertIn("90-second", result["reason"].lower())
            autopilot._event_executor.shutdown(wait=True)

    def test_agent_signal_within_90_seconds_is_accepted(self):
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
            self.assertTrue(result["success"])
            self.assertTrue(result["queued"])
            autopilot._event_executor.shutdown(wait=True)


class ImmediateBookTickerEntryTests(unittest.TestCase):
    class Coordinator(FakeCoordinator):
        def __init__(self, quote):
            self.market_feed = type(
                "Feed", (), {"fetch_book_ticker": staticmethod(
                    lambda _market, _symbol: dict(quote)
                )}
            )()
            self.demo_positions = {}
            self.last_signals = {}
            self.open_request = None

        def select_futures_leverage(self, _signal, _plan):
            return 1

        def request_demo_futures_trade(self, *args, **kwargs):
            self.open_request = {"args": args, "kwargs": kwargs}
            return {
                "approved": True, "trade_id": "demo-test-1",
                "entry_price": kwargs.get("entry_reference_price"),
            }

    def test_approved_signal_executes_at_current_ask_without_waiting_for_next_candle(self):
        coordinator = self.Coordinator({"bid": "100.05", "ask": "100.10"})
        with tempfile.TemporaryDirectory() as directory:
            autopilot = Autopilot(
                coordinator, config_path=directory + "/autopilot.json"
            )
            autopilot.update({
                "enabled": True, "trade_futures": True,
                "min_confidence": 60, "risk_pct": 0.5,
            }, persist=False)
            now_ms = int(time.time() * 1000)
            candle_open = (now_ms // 300000) * 300000
            signal = {
                "symbol": "BTCUSDT", "direction": "LONG",
                "strategy_version": "v3", "confidence": 90,
                "signal_generated_at_ms": now_ms,
                "signal_candle_open_time": candle_open,
                "signal_candle_close_time": candle_open + 299999,
                "trade_plan": {
                    "success": True, "strategy_version": "v3",
                    "timeframe": "5m", "direction": "LONG",
                    "entry": 100.0, "stop_loss": 99.0,
                    "take_profit": 102.0, "atr": 1.0,
                    "reward_risk": 2.0, "setup": "PULLBACK_LONG",
                },
            }
            autopilot._execute_ws_entry(
                "futures", "BTCUSDT",
                {"signal": signal, "immediate": True}, None,
            )
            result = autopilot._last_execution_result
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(result["phase"], "opened")
            self.assertEqual(
                coordinator.open_request["kwargs"]["entry_reference_price"],
                100.10,
            )
            executed_signal = coordinator.open_request["kwargs"]["signal_override"]
            self.assertEqual(executed_signal["entry_price_source"], "BINANCE_BOOK_TICKER")
            self.assertAlmostEqual(executed_signal["entry_deviation_pct"], 0.1, places=5)
            autopilot._event_executor.shutdown(wait=True)
            for pool in autopilot._analysis_pools.values():
                pool.shutdown(wait=True)

    def test_short_signal_uses_current_bid(self):
        coordinator = self.Coordinator({"bid": "99.90", "ask": "100.00"})
        with tempfile.TemporaryDirectory() as directory:
            autopilot = Autopilot(
                coordinator, config_path=directory + "/autopilot.json"
            )
            autopilot.update({
                "enabled": True, "trade_futures": True,
                "min_confidence": 60,
            }, persist=False)
            now_ms = int(time.time() * 1000)
            candle_open = (now_ms // 300000) * 300000
            signal = {
                "symbol": "BTCUSDT", "direction": "SHORT",
                "strategy_version": "v3", "confidence": 90,
                "signal_generated_at_ms": now_ms,
                "signal_candle_open_time": candle_open,
                "signal_candle_close_time": candle_open + 299999,
                "trade_plan": {
                    "success": True, "strategy_version": "v3",
                    "timeframe": "5m", "direction": "SHORT",
                    "entry": 100.0, "stop_loss": 101.0,
                    "take_profit": 98.0, "atr": 1.0,
                    "reward_risk": 2.0, "setup": "PULLBACK_SHORT",
                },
            }
            autopilot._execute_ws_entry(
                "futures", "BTCUSDT",
                {"signal": signal, "immediate": True}, None,
            )
            result = autopilot._last_execution_result
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(
                coordinator.open_request["kwargs"]["entry_reference_price"],
                99.90,
            )
            autopilot._event_executor.shutdown(wait=True)
            for pool in autopilot._analysis_pools.values():
                pool.shutdown(wait=True)


class LatestEventQueueTests(unittest.TestCase):
    def test_only_latest_pending_candle_is_kept_per_symbol(self):
        with tempfile.TemporaryDirectory() as directory:
            autopilot = Autopilot(
                FakeCoordinator(), config_path=directory + "/autopilot.json"
            )
            # Keep the drain queued so two incoming closes can be coalesced.
            autopilot.submit_analysis = lambda *_args: True
            first = {
                "symbol": "BTCUSDT", "interval": "5m",
                "open_time": 300000, "close_time": 599999, "is_closed": True,
            }
            second = {
                "symbol": "BTCUSDT", "interval": "5m",
                "open_time": 600000, "close_time": 899999, "is_closed": True,
            }
            self.assertTrue(autopilot._queue_latest_analysis("futures", "BTCUSDT", first))
            self.assertTrue(autopilot._queue_latest_analysis("futures", "BTCUSDT", second))
            pending = autopilot._analysis_pending[("futures", "BTCUSDT")]
            self.assertEqual(pending["close_time"], 899999)
            stats = autopilot.status()["analysis_pools"]["futures"]
            self.assertEqual(stats["superseded"], 1)
            self.assertEqual(stats["queue_depth"], 1)
            self.assertEqual(len(autopilot._analysis_scheduled), 1)
            autopilot._event_executor.shutdown(wait=True)
            autopilot._event_executor.shutdown(wait=True)
            for pool in autopilot._analysis_pools.values():
                pool.shutdown(wait=True)

    def test_per_symbol_queue_keeps_more_than_eight_distinct_coins(self):
        with tempfile.TemporaryDirectory() as directory:
            autopilot = Autopilot(
                FakeCoordinator(), config_path=directory + "/autopilot.json"
            )
            autopilot.submit_analysis = lambda *_args: True
            for index in range(24):
                symbol = "COIN%02dUSDT" % index
                candle = {
                    "symbol": symbol, "interval": "5m",
                    "open_time": 300000, "close_time": 599999,
                    "is_closed": True,
                }
                self.assertTrue(autopilot._queue_latest_analysis(
                    "futures", symbol, candle))
            stats = autopilot.status()["analysis_pools"]["futures"]
            audit = autopilot.status()["symbol_auditors"]["futures"]
            self.assertEqual(stats["queue_depth"], 24)
            self.assertEqual(stats["rejected"], 0)
            self.assertEqual(audit["symbols_tracked"], 24)
            self.assertEqual(audit["events_received"], 24)
            self.assertEqual(audit["queue_rejected"], 0)
            autopilot._event_executor.shutdown(wait=True)
            for pool in autopilot._analysis_pools.values():
                pool.shutdown(wait=True)

    def test_demo_execution_has_one_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            autopilot = Autopilot(
                FakeCoordinator(), config_path=directory + "/autopilot.json"
            )
            self.assertEqual(autopilot._event_executor._max_workers, 1)
            autopilot._event_executor.shutdown(wait=True)
            for pool in autopilot._analysis_pools.values():
                pool.shutdown(wait=True)


class ParallelAnalysisPoolTests(unittest.TestCase):
    def test_futures_and_spot_analysis_run_in_parallel(self):
        with tempfile.TemporaryDirectory() as directory:
            autopilot = Autopilot(
                FakeCoordinator(), config_path=directory + "/autopilot.json"
            )
            barrier = threading.Barrier(2)
            completed = threading.Event()
            results = []
            result_lock = threading.Lock()

            def analyze(domain):
                barrier.wait(timeout=3)
                with result_lock:
                    results.append(domain)
                    if len(results) == 2:
                        completed.set()

            self.assertTrue(autopilot.submit_analysis("futures", analyze, "futures"))
            self.assertTrue(autopilot.submit_analysis("spot", analyze, "spot"))
            self.assertTrue(completed.wait(timeout=4))
            self.assertCountEqual(results, ["futures", "spot"])
            status = autopilot.status()["analysis_pools"]
            self.assertEqual(status["futures"]["completed"], 1)
            self.assertEqual(status["spot"]["completed"], 1)
            autopilot._event_executor.shutdown(wait=True)
            for pool in autopilot._analysis_pools.values():
                pool.shutdown(wait=True)



class AutopilotEntryLifecycleTests(unittest.TestCase):
    def _autopilot(self, directory):
        autopilot = Autopilot(
            FakeCoordinator(), config_path=directory + "/autopilot.json"
        )
        autopilot._submitted_entries = []

        def record_submit(fn, *args):
            autopilot._submitted_entries.append((fn, args))
            return object()

        autopilot._event_executor.submit = record_submit
        return autopilot

    def test_exact_next_5m_open_submits_pending_entry_and_audits_it(self):
        with tempfile.TemporaryDirectory() as directory:
            autopilot = self._autopilot(directory)
            expected_open = 1_800_000_000_000
            pending = {
                "signal": {"symbol": "BTCUSDT", "direction": "LONG"},
                "next_open_time": expected_open,
                "created_at": 1_800_000_000,
            }
            autopilot._pending_entries[("futures", "BTCUSDT")] = pending
            event = {
                "symbol": "BTCUSDT", "interval": "5m",
                "open_time": expected_open, "event_time": expected_open + 100,
                "first_update": True, "is_closed": False,
                "source": "binance_websocket", "open": 100.0,
            }

            autopilot.on_market_kline("futures", event)

            self.assertEqual(len(autopilot._submitted_entries), 1)
            self.assertNotIn(("futures", "BTCUSDT"), autopilot._pending_entries)
            status = autopilot.status()
            self.assertEqual(
                status["symbol_auditors"]["futures"]["entry_submitted"], 1
            )
            self.assertEqual(
                status["recent_entry_audits"][0]["state"], "entry_submitted"
            )
            autopilot._event_executor.shutdown(wait=True)
            for pool in autopilot._analysis_pools.values():
                pool.shutdown(wait=True)

    def test_missed_exact_next_5m_open_expires_entry_with_expected_and_observed_times(self):
        with tempfile.TemporaryDirectory() as directory:
            autopilot = self._autopilot(directory)
            expected_open = 1_800_000_000_000
            observed_open = expected_open + 300_000
            autopilot._pending_entries[("spot", "ETHUSDT")] = {
                "signal": {"symbol": "ETHUSDT", "direction": "BUY"},
                "next_open_time": expected_open,
                "created_at": 1_800_000_000,
            }
            event = {
                "symbol": "ETHUSDT", "interval": "5m",
                "open_time": observed_open, "event_time": observed_open + 100,
                "first_update": True, "is_closed": False,
                "source": "binance_websocket", "open": 2000.0,
            }

            autopilot.on_market_kline("spot", event)

            self.assertNotIn(("spot", "ETHUSDT"), autopilot._pending_entries)
            self.assertEqual(len(autopilot._submitted_entries), 0)
            status = autopilot.status()
            audit = status["recent_entry_audits"][0]
            self.assertEqual(audit["state"], "entry_expired")
            self.assertEqual(audit["expected_open_time_ms"], expected_open)
            self.assertEqual(audit["observed_open_time_ms"], observed_open)
            self.assertEqual(status["symbol_auditors"]["spot"]["entry_expired"], 1)
            autopilot._event_executor.shutdown(wait=True)
            for pool in autopilot._analysis_pools.values():
                pool.shutdown(wait=True)


class FullUniverseCacheReconciliationTests(unittest.TestCase):
    def test_cache_reconciliation_returns_only_exact_expected_closed_candle(self):
        stream = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None, max_symbols=0
        )
        frame = candle_frame("5min", rows=800)
        frame.attrs["timeframe"] = "5m"
        stream._merge_frame("futures", "BTCUSDT", "5m", frame)
        expected_open = int(frame.iloc[-1]["open_time"].timestamp() * 1000)
        stream._market_symbols["futures"] = {"BTCUSDT", "ETHUSDT"}

        batch = stream.latest_closed_events("futures", expected_open)

        self.assertEqual(batch["registered"], 2)
        self.assertEqual(batch["requested"], 2)
        self.assertEqual(batch["available"], 1)
        self.assertEqual(
            [event["symbol"] for event in batch["events"]], ["BTCUSDT"]
        )
        event = batch["events"][0]
        self.assertEqual(event["open_time"], expected_open)
        self.assertTrue(event["is_closed"])
        self.assertIn("audit_only", event)
        self.assertEqual(event["source"], "binance_cache_reconciliation")
        self.assertEqual(batch["missing_symbols"], ["ETHUSDT"])

    def test_cache_reconciliation_finds_expected_cycle_before_newer_cached_row(self):
        stream = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None, max_symbols=0
        )
        frame = candle_frame("5min", rows=801)
        frame.attrs["timeframe"] = "5m"
        stream._merge_frame("futures", "BTCUSDT", "5m", frame)
        expected_open = int(frame.iloc[-2]["open_time"].timestamp() * 1000)
        stream._market_symbols["futures"] = {"BTCUSDT"}

        batch = stream.latest_closed_events("futures", expected_open)

        self.assertEqual(batch["available"], 1)
        self.assertEqual(batch["events"][0]["open_time"], expected_open)
        self.assertEqual(batch["missing_symbols"], [])

    def test_stale_healthy_cache_is_refreshed_once_per_5m_cycle(self):
        class RefreshFeed:
            def __init__(self, frame):
                self.frame = frame
                self.calls = []

            def fetch_candles(self, market, symbol, interval, **kwargs):
                self.calls.append((market, symbol, interval, kwargs))
                return self.frame.copy()

        now_ms = int(time.time() * 1000)
        expected_open = ((now_ms // 300000) - 1) * 300000
        stale = candle_frame("5min", rows=100)
        stale.attrs["timeframe"] = "5m"
        stale["open_time"] -= pd.Timedelta(minutes=5)
        stale["close_time"] -= pd.Timedelta(minutes=5)
        fresh = candle_frame("5min", rows=100)
        fresh.attrs["timeframe"] = "5m"
        feed = RefreshFeed(fresh)
        stream = BinanceKlineStream(
            feed, lambda _market, _event: None, max_symbols=0
        )
        stream._merge_frame("futures", "BTCUSDT", "5m", stale)

        refreshed = stream._refresh_stale_5m(
            "futures", ["BTCUSDT"], now_ms=now_ms
        )
        self.assertEqual(refreshed, ["BTCUSDT"])
        self.assertEqual(len(feed.calls), 1)
        self.assertEqual(
            stream._row_key(
                stream._cache[("futures", "BTCUSDT", "5m")].iloc[-1]["open_time"]
            ),
            expected_open,
        )

        # Repeated warmup passes in the same cycle must not repeat REST calls.
        self.assertEqual(
            stream._refresh_stale_5m("futures", ["BTCUSDT"], now_ms=now_ms),
            [],
        )
        self.assertEqual(len(feed.calls), 1)

    def test_missing_exact_cycle_refreshes_even_with_newer_open_cache_row(self):
        class RefreshFeed:
            def __init__(self, frame):
                self.frame = frame
                self.calls = 0

            def fetch_candles(self, market, symbol, interval, **kwargs):
                self.calls += 1
                return self.frame.copy()

        now_ms = int(time.time() * 1000)
        expected_open = ((now_ms // 300000) - 1) * 300000
        cached = candle_frame("5min", rows=100)
        cached.attrs["timeframe"] = "5m"
        cached = cached.loc[
            cached["open_time"].map(
                lambda value: int(value.timestamp() * 1000)
            ) != expected_open
        ].copy()
        forming = cached.iloc[-1].copy()
        forming["open_time"] = pd.Timestamp(expected_open + 300000, unit="ms", tz="UTC")
        forming["close_time"] = (
            forming["open_time"] + pd.Timedelta(minutes=5)
            - pd.Timedelta(milliseconds=1)
        )
        forming["is_closed"] = False
        cached = pd.concat([cached, forming.to_frame().T], ignore_index=True)
        cached["open_time"] = pd.to_datetime(cached["open_time"], utc=True)
        cached["close_time"] = pd.to_datetime(cached["close_time"], utc=True)
        cached.attrs["timeframe"] = "5m"

        fresh = candle_frame("5min", rows=100)
        fresh.attrs["timeframe"] = "5m"
        feed = RefreshFeed(fresh)
        stream = BinanceKlineStream(
            feed, lambda _market, _event: None, max_symbols=0
        )
        stream._merge_frame("futures", "BTCUSDT", "5m", cached)

        self.assertEqual(
            stream._row_key(
                stream._cache[("futures", "BTCUSDT", "5m")].iloc[-1]["open_time"]
            ),
            expected_open + 300000,
        )
        self.assertEqual(
            stream._refresh_stale_5m(
                "futures", ["BTCUSDT"], now_ms=now_ms
            ),
            ["BTCUSDT"],
        )
        self.assertEqual(feed.calls, 1)
        refreshed = stream._cache[("futures", "BTCUSDT", "5m")]
        self.assertIn(
            expected_open,
            {
                stream._row_key(value)
                for value in refreshed["open_time"]
            },
        )

    def test_successful_but_incomplete_rest_response_is_retried_with_backoff(self):
        class IncompleteThenFreshFeed:
            def __init__(self, incomplete, fresh):
                self.incomplete = incomplete
                self.fresh = fresh
                self.calls = 0
                self.kwargs = []

            def fetch_candles(self, market, symbol, interval, **kwargs):
                self.calls += 1
                self.kwargs.append(kwargs)
                return (self.incomplete if self.calls == 1 else self.fresh).copy()

        now_ms = int(time.time() * 1000)
        expected_open = ((now_ms // 300000) - 1) * 300000
        fresh = candle_frame("5min", rows=100)
        fresh.attrs["timeframe"] = "5m"
        incomplete = fresh.loc[
            fresh["open_time"].map(
                lambda value: int(value.timestamp() * 1000)
            ) != expected_open
        ].copy()
        incomplete.attrs["timeframe"] = "5m"
        feed = IncompleteThenFreshFeed(incomplete, fresh)
        stream = BinanceKlineStream(
            feed, lambda _market, _event: None, max_symbols=0
        )
        stream._merge_frame("futures", "BTCUSDT", "5m", incomplete)

        self.assertEqual(
            stream._refresh_stale_5m("futures", ["BTCUSDT"], now_ms=now_ms),
            [],
        )
        self.assertEqual(feed.calls, 1)
        # A valid HTTP response without the exact candle is not marked done;
        # it is retried after the bounded 30-second cooldown.
        key = ("futures", "BTCUSDT", expected_open)
        self.assertIn(key, stream._last_rest_refresh_cycle)
        self.assertEqual(
            stream._refresh_stale_5m("futures", ["BTCUSDT"], now_ms=now_ms),
            [],
        )
        self.assertEqual(feed.calls, 1)
        stream._last_rest_refresh_attempt_at[key] -= 31.0

        self.assertEqual(
            stream._refresh_stale_5m("futures", ["BTCUSDT"], now_ms=now_ms),
            ["BTCUSDT"],
        )
        self.assertEqual(feed.calls, 2)
        self.assertEqual(feed.kwargs[-1]["start_time"], expected_open - 300000)
        self.assertEqual(feed.kwargs[-1]["end_time"], expected_open + 299999)

    def test_failed_stale_refresh_can_retry_in_same_cycle(self):
        class RetryFeed:
            def __init__(self, frame):
                self.frame = frame
                self.calls = 0

            def fetch_candles(self, market, symbol, interval, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary Binance timeout")
                return self.frame.copy()

        now_ms = int(time.time() * 1000)
        stale = candle_frame("5min", rows=100)
        stale.attrs["timeframe"] = "5m"
        stale["open_time"] -= pd.Timedelta(minutes=5)
        stale["close_time"] -= pd.Timedelta(minutes=5)
        fresh = candle_frame("5min", rows=100)
        fresh.attrs["timeframe"] = "5m"
        feed = RetryFeed(fresh)
        stream = BinanceKlineStream(
            feed, lambda _market, _event: None, max_symbols=0
        )
        stream._merge_frame("futures", "BTCUSDT", "5m", stale)

        self.assertEqual(
            stream._refresh_stale_5m("futures", ["BTCUSDT"], now_ms=now_ms),
            [],
        )
        self.assertEqual(
            stream._refresh_stale_5m("futures", ["BTCUSDT"], now_ms=now_ms),
            [],
        )
        self.assertEqual(feed.calls, 1)
        key = ("futures", "BTCUSDT", ((now_ms // 300000) - 1) * 300000)
        stream._last_rest_refresh_attempt_at[key] -= 31.0
        self.assertEqual(
            stream._refresh_stale_5m("futures", ["BTCUSDT"], now_ms=now_ms),
            ["BTCUSDT"],
        )
        self.assertEqual(feed.calls, 2)

    def test_full_250_symbol_cycle_reaches_the_analysis_queue(self):
        frame = candle_frame("5min", rows=800)
        frame.attrs["timeframe"] = "5m"
        expected_open = int(frame.iloc[-1]["open_time"].timestamp() * 1000)
        symbols = ["COIN%03dUSDT" % index for index in range(250)]
        stream = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None, max_symbols=250
        )
        stream._market_symbols["futures"] = set(symbols)
        for symbol in symbols:
            stream._merge_frame("futures", symbol, "5m", frame)

        batch = stream.latest_closed_events(
            "futures", expected_open, symbols=symbols
        )
        self.assertEqual(batch["requested"], 250)
        self.assertEqual(batch["available"], 250)
        self.assertEqual(batch["missing_symbols"], [])

        with tempfile.TemporaryDirectory() as directory:
            autopilot = Autopilot(
                FakeCoordinator(), config_path=directory + "/autopilot.json"
            )
            try:
                autopilot.register_symbol_universe("futures", symbols)
                # Hold worker submissions so this test can assert the exact
                # 250 distinct cycle events accepted by the bounded queue.
                autopilot.submit_analysis = lambda *_args: True
                for event in batch["events"]:
                    self.assertTrue(autopilot.on_market_kline("futures", event))
                self.assertEqual(
                    sum(autopilot.has_seen_candle(
                        "futures", symbol, expected_open
                    ) for symbol in symbols),
                    250,
                )
                self.assertEqual(
                    len(autopilot._analysis_pending), 250
                )
                self.assertEqual(
                    autopilot.status()["analysis_pools"]["futures"]["rejected"], 0
                )
            finally:
                autopilot._event_executor.shutdown(wait=True)
                for pool in autopilot._analysis_pools.values():
                    pool.shutdown(wait=True)

    def test_cache_reconciliation_rejects_a_different_candle_cycle(self):
        stream = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None, max_symbols=0
        )
        frame = candle_frame("5min", rows=60)
        frame.attrs["timeframe"] = "5m"
        stream._merge_frame("spot", "BTCUSDT", "5m", frame)
        stream._market_symbols["spot"] = {"BTCUSDT"}
        actual_open = int(frame.iloc[-1]["open_time"].timestamp() * 1000)
        batch = stream.latest_closed_events("spot", actual_open + 300000)
        self.assertEqual(batch["events"], [])
        self.assertEqual(batch["missing_symbols"], ["BTCUSDT"])

    def test_queue_rejection_can_be_retried_without_double_counting_event(self):
        with tempfile.TemporaryDirectory() as directory:
            autopilot = Autopilot(
                FakeCoordinator(), config_path=directory + "/autopilot.json"
            )
            attempts = {"count": 0}

            def submit(_domain, _fn, *_args):
                attempts["count"] += 1
                return attempts["count"] > 1

            autopilot.submit_analysis = submit
            event = {
                "symbol": "BTCUSDT", "interval": "5m",
                "open_time": 1800000000000,
                "close_time": 1800000299999,
                "event_time": 1800000300000,
                "is_closed": True, "first_update": False,
                "market_data": {"5m": "healthy"},
            }

            self.assertFalse(autopilot.on_market_kline("futures", event))
            self.assertFalse(autopilot.has_seen_candle(
                "futures", "BTCUSDT", event["open_time"]
            ))
            self.assertTrue(autopilot.on_market_kline("futures", event))
            self.assertTrue(autopilot.has_seen_candle(
                "futures", "BTCUSDT", event["open_time"]
            ))
            audit = autopilot.status()["symbol_auditors"]["futures"]
            self.assertEqual(audit["events_received"], 1)

            autopilot._event_executor.shutdown(wait=True)
            for pool in autopilot._analysis_pools.values():
                pool.shutdown(wait=True)



class LateClosedCandleAuditTests(unittest.TestCase):
    def test_recent_delayed_close_is_audited_without_becoming_stale_or_tradable(self):
        with tempfile.TemporaryDirectory() as directory:
            autopilot = Autopilot(
                FakeCoordinator(), config_path=directory + "/autopilot.json"
            )
            try:
                class FakeStrategy:
                    @staticmethod
                    def prefilter_5m(_frame):
                        return {"candidate": True, "setups": ["test_setup"]}

                fake_agent = type(
                    "FakeAgent", (), {"strategy": FakeStrategy()}
                )()
                autopilot.worker_agent = lambda _domain: fake_agent
                autopilot.c.market_engine = type(
                    "FakeMarketEngine", (),
                    {"_prepare_frame": staticmethod(
                        lambda *_args: "prepared_5m_frame"
                    )},
                )()
                autopilot.c.last_signals = {}
                autopilot.c.analyze_live_from_frames = lambda *_args, **_kwargs: {
                    "signals": {
                        "futures": {
                            "symbol": "BTCUSDT",
                            "direction": "LONG",
                            "strategy_version": "v3",
                            "confidence": 80,
                            "trade_plan": {
                                "success": True,
                                "strategy_version": "v3",
                                "timeframe": "5m",
                                "setup": "test_setup",
                            },
                        }
                    }
                }

                now_ms = int(time.time() * 1000)
                event_time = now_ms - 120_000
                close_time = event_time - 1
                open_time = close_time - 299_999
                candle = {
                    "symbol": "BTCUSDT",
                    "interval": "5m",
                    "open_time": open_time,
                    "close_time": close_time,
                    "event_time": event_time,
                    "is_closed": True,
                    "audit_only": False,
                    "market_data": {"5m": "cached_frame"},
                }

                autopilot._analyze_ws_close(
                    "futures", "BTCUSDT", candle, received_at_ms=now_ms
                )

                audit = autopilot.status()["symbol_auditors"]["futures"]
                self.assertEqual(audit["audit_only"], 1)
                self.assertEqual(audit["stale"], 0)
                self.assertEqual(audit["entry_queued"], 0)
                completion = autopilot.completed_symbols_for_cycle(
                    "futures", open_time, ["BTCUSDT"]
                )
                self.assertEqual(completion["completed"], ["BTCUSDT"])
                self.assertEqual(completion["errors"], [])
            finally:
                autopilot._event_executor.shutdown(wait=True)
                for pool in autopilot._analysis_pools.values():
                    pool.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
