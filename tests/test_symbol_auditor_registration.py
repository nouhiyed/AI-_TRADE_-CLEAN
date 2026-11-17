import tempfile
import unittest

from core.autopilot import Autopilot
from core.binance_websocket import BinanceKlineStream


class FakeFeed:
    pass


class FakeCoordinator:
    mode = "DEMO"
    demo_positions = {}
    learning = type("Learning", (), {"allows": staticmethod(lambda _key: True)})()


class FullUniverseAuditorRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.now = 1_800_000_000
        self.autopilot = Autopilot(
            FakeCoordinator(),
            config_path=self.temp.name + "/autopilot.json",
            clock=lambda: self.now,
        )

    def tearDown(self):
        self.autopilot._event_executor.shutdown(wait=True)
        for pool in self.autopilot._analysis_pools.values():
            pool.shutdown(wait=True)
        self.temp.cleanup()

    def test_every_market_symbol_is_registered_before_first_candle_event(self):
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HUMAUSDT"]
        result = self.autopilot.register_symbol_universe("futures", symbols)
        status = self.autopilot.symbol_auditor_status()["futures"]

        self.assertTrue(result["success"])
        self.assertEqual(result["registered"], 4)
        self.assertEqual(status["symbols_registered"], 4)
        self.assertEqual(status["symbols_tracked"], 0)
        self.assertEqual(status["symbols_waiting_for_first_event"], 4)

    def test_status_distinguishes_registered_seen_and_prefiltered_symbols(self):
        self.autopilot.register_symbol_universe(
            "spot", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        self.autopilot._record_symbol_audit(
            "spot", "BTCUSDT", "queued",
            increment="events_received", close_time_ms=1_799_999_999_999,
        )
        self.autopilot._record_symbol_audit(
            "spot", "BTCUSDT", "prefilter_rejected",
            increment="prefilter_rejected",
        )
        status = self.autopilot.symbol_auditor_status()["spot"]

        self.assertEqual(status["symbols_registered"], 3)
        self.assertEqual(status["symbols_seen"], 1)
        self.assertEqual(status["symbols_tracked"], 1)
        self.assertEqual(status["symbols_audited"], 1)
        self.assertEqual(status["symbols_waiting_for_first_event"], 2)
        self.assertEqual(status["symbols_seen_recently"], 1)
        self.assertEqual(status["symbols_audited_recently"], 1)

    def test_stream_provisions_auditors_for_the_complete_market_universe(self):
        captured = []
        stream = BinanceKlineStream(
            FakeFeed(), lambda _market, _event: None,
            on_universe=lambda market, symbols: captured.append((market, symbols)),
        )
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        stream._register_universe("futures", symbols)
        self.assertEqual(captured, [("futures", symbols)])

    def test_universe_refresh_marks_removed_symbols_inactive(self):
        self.autopilot.register_symbol_universe("futures", ["BTCUSDT", "ETHUSDT"])
        self.autopilot.register_symbol_universe("futures", ["BTCUSDT"])
        status = self.autopilot.symbol_auditor_status()["futures"]
        self.assertEqual(status["symbols_registered"], 1)
        self.assertEqual(status["symbols_waiting_for_first_event"], 1)
        self.assertEqual(
            self.autopilot._symbol_audits["futures"]["ETHUSDT"]["state"],
            "not_in_current_universe",
        )


if __name__ == "__main__":
    unittest.main()
