import unittest
from decimal import Decimal

from core.coordinator import TradingCoordinator


class DemoMonitorTriggerTests(unittest.TestCase):
    def test_monitor_prefers_short_deadline_execution_feed(self):
        coordinator = TradingCoordinator.__new__(TradingCoordinator)
        fast_feed = object()
        coordinator.execution_quote_feed = fast_feed
        coordinator.market_feed = object()
        self.assertIs(coordinator._execution_market_feed(), fast_feed)

    def test_monitor_falls_back_to_market_feed_when_fast_feed_is_absent(self):
        coordinator = TradingCoordinator.__new__(TradingCoordinator)
        market_feed = object()
        coordinator.market_feed = market_feed
        self.assertIs(coordinator._execution_market_feed(), market_feed)

    def test_replayed_stop_uses_latest_executable_quote_not_stop_level_as_fill(self):
        trigger = TradingCoordinator._resolve_demo_exit_trigger(
            "LONG",
            Decimal("63.060"),
            Decimal("63.076714"),
            None,
            True,
            missed=("TRAILING_STOP", Decimal("63.076714")),
        )
        self.assertEqual(trigger["reason"], "TRAILING_STOP")
        self.assertEqual(trigger["source"], "ONE_MINUTE_CANDLE_REPLAY")
        self.assertEqual(trigger["trigger_price"], Decimal("63.076714"))
        self.assertEqual(trigger["market_price"], Decimal("63.060"))

    def test_long_trailing_stop_uses_bid_quote_and_marks_trigger_level(self):
        trigger = TradingCoordinator._resolve_demo_exit_trigger(
            "LONG", "63.060", "63.076714", None, True
        )
        self.assertEqual(trigger["reason"], "TRAILING_STOP")
        self.assertEqual(trigger["trigger_price"], Decimal("63.076714"))
        self.assertEqual(trigger["market_price"], Decimal("63.060"))

    def test_short_stop_uses_ask_quote(self):
        trigger = TradingCoordinator._resolve_demo_exit_trigger(
            "SHORT", "101.2", "101", "90", False
        )
        self.assertEqual(trigger["reason"], "STOP_LOSS")
        self.assertEqual(trigger["trigger_price"], Decimal("101"))
        self.assertEqual(trigger["market_price"], Decimal("101.2"))

    def test_no_trigger_when_price_is_between_levels(self):
        trigger = TradingCoordinator._resolve_demo_exit_trigger(
            "LONG", "100", "95", "110", False
        )
        self.assertIsNone(trigger)


if __name__ == "__main__":
    unittest.main()
