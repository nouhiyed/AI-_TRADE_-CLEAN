import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

from api.main import build_cycle_coverage
from core.autopilot import Autopilot
from core.risk_manager import RiskManager


class FunnelCoverageTests(unittest.TestCase):
    def test_cycle_completes_at_95_percent_and_reports_failures_separately(self):
        universe = {"S%03dUSDT" % i for i in range(100)}
        completed = set(sorted(universe)[:96])
        accepted = set(completed)
        coverage = build_cycle_coverage(
            universe, accepted, completed, set(), set(), set(),
            analysis_errors=1, cycle_open_time_ms=300000,
        )
        self.assertTrue(coverage["complete"])
        self.assertEqual(coverage["successful_symbols"], 95)
        self.assertEqual(coverage["failed_symbols"], 1)
        self.assertEqual(coverage["completion_threshold_percent"], 95)
        self.assertEqual(coverage["pending_reconcile_symbols"], 4)

    def test_cycle_stays_open_below_95_percent_success(self):
        universe = {"S%03dUSDT" % i for i in range(100)}
        completed = set(sorted(universe)[:95])
        coverage = build_cycle_coverage(
            universe, completed, completed, set(), set(), set(),
            analysis_errors=1, cycle_open_time_ms=300000,
        )
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["successful_symbols"], 94)
        self.assertEqual(coverage["failed_symbols"], 1)

    def test_funnel_counts_stages_and_returns_top_five_rejections(self):
        autopilot = object.__new__(Autopilot)
        autopilot._event_lock = threading.RLock()
        from collections import OrderedDict
        autopilot._funnel_cycles = {"futures": OrderedDict(), "spot": OrderedDict()}
        autopilot.clock = lambda: 123.0
        for stage in ("events_received", "filter_passed",
                      "strategy_accepted", "valid_plan",
                      "risk_approved", "trade_opened"):
            autopilot._record_funnel("spot", 300000, stage)
        for index in range(7):
            autopilot._record_funnel("spot", 300000, reason="reject-%d" % index)
        result = autopilot.funnel_status()
        self.assertEqual(len(result["markets"]["spot"]), 1)
        cycle = result["markets"]["spot"][0]
        self.assertEqual(cycle["events_received"], 1)
        self.assertEqual(cycle["filter_passed"], 1)
        self.assertEqual(cycle["strategy_accepted"], 1)
        self.assertEqual(cycle["valid_plan"], 1)
        self.assertEqual(cycle["risk_approved"], 1)
        self.assertEqual(cycle["trade_opened"], 1)
        self.assertEqual(len(cycle["top_rejection_reasons"]), 5)


class RiskSizeCapTests(unittest.TestCase):
    def make_risk_manager(self, directory):
        risk = RiskManager.__new__(RiskManager)
        risk.lock = threading.RLock()
        risk.state_file = Path(directory) / "risk_state.json"
        risk.balance = 1000.0
        risk.equity = 1000.0
        risk.max_daily_loss_pct = 3.0
        risk.max_total_risk_pct = 2.0
        risk.daily_loss = 0.0
        risk.reservations = {}
        risk.domains = {
            "futures": {"enabled": True},
            "spot": {"enabled": True},
        }
        risk.day = datetime.now(timezone.utc).date()
        risk.trade_counter = 0
        return risk

    def test_futures_size_is_capped_at_25_percent_equity_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            risk = self.make_risk_manager(directory)
            result = risk.reserve_trade(
                "futures", "BTCUSDT", "LONG",
                entry=100.0, stop_loss=99.9,
                risk_pct=0.5, leverage=5,
            )
            self.assertTrue(result["approved"])
            self.assertTrue(result["size_capped"])
            self.assertLessEqual(result["margin"], 250.0)
            self.assertAlmostEqual(
                result["risk"],
                result["notional"] * abs(100.0 - 99.9) / 100.0,
            )
            self.assertLess(result["risk"], result["requested_risk"])

    def test_spot_size_is_capped_without_breaching_risk_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            risk = self.make_risk_manager(directory)
            result = risk.reserve_trade(
                "spot", "BTCUSDT", "LONG",
                entry=100.0, stop_loss=99.8,
                risk_pct=0.5, leverage=1,
            )
            self.assertTrue(result["approved"])
            self.assertTrue(result["size_capped"])
            self.assertLessEqual(result["margin"], 250.0)
            self.assertLessEqual(result["risk"], 5.0)
            self.assertLessEqual(
                risk.get_reserved_risk(),
                risk.equity * risk.max_total_risk_pct / 100.0,
            )


if __name__ == "__main__":
    unittest.main()
