import tempfile
import unittest

from core.autopilot import Autopilot


class FunnelTelemetryTests(unittest.TestCase):
    def _close(self, autopilot):
        autopilot._event_executor.shutdown(wait=True)
        for pool in autopilot._analysis_pools.values():
            pool.shutdown(wait=True)

    def test_per_candle_stages_and_top_five_rejections_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            autopilot = Autopilot(object(), config_path=directory + "/autopilot.json")
            try:
                candle = 1_800_000_000_000
                for stage in (
                    "events_received", "filter_passed", "strategy_accepted",
                    "valid_plan", "risk_approved", "trade_opened",
                ):
                    autopilot._record_funnel("futures", candle, stage)
                for index in range(7):
                    autopilot._record_funnel(
                        "futures", candle, "rejection", reason="reason-%d" % index
                    )

                snapshot = autopilot.funnel_status()["markets"]["futures"][-1]
                self.assertEqual(snapshot["events_received"], 1)
                self.assertEqual(snapshot["filter_passed"], 1)
                self.assertEqual(snapshot["strategy_accepted"], 1)
                self.assertEqual(snapshot["valid_plan"], 1)
                self.assertEqual(snapshot["risk_approved"], 1)
                self.assertEqual(snapshot["trade_opened"], 1)
                self.assertEqual(len(snapshot["top_rejection_reasons"]), 5)
            finally:
                self._close(autopilot)

    def test_full_rejection_reason_is_not_truncated_and_symbol_detail_is_exposed(self):
        with tempfile.TemporaryDirectory() as directory:
            autopilot = Autopilot(object(), config_path=directory + "/autopilot.json")
            try:
                candle = 1_800_000_300_000
                long_reason = "V3 rejected: " + ("diagnostic-detail-" * 40)
                autopilot._record_funnel(
                    "spot", candle, "rejection", reason=long_reason
                )
                autopilot._record_symbol_audit(
                    "spot", "BTCUSDT", "strategy_rejected",
                    reason=long_reason, setup_checks={"PULLBACK_LONG": False},
                    metrics={"adx": 17.2, "rsi": 48.1},
                    strategy_diagnostics={
                        "setup_checks": {"PULLBACK_LONG": False},
                        "metrics": {"adx": 17.2, "rsi": 48.1},
                    },
                )

                diagnostics = autopilot.strategy_diagnostics("spot", limit=10)
                market = diagnostics["markets"]["spot"]
                self.assertEqual(market["symbols"][0]["reason"], long_reason)
                self.assertFalse(
                    market["symbols"][0]["setup_checks"]["PULLBACK_LONG"]
                )
                self.assertEqual(
                    market["symbols"][0]["strategy_diagnostics"]["metrics"]["adx"],
                    17.2,
                )
                self.assertEqual(
                    market["funnel_cycles"][0]["rejection_reasons"][0]["reason"],
                    long_reason,
                )
                self.assertGreater(
                    len(market["funnel_cycles"][0]["rejection_reasons"][0]["reason"]),
                    240,
                )
            finally:
                self._close(autopilot)


if __name__ == "__main__":
    unittest.main()
