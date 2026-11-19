import tempfile
import unittest

from core.autopilot import Autopilot


class FunnelTelemetryTests(unittest.TestCase):
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
                autopilot._event_executor.shutdown(wait=True)
                for pool in autopilot._analysis_pools.values():
                    pool.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
