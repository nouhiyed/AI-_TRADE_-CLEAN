import unittest
import pandas as pd

from scripts.backtest_v3_2y import entry_is_valid, resample_ohlcv, summarize


class V3HistoricalBacktestTests(unittest.TestCase):
    def test_entry_validation_matches_demo_drift_and_net_rr_gate(self):
        plan = {
            "entry": 100.0, "stop_loss": 99.0,
            "take_profit": 103.0, "atr": 1.0,
        }
        self.assertTrue(entry_is_valid(plan, "LONG", "spot", 100.0))
        self.assertFalse(entry_is_valid(plan, "LONG", "spot", 100.11))

    def test_resample_requires_complete_higher_timeframe_bar(self):
        frame = pd.DataFrame({
            "open_time": pd.to_datetime([
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:05:00Z",
                "2026-01-01T00:10:00Z",
                "2026-01-01T00:15:00Z",
            ], utc=True),
            "open": [10, 11, 12, 13],
            "high": [12, 13, 14, 15],
            "low": [9, 10, 11, 12],
            "close": [11, 12, 13, 14],
            "volume": [1, 2, 3, 4],
        })
        result = resample_ohlcv(frame, 15)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["open"], 10)
        self.assertEqual(result.iloc[0]["high"], 14)
        self.assertEqual(result.iloc[0]["low"], 9)
        self.assertEqual(result.iloc[0]["close"], 13)
        self.assertEqual(result.iloc[0]["volume"], 6)

    def test_summary_reports_net_r_and_drawdown(self):
        result = summarize([
            {"net_R": 1.0}, {"net_R": -0.5}, {"net_R": 0.25},
        ])
        self.assertEqual(result["trades"], 3)
        self.assertEqual(result["wins"], 2)
        self.assertEqual(result["net_R"], 0.75)


if __name__ == "__main__":
    unittest.main()
