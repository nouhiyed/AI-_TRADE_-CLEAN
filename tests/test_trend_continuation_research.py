import unittest

import pandas as pd

from scripts.research_trend_continuation import simulate_exit


class TrendContinuationResearchTests(unittest.TestCase):
    def test_same_candle_stop_and_target_assumes_stop_first(self):
        frame = pd.DataFrame([
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
            {"open": 100.0, "high": 103.0, "low": 97.0, "close": 101.0},
            {"open": 101.0, "high": 102.0, "low": 100.0, "close": 101.0},
        ])
        result = simulate_exit(
            frame, signal_index=0, direction="LONG",
            entry=100.0, stop=98.0, target=102.0, market="spot",
        )
        self.assertEqual(result["reason"], "SL")
        self.assertLess(result["net_return_pct"], 0)


if __name__ == "__main__":
    unittest.main()
