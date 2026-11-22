import unittest

import pandas as pd

from scripts.research_v3_gate_attribution import gate_sets


class V3GateAttributionTests(unittest.TestCase):
    def setUp(self):
        self.five = pd.DataFrame({
            "open": [99.0] * 29 + [99.8],
            "high": [101.0] * 30,
            "low": [98.0] * 30,
            "close": [100.0] * 30,
            "EMA_9": [99.5] * 30,
            "EMA_20": [100.0] * 30,
            "EMA_21": [99.5] * 30,
            "ATR": [1.0] * 30,
            "RSI": [55.0] * 30,
            "MACD_hist": [0.1] * 29 + [0.2],
            "volume": [100.0] * 30,
            "BB_lower": [97.0] * 30,
            "BB_upper": [103.0] * 30,
        })
        self.fifteen = pd.DataFrame({
            "close": [110.0], "EMA_21": [105.0],
            "EMA_50": [100.0], "ADX": [25.0],
        })
        self.hourly = pd.DataFrame({
            "close": [120.0], "EMA_200": [100.0],
        })

    def test_pullback_gate_mirror_accepts_clean_synthetic_setup(self):
        gates = gate_sets(self.five, self.fifteen, self.hourly)
        self.assertTrue(all(gates["PULLBACK_LONG"].values()))

    def test_breakout_gate_does_not_pass_without_prior_high_break(self):
        gates = gate_sets(self.five, self.fifteen, self.hourly)
        self.assertFalse(gates["BREAKOUT_LONG"]["breakout"])


if __name__ == "__main__":
    unittest.main()
