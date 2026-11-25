import unittest
from core.research_outcomes import evaluate_forward_path, classify_market_regime


class ResearchOutcomeTests(unittest.TestCase):
    def test_long_target_after_entry(self):
        out = evaluate_forward_path("LONG", 100, 98, 104,
                                    [{"high": 103, "low": 99},
                                     {"high": 105, "low": 101}],
                                    fee_rate=0, slippage_rate=0)
        self.assertEqual(out.status, "TAKE_PROFIT")
        self.assertEqual(out.result_r, 2.0)
        self.assertEqual(out.bars, 2)

    def test_same_bar_stop_and_target_is_conservative(self):
        out = evaluate_forward_path("LONG", 100, 98, 104,
                                    [{"high": 105, "low": 97}],
                                    fee_rate=0, slippage_rate=0)
        self.assertEqual(out.status, "STOP_LOSS")
        self.assertEqual(out.result_r, -1.0)

    def test_short_stop(self):
        out = evaluate_forward_path("SHORT", 100, 102, 96,
                                    [{"high": 103, "low": 95}],
                                    fee_rate=0, slippage_rate=0)
        self.assertEqual(out.status, "STOP_LOSS")

    def test_regime_classification(self):
        self.assertEqual(classify_market_regime({"adx_15m": 30,
                           "ema20_slope_1h": 1}), "TREND_UP")
        self.assertEqual(classify_market_regime({"adx_15m": 12}), "RANGE")


    def test_buy_alias_and_entry_slippage_are_accounted_for(self):
        out = evaluate_forward_path(
            "BUY", 100, 98, 104, [{"high": 105, "low": 99}],
            fee_rate=0, slippage_rate=0.001,
        )
        self.assertEqual(out.status, "TAKE_PROFIT")
        self.assertLess(out.result_r, 2.0)


if __name__ == "__main__":
    unittest.main()
