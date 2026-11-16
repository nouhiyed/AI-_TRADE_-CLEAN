import unittest

import pandas as pd

from core.trailing_stop import (
    active_fixed_take_profit,
    calculate_trailing_stop,
    initialize_trailing_take_profit,
    is_trailing_take_profit_enabled,
)


class TrailingTakeProfitPolicyTests(unittest.TestCase):
    def test_new_positions_are_trailing_and_have_no_fixed_exit_target(self):
        for domain in ("futures", "spot"):
            with self.subTest(domain=domain):
                record = {"domain": domain, "take_profit": "125.0"}
                initialize_trailing_take_profit(record, "100.0")
                self.assertEqual(record["take_profit_mode"], "TRAILING")
                self.assertTrue(record["trailing_take_profit_enabled"])
                self.assertTrue(record["trailing_no_fixed_target"])
                self.assertFalse(record["trailing_take_profit_armed"])
                self.assertFalse(record["trailing_stop_active"])
                self.assertEqual(record["trailing_best_price"], "100.0")
                self.assertTrue(is_trailing_take_profit_enabled(record))
                self.assertIsNone(active_fixed_take_profit(record))

    def test_legacy_position_defaults_to_trailing_not_fixed_take_profit(self):
        record = {"take_profit": "125.0"}
        self.assertTrue(is_trailing_take_profit_enabled(record))
        self.assertIsNone(active_fixed_take_profit(record))

    def test_fixed_target_is_only_available_when_trailing_is_explicitly_disabled(self):
        record = {
            "trailing_take_profit_enabled": False,
            "take_profit": "125.0",
        }
        self.assertFalse(is_trailing_take_profit_enabled(record))
        self.assertEqual(active_fixed_take_profit(record), "125.0")

    def test_atr_structure_trailing_stop_never_moves_away_from_market(self):
        frame = pd.DataFrame({
            "high": [102.0, 103.0, 102.5, 104.0, 105.0, 104.5, 106.0,
                     107.0, 106.5, 108.0, 109.0, 108.5, 110.0, 111.0, 112.0, 113.0],
            "low":  [100.0, 101.0, 100.5, 102.0, 103.0, 102.5, 104.0,
                     105.0, 104.5, 106.0, 107.0, 106.5, 108.0, 109.0, 110.0, 111.0],
            "close":[101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0,
                     106.0, 105.5, 107.0, 108.0, 107.5, 109.0, 110.0, 111.0, 112.0],
        })
        long_update = calculate_trailing_stop(
            frame, "LONG", previous_stop=100.0,
            best_price=105.0, current_price=111.0)
        short_update = calculate_trailing_stop(
            frame, "SHORT", previous_stop=120.0,
            best_price=115.0, current_price=111.0)
        self.assertIsNotNone(long_update)
        self.assertIsNotNone(short_update)
        self.assertGreaterEqual(long_update["stop"], 100.0)
        self.assertLessEqual(short_update["stop"], 120.0)


if __name__ == "__main__":
    unittest.main()
