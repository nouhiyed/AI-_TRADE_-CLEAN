import unittest
from unittest.mock import patch

from core.coordinator import market_data_timeout_seconds


class MarketDataTimeoutBudgetTests(unittest.TestCase):
    def test_default_timeout_is_twenty_seconds(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(market_data_timeout_seconds(), 20.0)

    def test_timeout_is_clamped_to_safe_bounds(self):
        self.assertEqual(market_data_timeout_seconds(1), 5.0)
        self.assertEqual(market_data_timeout_seconds(240), 30.0)

    def test_invalid_or_non_finite_override_uses_default(self):
        self.assertEqual(market_data_timeout_seconds("invalid"), 20.0)
        self.assertEqual(market_data_timeout_seconds(float("inf")), 20.0)


if __name__ == "__main__":
    unittest.main()
