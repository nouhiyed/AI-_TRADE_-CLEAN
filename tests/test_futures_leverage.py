import unittest

from core.coordinator import TradingCoordinator


class FuturesLeveragePolicyTests(unittest.TestCase):
    def test_futures_leverage_never_downgrades_below_five_x(self):
        cases = (
            ({"confidence": 20}, {"entry": 100.0, "atr": 4.0}),
            ({"confidence": 55}, {"entry": 100.0, "atr": 2.0}),
            ({"confidence": 95}, {"entry": 100.0, "atr": 0.2}),
            ({}, {}),
        )
        for signal, plan in cases:
            with self.subTest(signal=signal, plan=plan):
                self.assertEqual(
                    TradingCoordinator.select_futures_leverage(signal, plan),
                    5,
                )


if __name__ == "__main__":
    unittest.main()
