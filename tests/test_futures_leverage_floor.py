import unittest

from core.coordinator import TradingCoordinator


class FuturesLeverageFloorTests(unittest.TestCase):
    def test_futures_leverage_is_never_below_five_x(self):
        cases = [
            ({"confidence": 20}, {"entry": 100.0, "atr": 3.0}),
            ({"confidence": 60}, {"entry": 100.0, "atr": 1.0}),
            ({"confidence": 95}, {"entry": 100.0, "atr": 0.2}),
            ({}, {}),
        ]
        for signal, plan in cases:
            with self.subTest(signal=signal, plan=plan):
                self.assertEqual(
                    TradingCoordinator.select_futures_leverage(signal, plan), 5
                )


if __name__ == "__main__":
    unittest.main()
