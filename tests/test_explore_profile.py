import unittest

from core.learning import LearningEngine


class FakeJournal:
    def __init__(self, trades):
        self.trades = trades

    def closed_trades(self):
        return list(self.trades)

    def disabled_setups(self):
        return {}

    def set_setup_disabled(self, *args, **kwargs):
        raise AssertionError("Learning must not disable setups without a random baseline")


class ExploreIsolationTests(unittest.TestCase):
    def test_learning_requires_150_standard_trades_and_excludes_explore(self):
        trades = [
            {"status": "CLOSED", "net_pnl": 10, "r_multiple": 1.0,
             "setup_key": "v3|LONG|c65|rr2+", "context": {"profile": "standard"}},
            {"status": "CLOSED", "net_pnl": -100, "r_multiple": -1.0,
             "setup_key": "v3|LONG|c65|rr2+", "context": {"profile": "explore"}},
        ]
        learning = LearningEngine(FakeJournal(trades))
        report = learning.report()

        self.assertEqual(report["min_samples"], 150)
        self.assertEqual(report["closed_trades"], 1)
        self.assertEqual(report["overall"]["trades"], 1)
        self.assertTrue(report["random_baseline_required"])
        self.assertFalse(report["random_baseline_compared"])
        self.assertIsNone(report["profitability_conclusion"])
        self.assertEqual(learning.evaluate(), [])


if __name__ == "__main__":
    unittest.main()
