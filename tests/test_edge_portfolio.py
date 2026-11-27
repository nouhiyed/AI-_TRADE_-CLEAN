import unittest
from core.edge_portfolio import non_overlapping_replay
from tests.test_edge_backtest import synthetic_bars

class PortfolioReplayTests(unittest.TestCase):
    def test_replay_is_explicit_and_has_metrics(self):
        result=non_overlapping_replay(synthetic_bars(), market="futures")
        self.assertEqual(result["replay"],"NON_OVERLAP_PER_STRATEGY")
        self.assertIn("portfolio_overall",result)
        self.assertIn("portfolio_by_strategy",result)
        self.assertEqual(set(result["portfolio_by_strategy"]),set(result["by_strategy"]))

if __name__=="__main__":
    unittest.main()
