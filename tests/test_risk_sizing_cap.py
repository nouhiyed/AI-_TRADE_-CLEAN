import tempfile
import unittest
from pathlib import Path

from core.risk_manager import RiskManager


class RiskSizingCapTests(unittest.TestCase):
    def test_tight_stop_is_capped_to_25_percent_wallet_and_actual_risk_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            risk = RiskManager(initial_balance=1000.0)
            risk.state_file = Path(directory) / "risk_state.json"
            risk.balance = 1000.0
            risk.equity = 1000.0
            risk.daily_loss = 0.0
            risk.reservations = {}

            result = risk.reserve_trade(
                domain="spot", symbol="TIGHTUSDT", direction="LONG",
                entry=100.0, stop_loss=99.99, risk_pct=1.0, leverage=1,
            )

            self.assertTrue(result["approved"])
            self.assertTrue(result["sizing_capped"])
            self.assertTrue(result["size_capped"])
            self.assertAlmostEqual(result["margin"], 250.0, places=6)
            self.assertLess(result["risk"], result["requested_risk"])
            self.assertAlmostEqual(result["actual_risk_pct"], 0.0025, places=6)
            self.assertAlmostEqual(
                risk.reservations[result["trade_id"]]["risk"],
                result["risk"], places=8,
            )


if __name__ == "__main__":
    unittest.main()
