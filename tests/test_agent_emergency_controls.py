import os
import tempfile
import unittest
from pathlib import Path

from core.risk_manager import RiskManager


class AgentEmergencyControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous_cwd = os.getcwd()
        os.chdir(self.temp.name)

    def tearDown(self):
        os.chdir(self.previous_cwd)
        self.temp.cleanup()

    def test_emergency_stop_blocks_new_risk_without_releasing_existing_position(self):
        risk = RiskManager(initial_balance=1000.0)
        existing = risk.reserve_trade(
            domain="futures",
            symbol="BTCUSDT",
            direction="LONG",
            entry=100.0,
            stop_loss=99.0,
            risk_pct=0.5,
            leverage=5,
        )
        self.assertTrue(existing["approved"])
        trade_id = existing["trade_id"]
        reserved_before = risk.get_reserved_risk()

        risk.emergency_stop()

        self.assertFalse(risk.get_status()["domains"]["futures"]["enabled"])
        self.assertFalse(risk.get_status()["domains"]["spot"]["enabled"])
        self.assertIn(trade_id, risk.reservations)
        self.assertEqual(risk.get_reserved_risk(), reserved_before)

        blocked = risk.reserve_trade(
            domain="spot",
            symbol="ETHUSDT",
            direction="LONG",
            entry=100.0,
            stop_loss=99.0,
            risk_pct=0.5,
            leverage=1,
        )
        self.assertFalse(blocked["approved"])
        self.assertEqual(blocked["reason"], "Domain disabled")

        risk.set_domain("futures", True)
        status = risk.get_status()["domains"]
        self.assertTrue(status["futures"]["enabled"])
        self.assertFalse(status["spot"]["enabled"])
        self.assertIn(trade_id, risk.reservations)
        self.assertEqual(risk.get_reserved_risk(), reserved_before)


if __name__ == "__main__":
    unittest.main()
