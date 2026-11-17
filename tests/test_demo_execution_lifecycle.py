import tempfile
import unittest
from pathlib import Path
from decimal import Decimal

from core.demo.database import DemoDatabase
from core.demo.engine import DemoEngine
from core.demo.ledger import Ledger


class DemoExecutionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = DemoDatabase(Path(self.temp.name) / "demo.db")
        self.ledger = Ledger(database=self.database)
        self.engine = DemoEngine(self.database, self.ledger)
        self.engine.create_account("test", Decimal("1000"))

    def tearDown(self):
        self.database.close()
        self.temp.cleanup()

    def test_manual_futures_close_is_archived_with_manual_reason(self):
        opened = self.engine.open_futures_position(
            "test", agent_id="futures", symbol="BTCUSDT",
            direction="LONG", quantity=Decimal("0.01"),
            market_price=Decimal("50000"), leverage=Decimal("2"),
            reference_id="manual-close-test",
        )
        position_id = opened["position_id"]

        partial = self.engine.close_futures_position(
            "test", position_id, market_price=Decimal("51000"),
            quantity=Decimal("0.004"), reason="MANUAL",
        )
        self.assertEqual(partial["status"], "PARTIALLY_CLOSED")
        self.assertEqual(len(self.engine.get_open_positions("test")), 1)

        final = self.engine.close_futures_position(
            "test", position_id, market_price=Decimal("51200"),
            reason="MANUAL",
        )
        self.assertEqual(final["status"], "CLOSED")
        self.assertEqual(final["remaining_quantity"], Decimal("0"))
        self.assertEqual(self.engine.get_open_positions("test"), [])

        events = self.database.connection.execute(
            "SELECT event_type FROM position_events "
            "WHERE position_id = ? ORDER BY rowid",
            (position_id,),
        ).fetchall()
        self.assertEqual(
            [row["event_type"] for row in events],
            ["OPEN", "MANUAL", "MANUAL"],
        )

    def test_alpha_execution_fails_closed(self):
        from core.demo.models import MarketType, Side

        with self.assertRaisesRegex(ValueError, "ALPHA execution is not enabled"):
            self.engine.submit_order(
                "test", agent_id="alpha", symbol="ABCUSDT",
                market_type=MarketType.ALPHA, side=Side.BUY,
                quantity=Decimal("1"), market_price=Decimal("1"),
            )


if __name__ == "__main__":
    unittest.main()
