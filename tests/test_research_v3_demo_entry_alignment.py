import unittest

from scripts.research_v3_demo_entry_alignment import demo_aligned_entry_is_valid


class DemoEntryAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "entry": 100.0,
            "stop_loss": 98.0,
            "take_profit": 104.0,
            "atr": 1.0,
        }

    def test_uses_demo_030_atr_drift_allowance(self):
        self.assertTrue(demo_aligned_entry_is_valid(
            self.plan, "LONG", "futures", 100.25
        ))

    def test_rejects_drift_above_demo_allowance(self):
        self.assertFalse(demo_aligned_entry_is_valid(
            self.plan, "LONG", "futures", 100.31
        ))

    def test_still_rejects_invalid_reward_risk(self):
        plan = dict(self.plan, take_profit=100.2)
        self.assertFalse(demo_aligned_entry_is_valid(
            plan, "LONG", "futures", 100.0
        ))

    def test_short_geometry(self):
        plan = {
            "entry": 100.0,
            "stop_loss": 102.0,
            "take_profit": 96.0,
            "atr": 1.0,
        }
        self.assertTrue(demo_aligned_entry_is_valid(
            plan, "SHORT", "futures", 100.25
        ))


if __name__ == "__main__":
    unittest.main()
