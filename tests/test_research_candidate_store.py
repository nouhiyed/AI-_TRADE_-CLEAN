import json
import tempfile
import unittest
from pathlib import Path

from core.research_candidate_store import ResearchCandidateStore
from core.research_outcomes import Outcome


class Frame:
    def __init__(self, row):
        self.row = row
        self.columns = list(row)

    def __len__(self):
        return 1

    @property
    def iloc(self):
        return self

    def __getitem__(self, key):
        if key == -1:
            return self
        raise KeyError(key)

    def to_dict(self):
        return self.row


class ResearchCandidateStoreTests(unittest.TestCase):
    def test_records_and_deduplicates_same_closed_candle(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResearchCandidateStore(Path(tmp) / "research.db")
            frames = {"5m": Frame({"open_time": 123, "close": 100.0,
                                   "volume": 50.0})}
            rejected = {"direction": "WAIT", "strategy_version": "v3",
                        "reason": "No qualifying setup",
                        "diagnostics": {"failed_setups": ["PULLBACK_LONG"]}}
            first = store.record("futures", "BTCUSDT", "explore", frames, rejected)
            second = store.record("futures", "BTCUSDT", "explore", frames, rejected)
            self.assertTrue(first["inserted"])
            self.assertFalse(second["inserted"])
            self.assertEqual(store.summary("futures"),
                             {"total": 1, "accepted": 0, "rejected": 1})

    def test_records_accepted_plan_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResearchCandidateStore(Path(tmp) / "research.db")
            frames = {"5m": Frame({"open_time": 456, "close": 101.0})}
            accepted = {"direction": "LONG", "strategy_version": "v3",
                        "trade_plan": {"success": True, "setup": "PULLBACK_LONG",
                                       "entry": 101, "stop_loss": 99,
                                       "take_profit": 105}}
            result = store.record("futures", "ETHUSDT", "standard", frames, accepted)
            self.assertEqual(result["decision"], "ACCEPTED")
            self.assertEqual(store.summary()["accepted"], 1)

    def test_outcome_label_is_persisted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResearchCandidateStore(Path(tmp) / "research.db")
            frames = {"5m": Frame({"open_time": 789, "close": 100.0})}
            signal = {"direction": "LONG", "trade_plan": {
                "success": True, "setup": "PULLBACK_LONG",
                "entry": 100, "stop_loss": 98, "take_profit": 104}}
            saved = store.record("futures", "BTCUSDT", "explore", frames, signal)
            label = Outcome("TAKE_PROFIT", 3, 104, 1.8, 2.0, -0.2, "TAKE_PROFIT")
            self.assertTrue(store.record_outcome(saved["candidate_id"], label))
            self.assertFalse(store.record_outcome(saved["candidate_id"], label))
            rows = store.outcome_summary("futures")
            self.assertEqual(rows[0]["count"], 1)


if __name__ == "__main__":
    unittest.main()
