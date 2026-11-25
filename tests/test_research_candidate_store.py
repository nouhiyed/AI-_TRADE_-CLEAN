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


    def test_forward_label_uses_only_later_closed_bars(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResearchCandidateStore(Path(tmp) / "research.db")
            candidate = store.record(
                "futures", "BTCUSDT", "standard",
                {"5m": Frame({"open_time": "2026-01-01T00:00:00+00:00"})},
                {"direction": "LONG", "trade_plan": {
                    "success": True, "setup": "TEST", "entry": 100,
                    "stop_loss": 98, "take_profit": 104}},
            )
            # Same candle and an unclosed later candle must not label it.
            same_and_open = [
                {"open_time": "2026-01-01T00:00:00+00:00",
                 "high": 110, "low": 90, "is_closed": True},
                {"open_time": "2026-01-01T00:05:00+00:00",
                 "high": 105, "low": 99, "is_closed": False},
            ]
            self.assertEqual(store.label_pending_from_bars(
                "futures", "BTCUSDT", same_and_open), 0)
            later = [
                {"open_time": "2026-01-01T00:05:00+00:00",
                 "high": 105, "low": 99, "is_closed": True},
            ]
            self.assertEqual(store.label_pending_from_bars(
                "futures", "BTCUSDT", later), 1)
            self.assertEqual(store.outcome_summary("futures")[0]["status"],
                             "TAKE_PROFIT")

    def test_incomplete_horizon_is_not_mislabelled_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResearchCandidateStore(Path(tmp) / "research.db")
            store.record(
                "spot", "ETHUSDT", "standard",
                {"5m": Frame({"open_time": "2026-01-01T00:00:00+00:00"})},
                {"direction": "LONG", "trade_plan": {
                    "success": True, "setup": "TEST", "entry": 100,
                    "stop_loss": 98, "take_profit": 110}},
            )
            later = [{"open_time": "2026-01-01T00:05:00+00:00",
                      "high": 101, "low": 99, "is_closed": True}]
            self.assertEqual(store.label_pending_from_bars(
                "spot", "ETHUSDT", later, horizon=48), 0)
            self.assertEqual(store.outcome_summary("spot"), [])


if __name__ == "__main__":
    unittest.main()
