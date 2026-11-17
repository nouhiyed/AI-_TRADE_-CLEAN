import unittest

import pandas as pd

from core.scalping_v3 import FuturesScalpingStrategyV3, SpotScalpingStrategyV3


def frame(timeframe, rows=80, pivot=None, direction="LONG"):
    index = pd.date_range("2026-01-01", periods=rows, freq={
        "5m": "5min", "15m": "15min", "1h": "1h"
    }[timeframe], tz="UTC")
    highs = [100.5] * rows
    lows = [99.5] * rows
    if pivot is not None:
        highs[70] = pivot if direction == "LONG" else 100.5
        lows[70] = pivot if direction == "SHORT" else 99.5
    result = pd.DataFrame({
        "open_time": index,
        "open": [100.0] * rows,
        "high": highs,
        "low": lows,
        "close": [100.0] * rows,
        "volume": [10.0] * rows,
        "is_closed": [True] * rows,
    })
    result.attrs["timeframe"] = timeframe
    return result


class OpposingStructureGuardTests(unittest.TestCase):
    def _frames(self, pivot=None, direction="LONG"):
        return {
            "5m": frame("5m"),
            "15m": frame("15m", pivot=pivot, direction=direction),
            "1h": frame("1h"),
        }

    def _plan(self, strategy, direction, frames):
        five = frame("5m")
        five["ATR"] = 1.0
        return strategy.plan_with_structure(
            strategy.MARKET, direction, five, frames,
            "TEST_SETUP", "BULL_TREND" if direction == "LONG" else "BEAR_TREND",
            1.6, 100.0,
        )

    def test_long_is_vetoed_when_target_runs_into_nearby_resistance(self):
        plan = self._plan(FuturesScalpingStrategyV3(), "LONG",
                          self._frames(101.0, "LONG"))
        self.assertFalse(plan["success"])
        self.assertIn("structure veto", plan["reason"].lower())
        self.assertEqual(plan["structure_guard"]["timeframe"], "15m")

    def test_short_is_vetoed_when_target_runs_into_nearby_support(self):
        plan = self._plan(FuturesScalpingStrategyV3(), "SHORT",
                          self._frames(99.0, "SHORT"))
        self.assertFalse(plan["success"])
        self.assertIn("structure veto", plan["reason"].lower())

    def test_long_with_clearance_beyond_target_remains_eligible(self):
        plan = self._plan(SpotScalpingStrategyV3(), "LONG",
                          self._frames(105.0, "LONG"))
        self.assertTrue(plan["success"])
        self.assertFalse(plan["structure_guard"]["blocked"])
        self.assertEqual(plan["structure_guard"]["timeframe"], "15m")

    def test_no_confirmed_opposing_pivot_is_not_misclassified_as_resistance(self):
        plan = self._plan(SpotScalpingStrategyV3(), "LONG", self._frames())
        self.assertTrue(plan["success"])
        self.assertIsNone(plan["structure_guard"]["level"])


if __name__ == "__main__":
    unittest.main()
