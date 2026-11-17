import unittest

import pandas as pd

from core.scalping_v3 import (
    FuturesScalpingStrategyV3,
    SpotScalpingStrategyV3,
)


def closed_5m_frame():
    index = pd.date_range("2026-01-01", periods=60, freq="5min", tz="UTC")
    data = {
        "open_time": index,
        "open": [100.0] * 60,
        "high": [101.0] * 60,
        "low": [99.0] * 60,
        "close": [100.2] * 60,
        "volume": [10.0] * 60,
        "EMA_9": [100.0] * 60,
        "EMA_20": [100.3] * 60,
        "ATR": [1.0] * 60,
        "RSI": [48.0] * 60,
        "MACD_hist": [0.2] * 60,
        "ADX": [20.0] * 60,
        "BB_lower": [98.0] * 60,
        "BB_upper": [102.0] * 60,
        "is_closed": [True] * 60,
    }
    data["MACD_hist"][-2] = 0.1
    data["open"][-1] = 100.0
    data["high"][-1] = 101.0
    data["low"][-1] = 98.0
    data["close"][-1] = 100.8
    data["EMA_9"][-1] = 100.4
    frame = pd.DataFrame(data)
    frame.attrs["timeframe"] = "5m"
    return frame


class V3EntryTimingTests(unittest.TestCase):
    def test_support_reclaim_is_detected_early_by_prefilter(self):
        frame = closed_5m_frame()
        for strategy in (FuturesScalpingStrategyV3(), SpotScalpingStrategyV3()):
            with self.subTest(strategy=strategy.MARKET):
                result = strategy.prefilter_5m(frame)
                self.assertTrue(result["candidate"])
                self.assertIn("SUPPORT_RECLAIM_LONG", result["setups"])

    def test_long_entry_too_far_above_ema9_is_rejected(self):
        frame = closed_5m_frame()
        frame["EMA_9"] = 99.0
        frame["ATR"] = 1.0
        plan = FuturesScalpingStrategyV3().plan_with_structure(
            "futures", "LONG", frame,
            {"5m": frame}, "TEST", "BULL_TREND", 1.6, 101.0,
        )
        self.assertFalse(plan["success"])
        self.assertTrue(plan["entry_timing_guard"]["blocked"])
        self.assertIn("late-entry veto", plan["reason"].lower())

    def test_short_entry_too_far_below_ema9_is_rejected(self):
        frame = closed_5m_frame()
        frame["EMA_9"] = 102.0
        frame["ATR"] = 1.0
        plan = FuturesScalpingStrategyV3().plan_with_structure(
            "futures", "SHORT", frame,
            {"5m": frame}, "TEST", "BEAR_TREND", 1.6, 100.0,
        )
        self.assertFalse(plan["success"])
        self.assertTrue(plan["entry_timing_guard"]["blocked"])


if __name__ == "__main__":
    unittest.main()
