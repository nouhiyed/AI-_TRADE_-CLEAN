import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research_v3_entry_quality.py"
spec = importlib.util.spec_from_file_location("research_v3_entry_quality", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_safe_ratio_rejects_missing_and_zero_denominator():
    assert np.isnan(module.safe_ratio(None, 2))
    assert np.isnan(module.safe_ratio(2, 0))
    assert module.safe_ratio(6, 3) == 2


def test_metrics_reports_net_and_expectancy():
    frame = pd.DataFrame({"net_R": [1.0, -1.2, 0.5]})
    result = module.metrics(frame)
    assert result["trades"] == 3
    assert result["wins"] == 2
    assert result["net_R"] == 0.3
    assert result["average_R"] == 0.1


def test_directional_features_are_normalized_for_short():
    class Strategy:
        volume_ratio = staticmethod(lambda frame: 1.2)
    idx = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    five = pd.DataFrame({
        "open_time": idx, "open": [100]*5, "high": [102]*5, "low": [98]*5,
        "close": [99]*5, "EMA_9": [100]*5, "EMA_20": [101]*5,
        "ATR": [2]*5, "RSI": [35]*5, "ADX": [22]*5, "MACD_hist": [-0.5]*5,
    })
    fifteen = pd.DataFrame({"ADX":[24]*5,"ATR":[4]*5,"EMA_21":[98]*5,"EMA_50":[100]*5})
    hourly = pd.DataFrame({"ATR":[8]*5,"EMA_21":[97]*5,"EMA_50":[100]*5})
    plan = {"success":True,"direction":"SHORT","setup":"PULLBACK_5M"}
    row = module.capture_features(Strategy(), {"5m":five,"15m":fifteen,"1h":hourly}, plan, "BTCUSDT")
    assert row["rsi_directional"] == 65
    assert row["close_location_directional"] == 0.75
    assert row["ema9_extension_atr"] == -0.5
    assert row["ema21_50_gap_15m_atr"] == 0.5
    assert row["ema21_50_gap_1h_atr"] == 0.375
