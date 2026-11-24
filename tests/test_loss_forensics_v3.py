import pandas as pd
from scripts.loss_forensics_v3 import analyze_trade

def test_mfe_mae_and_same_bar_ambiguity():
    trade=pd.Series({"entry_time":"2026-01-01T00:00:00Z","exit_time":"2026-01-01T00:10:00Z",
      "entry_price":100.0,"initial_stop":99.0,"direction":"LONG","symbol":"BTCUSDT",
      "net_R":-1.0,"exit_reason":"STOP_LOSS","period":"OOS"})
    candles=pd.DataFrame([
      {"open_time":pd.Timestamp("2026-01-01T00:00:00Z"),"close_time":pd.Timestamp("2026-01-01T00:04:59Z"),"high":101.2,"low":98.8},
      {"open_time":pd.Timestamp("2026-01-01T00:05:00Z"),"close_time":pd.Timestamp("2026-01-01T00:09:59Z"),"high":100.4,"low":99.5},
    ])
    result=analyze_trade(trade,candles)
    assert result["mfe_R"]==1.2
    assert result["mae_R"]==-1.2
    assert result["same_bar_initial_stop_and_1R"] is True
    assert result["reached_1R"] is True
