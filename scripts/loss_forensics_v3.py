#!/usr/bin/env python3
"""Read-only V3 trade forensics: MFE/MAE, stop/1R ordering ambiguity, and timing."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def analyze_trade(trade, candles):
    entry_time = pd.Timestamp(trade.entry_time)
    exit_time = pd.Timestamp(trade.exit_time)
    bars = candles[(candles.open_time >= entry_time) & (candles.open_time < exit_time)]
    # exit_time is the close timestamp of the final holding candle in the replay.
    if bars.empty:
        bars = candles[(candles.open_time >= entry_time) & (candles.close_time <= exit_time)]
    if bars.empty:
        return None
    entry=float(trade.entry_price); stop=float(trade.initial_stop)
    risk=abs(entry-stop); direction=str(trade.direction).upper()
    if risk <= 0: return None
    if direction=="LONG":
        favorable=(bars.high.astype(float)-entry)/risk
        adverse=(bars.low.astype(float)-entry)/risk
        stop_hit=bars.low.astype(float) <= stop
        one_r_hit=bars.high.astype(float) >= entry+risk
    else:
        favorable=(entry-bars.low.astype(float))/risk
        adverse=(entry-bars.high.astype(float))/risk
        stop_hit=bars.high.astype(float) >= stop
        one_r_hit=bars.low.astype(float) <= entry-risk
    both=stop_hit & one_r_hit
    first_1r=int(np.flatnonzero(one_r_hit.to_numpy())[0]) if one_r_hit.any() else None
    first_stop=int(np.flatnonzero(stop_hit.to_numpy())[0]) if stop_hit.any() else None
    result=trade.to_dict()
    result.update({
      "observed_bars":int(len(bars)),
      "mfe_R":round(float(favorable.max()),4),
      "mae_R":round(float(adverse.min()),4),
      "first_1R_bar":first_1r,
      "first_initial_stop_bar":first_stop,
      "same_bar_initial_stop_and_1R":bool(both.any()),
      "same_bar_count":int(both.sum()),
      "reached_0_5R":bool((favorable>=0.5).any()),
      "reached_1R":bool(one_r_hit.any()),
      "reached_1_5R":bool((favorable>=1.5).any()),
      "initial_stop_touched_anytime":bool(stop_hit.any()),
      "period":trade.get("period",""),
    })
    return result

def summarize(df, keys):
    rows=[]
    for key, g in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple): key=(key,)
        row=dict(zip(keys,key))
        row.update({
          "trades":len(g),
          "wins":int((g.net_R>0).sum()),
          "net_R":round(float(g.net_R.sum()),3),
          "avg_MFE_R":round(float(g.mfe_R.mean()),3),
          "median_MFE_R":round(float(g.mfe_R.median()),3),
          "avg_MAE_R":round(float(g.mae_R.mean()),3),
          "median_MAE_R":round(float(g.mae_R.median()),3),
          "reached_1R":int(g.reached_1R.sum()),
          "same_bar_stop_1R":int(g.same_bar_initial_stop_and_1R.sum()),
          "reached_1_5R":int(g.reached_1_5R.sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--trades",required=True)
    p.add_argument("--cache",default="data/backtest_cache")
    p.add_argument("--out",default="data/v3_loss_forensics")
    a=p.parse_args()
    trades=pd.read_csv(a.trades)
    cache=Path(a.cache); results=[]
    for symbol,g in trades.groupby("symbol"):
        path=cache/f"futures_{symbol}_5m.csv.gz"
        if not path.exists(): raise FileNotFoundError(path)
        candles=pd.read_csv(path,parse_dates=["open_time","close_time"])
        candles["open_time"]=pd.to_datetime(candles.open_time,utc=True)
        candles["close_time"]=pd.to_datetime(candles.close_time,utc=True)
        for _,trade in g.iterrows():
            row=analyze_trade(trade,candles)
            if row: results.append(row)
    df=pd.DataFrame(results)
    if len(df)!=len(trades):
        raise RuntimeError(f"Only matched {len(df)} of {len(trades)} trades")
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"trade_forensics.csv",index=False)
    summarize(df,["exit_reason"]).to_csv(out/"by_exit_reason.csv",index=False)
    summarize(df,["setup","direction"]).to_csv(out/"by_setup_direction.csv",index=False)
    summarize(df,["symbol"]).to_csv(out/"by_symbol.csv",index=False)
    summarize(df,["period"]).to_csv(out/"by_period.csv",index=False)
    losses=df[df.net_R<=0]; wins=df[df.net_R>0]
    report={
      "trades_analyzed":len(df),
      "total_net_R":round(float(df.net_R.sum()),3),
      "stop_loss_trades":int((df.exit_reason=="STOP_LOSS").sum()),
      "trailing_stop_trades":int((df.exit_reason=="TRAILING_STOP").sum()),
      "loss_avg_mfe_R":round(float(losses.mfe_R.mean()),3) if len(losses) else None,
      "loss_avg_mae_R":round(float(losses.mae_R.mean()),3) if len(losses) else None,
      "win_avg_mfe_R":round(float(wins.mfe_R.mean()),3) if len(wins) else None,
      "win_avg_mae_R":round(float(wins.mae_R.mean()),3) if len(wins) else None,
      "losses_that_reached_0_5R":int(losses.reached_0_5R.sum()),
      "losses_that_reached_1R":int(losses.reached_1R.sum()),
      "losses_that_reached_1_5R":int(losses.reached_1_5R.sum()),
      "trades_with_same_bar_stop_and_1R":int(df.same_bar_initial_stop_and_1R.sum()),
      "stop_loss_trades_with_same_bar_stop_and_1R":int(((df.exit_reason=="STOP_LOSS") & df.same_bar_initial_stop_and_1R).sum()),
      "method":"MFE/MAE computed from observed 5m OHLC between recorded entry and exit; same-bar touches are ambiguous because OHLC does not reveal intrabar ordering.",
      "limitations":[
        "MFE/MAE are observed excursions, not executable counterfactual PnL.",
        "Same-bar stop/1R ambiguity is counted, not resolved by assuming a favorable path.",
        "Trade list is the independently replayed per-symbol sample, not portfolio concurrency."
      ]
    }
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print("\nBY EXIT REASON\n",pd.read_csv(out/"by_exit_reason.csv").to_string(index=False))
    print("\nBY SETUP/DIRECTION\n",pd.read_csv(out/"by_setup_direction.csv").to_string(index=False))

if __name__=="__main__": main()
