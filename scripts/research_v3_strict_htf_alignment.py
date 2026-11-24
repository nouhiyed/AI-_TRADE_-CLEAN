#!/usr/bin/env python3
"""Isolated V3 research: strict 15m+1h EMA alignment; exit engine is untouched."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import backtest_v3_2y as bt

def frame_values(frame, value_fn):
    return {key: value_fn(frame, key) for key in ("close", "EMA_21", "EMA_50")}

def aligned_values(values, direction):
    if not values or any(values.get(k) is None for k in ("close", "EMA_21", "EMA_50")):
        return False
    close, ema21, ema50 = (float(values[k]) for k in ("close", "EMA_21", "EMA_50"))
    if direction == "LONG":
        return close > ema21 > ema50
    if direction == "SHORT":
        return close < ema21 < ema50
    return False

def strict_htf_alignment(data, direction, value_fn):
    # Both frames are the already-closed frames supplied by the original replay.
    return all(aligned_values(frame_values(data.get(tf), value_fn), direction)
               for tf in ("15m", "1h"))

def demo_entry_is_valid(plan, direction, market, entry):
    # Same DEMO 0.30 ATR drift gate used in the previous alignment experiment.
    try:
        planned, stop, target, atr = map(float, (
            plan["entry"], plan["stop_loss"], plan["take_profit"], plan["atr"]))
        if min(planned, stop, target, atr, float(entry)) <= 0:
            return False
        if abs(float(entry) - planned) > max(atr * 0.30, planned * 0.0002):
            return False
        if direction == "SHORT":
            if not target < entry < stop: return False
            risk, reward = stop-entry, entry-target
        else:
            if not stop < entry < target: return False
            risk, reward = entry-stop, target-entry
        cost = 2.0 * (bt.FEE[market] + bt.SLIPPAGE[market])
        net_reward, net_risk = reward/entry-cost, risk/entry+cost
        return net_reward > 0 and net_reward/net_risk >= 1.5
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default=",".join(bt.SYMBOLS))
    p.add_argument("--start", default=bt.START_UTC)
    p.add_argument("--out", default="data/v3_strict_htf_alignment")
    a=p.parse_args()
    counters=Counter()
    original_analyze=bt.FuturesScalpingStrategyV3.analyze
    original_validator=bt.entry_is_valid

    def gated_analyze(self, data, costs=None):
        plan=original_analyze(self, data, costs)
        if plan.get("success") is not True:
            return plan
        direction=str(plan.get("direction","")).upper()
        counters["valid_before_alignment"] += 1
        if not strict_htf_alignment(data, direction, self.value):
            counters["rejected_alignment"] += 1
            counters["rejected_"+str(plan.get("setup","UNKNOWN"))+"_"+direction] += 1
            return {"success":False,"reason":"Research gate: strict 15m+1h EMA stack not aligned"}
        counters["passed_alignment"] += 1
        counters["passed_"+str(plan.get("setup","UNKNOWN"))+"_"+direction] += 1
        return plan

    bt.FuturesScalpingStrategyV3.analyze=gated_analyze
    bt.entry_is_valid=demo_entry_is_valid
    all_trades=[]; per_symbol=[]
    try:
        start_ms=int(pd.Timestamp(a.start).timestamp()*1000)
        end_ms=int(pd.Timestamp.now(tz="UTC").timestamp()*1000)
        for symbol in [x.strip().upper() for x in a.symbols.split(",") if x.strip()]:
            print(f"[FUTURES] {symbol}: strict 15m+1h EMA stack",flush=True)
            raw=bt.fetch_5m("futures",symbol,start_ms,end_ms)
            trades,meta=bt.replay_symbol("futures",symbol,raw)
            all_trades.extend(trades)
            per_symbol.append({"symbol":symbol,**bt.summarize(trades),
                               "candles":meta["candles"],"open_at_end":bool(meta["open_at_end"])})
            print(f"  trades={len(trades)} net_R={bt.summarize(trades)['net_R']}",flush=True)
    finally:
        bt.FuturesScalpingStrategyV3.analyze=original_analyze
        bt.entry_is_valid=original_validator

    all_trades.sort(key=lambda x:(x.get("exit_time",""),x.get("symbol","")))
    split=pd.Timestamp("2025-11-16T00:00:00Z")
    for t in all_trades:
        t["period"]="OOS_SECOND_YEAR" if pd.Timestamp(t["entry_time"])>=split else "IS_FIRST_YEAR"
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_trades).to_csv(out/"trades_strict_alignment.csv",index=False)
    pd.DataFrame(per_symbol).to_csv(out/"summary_by_symbol.csv",index=False)
    periods=[]
    for name in ("IS_FIRST_YEAR","OOS_SECOND_YEAR"):
        rows=[t for t in all_trades if t["period"]==name]
        periods.append({"period":name,**bt.summarize(rows)})
    pd.DataFrame(periods).to_csv(out/"summary_by_period.csv",index=False)
    report={
      "experiment":"V3 unchanged signals with strict directional EMA stack on both closed 15m and 1h frames",
      "gate":{"LONG":"15m close > EMA21 > EMA50 AND 1h close > EMA21 > EMA50",
              "SHORT":"15m close < EMA21 < EMA50 AND 1h close < EMA21 < EMA50"},
      "entry_validation":"Same DEMO 0.30 ATR / 0.02% drift gate and net RR >= 1.5 as prior entry-alignment experiment.",
      "counts":dict(counters),
      "pooled_result":bt.summarize(all_trades),
      "per_symbol":per_symbol,"period_split":periods,
      "exit_engine_unchanged":True,
      "exit_engine_note":"Uses the original backtest_v3_2y.replay_symbol without edits: initial stop, +1R trailing activation, same trailing calculation, same cost-aware trailing floor, same candle ordering and same update sequence. No trailing stop/take-profit parameters or code changed.",
      "scope":"Futures only; independent symbol replays, not shared portfolio simulation.",
      "limitations":["Research result, not evidence of future profitability.",
                     "Strict alignment is one isolated filter; no other signal, stop, target, trailing, or exit rule was changed.",
                     "The strategy plan's take_profit remains used only by the existing net reward/risk entry validation; realized exits remain the original trailing engine."]
    }
    (out/"report.json").write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__": main()
