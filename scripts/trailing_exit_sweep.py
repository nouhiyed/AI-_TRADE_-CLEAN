#!/usr/bin/env python3
"""Retained-trailing-stop robustness sweep. Entry logic is immutable."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import pandas as pd
import backtest_v3_2y as bt

PROFILES = [
    ("baseline_1p5x_5bar", 1.5, 5),
    ("tight_1p0x_5bar", 1.0, 5),
    ("wide_2p0x_5bar", 2.0, 5),
    ("structure_1p5x_8bar", 1.5, 8),
]

def summarize(rows):
    return bt.summarize(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=",".join(bt.SYMBOLS))
    ap.add_argument("--start", default=bt.START_UTC)
    ap.add_argument("--out", default="data/trailing_exit_sweep")
    args=ap.parse_args()
    start_ms=int(pd.Timestamp(args.start).timestamp()*1000)
    end_ms=int(pd.Timestamp.now(tz="UTC").timestamp()*1000)
    symbols=[x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    original=bt.calculate_trailing_stop
    all_rows=[]; all_trades=[]
    try:
        for name,mult,swing in PROFILES:
            def configured(frame,direction,previous_stop,best_price,current_price,
                           atr_period=14,atr_multiple=mult,swing_lookback=swing):
                return original(frame,direction,previous_stop,best_price,current_price,
                                atr_period=atr_period,atr_multiple=mult,
                                swing_lookback=swing)
            bt.calculate_trailing_stop=configured
            rows=[]; trades=[]
            print(f"\nPROFILE {name}: trailing retained; activation remains +1R",flush=True)
            for symbol in symbols:
                raw=bt.fetch_5m("futures",symbol,start_ms,end_ms)
                result,meta=bt.replay_symbol("futures",symbol,raw)
                for t in result:
                    t["profile"]=name
                    t["period"]="OOS" if pd.Timestamp(t["entry_time"])>=pd.Timestamp("2025-11-16T00:00:00Z") else "IS"
                trades.extend(result)
                rows.append({"profile":name,"symbol":symbol,**summarize(result),
                             "open_at_end":bool(meta["open_at_end"])})
                print(f"  {symbol}: {len(result)} trades, {summarize(result)['net_R']}R",flush=True)
            trades.sort(key=lambda x:(x["exit_time"],x["symbol"]))
            for period in ("IS","OOS"):
                part=[t for t in trades if t["period"]==period]
                rows.append({"profile":name,"symbol":"ALL","period":period,**summarize(part)})
            rows.append({"profile":name,"symbol":"ALL","period":"TOTAL",**summarize(trades)})
            all_rows.extend(rows); all_trades.extend(trades)
    finally:
        bt.calculate_trailing_stop=original
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out/"profile_summary.csv",index=False)
    pd.DataFrame(all_trades).to_csv(out/"all_trades.csv",index=False)
    report={
      "purpose":"Compare trailing exit parameter robustness without removing trailing.",
      "immutable_entry":"V3 signal logic, entry timing, drift validation, stop placement, fees and slippage unchanged.",
      "trailing_policy":"Every profile retains +1R trailing activation, ATR/structure trailing, monotonic stop tightening and cost-aware break-even floor. No fixed-TP or no-trailing profile exists.",
      "profiles":[{"name":n,"atr_multiple":m,"swing_lookback":s,"activation_initial_R":1.0} for n,m,s in PROFILES],
      "results":all_rows,
      "acceptance_rule":"Do not select on pooled total alone. Candidate must be positive net_R and PF>1 in OOS, retain sufficient trade count, and not materially worsen drawdown; otherwise reject all profiles and redesign entry logic while keeping trailing.",
      "limitations":["Per-symbol sequential replay; no shared-account concurrency or capital contention.","Parameter sweep is diagnostic, not proof of future profitability."]
    }
    (out/"report.json").write_text(json.dumps(report,indent=2,default=str))
    print("\nSAVED",out,flush=True)
if __name__=="__main__": main()
