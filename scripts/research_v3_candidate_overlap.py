#!/usr/bin/env python3
"""Read-only diagnostic: candidate overlap and successful-plan multiplicity."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import backtest_v3_2y as bt
from research_v3_setup_selection import candidates

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--symbols",default=",".join(bt.SYMBOLS)); ap.add_argument("--out",default="data/v3_candidate_overlap"); args=ap.parse_args()
 start_ms=int(pd.Timestamp(bt.START_UTC).timestamp()*1000); end_ms=int(pd.Timestamp.now(tz="UTC").timestamp()*1000)
 rows=[]
 for symbol in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
  raw=bt.fetch_5m("futures",symbol,start_ms,end_ms); five,fifteen,hourly,_=bt.build_indicators(raw,"futures")
  five=five.reset_index(drop=True); times=bt.to_ms(five["open_time"]); closes=times+bt.INTERVAL_MS
  f15=bt.htf_end_indices(fifteen,15); f1h=bt.htf_end_indices(hourly,60)
  for i in range(59,len(five)-1):
   trigger=bt.set_frame_attrs(five.iloc[i-59:i+1].copy(),"5m")
   if not bt.FuturesScalpingStrategyV3.prefilter_5m(trigger).get("candidate"): continue
   close_ms=int(closes[i]); j15=int(__import__("numpy").searchsorted(f15,close_ms,side="right")); j1h=int(__import__("numpy").searchsorted(f1h,close_ms,side="right"))
   if j15<60 or j1h<60: continue
   data={"5m":trigger,"15m":bt.set_frame_attrs(fifteen.iloc[max(0,j15-60):j15].copy(),"15m"),"1h":bt.set_frame_attrs(hourly.iloc[max(0,j1h-60):j1h].copy(),"1h")}
   found=candidates(bt.FuturesScalpingStrategyV3(),data)
   if found:
    rows.append({"symbol":symbol,"signal_time":pd.Timestamp(int(closes[i]),unit="ms",tz="UTC").isoformat(),"count":len(found),"successful":sum(x["plan"].get("success") is True for x in found),"setups":";".join(x["setup"] for x in found),"directions":";".join(x["direction"] for x in found),"successful_setups":";".join(x["setup"] for x in found if x["plan"].get("success") is True),"clearances":";".join(str(round(x["clearance"],4)) for x in found)})
  print(symbol,"processed",flush=True)
 df=pd.DataFrame(rows); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); df.to_csv(out/"candidate_events.csv",index=False)
 if len(df):
  multi=df[df["count"]>1]; multi_ok=df[df["successful"]>1]; zero_ok=df[df["successful"]==0]
  report={"candidate_events":len(df),"events_multiple_candidates":len(multi),"events_multiple_successful_plans":len(multi_ok),"events_no_successful_plans":len(zero_ok),"successful_plan_multiplicity_distribution":df.successful.value_counts().sort_index().to_dict(),"candidate_multiplicity_distribution":df['count'].value_counts().sort_index().to_dict(),"multiple_candidate_examples":multi.head(100).to_dict("records"),"multiple_success_examples":multi_ok.head(100).to_dict("records"),"note":"Read-only signal-bar diagnostic; no replay or exit engine edits."}
 else: report={"candidate_events":0,"note":"No candidate bars captured"}
 (out/"report.json").write_text(json.dumps(report,indent=2,default=str)); print(json.dumps(report,indent=2,default=str))
if __name__=="__main__": main()
