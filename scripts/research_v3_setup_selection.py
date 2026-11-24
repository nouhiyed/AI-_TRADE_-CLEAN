#!/usr/bin/env python3
"""Read-only V3 setup-selection experiment; original replay and exit engine remain untouched."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_v3_2y as bt

SPLIT = pd.Timestamp("2025-11-16T00:00:00Z")
POLICIES = ("FIRST_MATCH", "MAX_CLEARANCE", "MOMENTUM", "HTF_ALIGNMENT")


def candidates(strategy, data):
    """Reproduce V3 Futures setup predicates, then build every eligible plan."""
    five, fifteen, hourly = (data.get(k) for k in ("5m", "15m", "1h"))
    C = strategy
    v = C.value
    c,o,h,low,ema9,ema20,atr,rsi = (v(five,x) for x in
        ("close","open","high","low","EMA_9","EMA_20","ATR","RSI"))
    hist,prev,adx5 = v(five,"MACD_hist"),v(five,"MACD_hist",-2),v(five,"ADX")
    if None in (c,o,h,low,ema9,ema20,atr,rsi,hist,prev,adx5) or atr <= 0:
        return []
    shape=C.candle_shape(five); vr=C.volume_ratio(five)
    if shape is None or vr is None: return []
    body=abs(c-o)/atr; loc=shape["close_location"]
    prior_high=float(five["high"].iloc[-C.BREAKOUT_LOOKBACK-1:-1].max())
    prior_low=float(five["low"].iloc[-C.BREAKOUT_LOOKBACK-1:-1].min())
    bull,bear=C.bias(fifteen,hourly,"LONG"),C.bias(fifteen,hourly,"SHORT")
    tests=[
      (bull and low<=prior_low+.12*atr and c>o and c>ema9 and c<=ema9+.45*atr and loc>=.66 and shape["lower_wick_range"]>=.28 and 35<=rsi<=62 and hist>=prev and vr>=.82,"LONG","SUPPORT_RECLAIM_5M","BULL_TREND",1.60),
      (bull and low<=ema20+.12*atr and c>ema9 and c>o and 45<=rsi<=67 and hist>=prev and loc>=C.PULLBACK_CLOSE_LOCATION and body>=C.PULLBACK_BODY_ATR and vr>=C.PULLBACK_VOLUME and shape["lower_wick_range"]>=.16,"LONG","PULLBACK_5M","BULL_TREND",1.60),
      (bear and h>=ema20-.12*atr and c<ema9 and c<o and 33<=rsi<=55 and hist<=prev and loc<=1-C.PULLBACK_CLOSE_LOCATION and body>=C.PULLBACK_BODY_ATR and vr>=C.PULLBACK_VOLUME and shape["upper_wick_range"]>=.16,"SHORT","PULLBACK_5M","BEAR_TREND",1.60),
      (bull and c>prior_high+.05*atr and c>o and 52<=rsi<=74 and loc>=C.BREAKOUT_CLOSE_LOCATION and body>=C.BREAKOUT_BODY_ATR and vr>=C.BREAKOUT_VOLUME and shape["upper_wick_range"]<=C.BREAKOUT_MAX_OPPOSING_WICK,"LONG","BREAKOUT_5M","BULL_TREND",1.80),
      (bear and c<prior_low-.05*atr and c<o and 26<=rsi<=48 and loc<=1-C.BREAKOUT_CLOSE_LOCATION and body>=C.BREAKOUT_BODY_ATR and vr>=C.BREAKOUT_VOLUME and shape["lower_wick_range"]<=C.BREAKOUT_MAX_OPPOSING_WICK,"SHORT","BREAKOUT_5M","BEAR_TREND",1.80),
    ]
    adx15=v(fifteen,"ADX"); bb_low,bb_high=v(five,"BB_lower"),v(five,"BB_upper")
    tests += [
      (adx15 is not None and adx15<=18 and loc>=.72 and c>o and rsi<=30 and vr>=.80 and shape["lower_wick_range"]>=C.REJECTION_WICK_RANGE and bb_low is not None and c<=bb_low*1.002,"LONG","RANGE_REJECTION_5M","RANGE",1.55),
      (adx15 is not None and adx15<=18 and loc<=.28 and c<o and rsi>=70 and vr>=.80 and shape["upper_wick_range"]>=C.REJECTION_WICK_RANGE and bb_high is not None and c>=bb_high*.998,"SHORT","RANGE_REJECTION_5M","RANGE",1.55),
    ]
    result=[]
    for ok,direction,setup,regime,target_r in tests:
        if not ok: continue
        plan=C.plan_with_structure("futures",direction,five,data,setup,regime,target_r,c,data.get("profile","standard"))
        sign=1 if direction=="LONG" else -1
        atr15=v(fifteen,"ATR"); atr1h=v(hourly,"ATR")
        gap15=((v(fifteen,"EMA_21")-v(fifteen,"EMA_50"))*sign/atr15
               if atr15 else 0.0)
        gap1h=((v(hourly,"EMA_21")-v(hourly,"EMA_50"))*sign/atr1h
               if atr1h else 0.0)
        macd=sign*(hist-prev)/atr
        ext=sign*(c-ema9)/atr
        clearance=(plan.get("structure_guard") or {}).get("distance_atr")
        if clearance is None: clearance=99.0
        result.append({"plan":plan,"setup":setup,"direction":direction,
          "clearance":float(clearance),"momentum":float(macd),
          "htf":float(gap15+gap1h),"volume":float(vr),"extension":float(ext)})
    return result


def choose(strategy,data,policy):
    items=candidates(strategy,data)
    if not items: return {"success":False,"reason":"No qualifying V3 Futures setup"}
    if policy=="FIRST_MATCH": return items[0]["plan"]
    items=[x for x in items if x["plan"].get("success") is True]
    if not items: return {"success":False,"reason":"No successful candidate plan"}
    if policy=="MAX_CLEARANCE": key=lambda x:(x["clearance"],x["momentum"])
    elif policy=="MOMENTUM": key=lambda x:(x["momentum"],x["clearance"])
    elif policy=="HTF_ALIGNMENT": key=lambda x:(x["htf"],x["momentum"])
    else: raise ValueError(policy)
    return max(items,key=key)["plan"]


def metrics(trades):
    if not trades: return {"trades":0,"wins":0,"win_rate_pct":0.0,"net_R":0.0,"average_R":None,"profit_factor":None}
    x=np.asarray([float(t["net_R"]) for t in trades]); gain=x[x>0].sum(); loss=-x[x<0].sum()
    return {"trades":len(x),"wins":int((x>0).sum()),"win_rate_pct":round(float((x>0).mean()*100),2),
      "net_R":round(float(x.sum()),3),"average_R":round(float(x.mean()),4),
      "profit_factor":round(float(gain/loss),3) if loss else None}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbols",default=",".join(bt.SYMBOLS))
    ap.add_argument("--start",default=bt.START_UTC)
    ap.add_argument("--out",default="data/v3_setup_selection")
    args=ap.parse_args()
    original=bt.FuturesScalpingStrategyV3.analyze
    rows=[]; alltrades=[]
    start_ms=int(pd.Timestamp(args.start).timestamp()*1000)
    end_ms=int(pd.Timestamp.now(tz="UTC").timestamp()*1000)
    try:
      for policy in POLICIES:
        def wrapped(self,data,costs=None,_p=policy): return choose(self,data,_p)
        bt.FuturesScalpingStrategyV3.analyze=wrapped
        trades_for_policy=[]
        for symbol in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
          raw=bt.fetch_5m("futures",symbol,start_ms,end_ms)
          trades,meta=bt.replay_symbol("futures",symbol,raw)
          for t in trades: t["policy"]=policy
          trades_for_policy.extend(trades)
          print(f"{policy} {symbol}: {len(trades)} trades",flush=True)
        for period,lo,hi in (("IS_FIRST_YEAR",None,SPLIT),("OOS_SECOND_YEAR",SPLIT,None),("ALL",None,None)):
          subset=[t for t in trades_for_policy if (lo is None or pd.Timestamp(t["entry_time"])>=lo) and (hi is None or pd.Timestamp(t["entry_time"])<hi)]
          rows.append({"policy":policy,"period":period,**metrics(subset)})
        alltrades.extend(trades_for_policy)
    finally:
      bt.FuturesScalpingStrategyV3.analyze=original
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(alltrades).to_csv(out/"trades_by_policy.csv",index=False)
    report={"experiment":"All qualifying V3 Futures setup candidates compared under frozen replay",
      "results":rows,"policies":list(POLICIES),
      "selection":"Candidate predicates mirror original Futures analyze(); FIRST_MATCH preserves original precedence. Alternative policies choose one successful plan per signal candle.",
      "execution":"Original bt.replay_symbol called unchanged for every policy; next-open execution, entry validation and one-position-per-symbol sequencing preserved.",
      "exit_engine_unchanged":True,
      "exit_engine_note":"No changes to replay/exit code, +1R activation, trailing calculations, cost-aware floor, candle ordering or stop update sequence.",
      "split":"First year IS ending 2025-11-16 UTC; second year OOS. Policies are diagnostic fixed rankings, not production candidates.",
      "scope":"Eight Futures symbols replayed independently; not portfolio simulation."}
    (out/"report.json").write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__": main()
