#!/usr/bin/env python3
"""Two-year V3 historical backtest using the production ATR/structure trailing stop."""
import os, io, re, time, zipfile, hashlib, calendar, json, math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from core.trailing_stop import calculate_trailing_stop

SYMBOLS=[x.strip().upper() for x in os.getenv("SYMBOLS","BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT").split(",") if x.strip()]
DAYS=int(os.getenv("DAYS","730"))
MARKETS=[x.strip().lower() for x in os.getenv("MARKETS","spot,futures").split(",") if x.strip()]
if DAYS<30 or not SYMBOLS or not MARKETS or any(x not in ("spot","futures") for x in MARKETS):
    raise ValueError("DAYS must be >=30 and SYMBOLS/MARKETS must be valid")
OUT=Path("backtest-trailing-results"); CACHE=Path(".backtest-trailing-cache")
OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
DAY_MS=86400000; NOW=int(time.time()*1000)
END=(NOW//DAY_MS)*DAY_MS-60000; START=END-DAYS*DAY_MS+60000
BASE="https://data.binance.vision/data"; SESSION=requests.Session()
SESSION.headers.update({"User-Agent":"AI-Trader-V3-Trailing-Backtest/1.0"})
FEE={"spot":0.001,"futures":0.0005}; ADVERSE=0.0005
MIN_COVERAGE=0.995

def download(url,label):
    error=None
    for attempt in range(6):
        try:
            r=SESSION.get(url,timeout=(20,120))
            if r.status_code in (418,429) or r.status_code>=500:
                time.sleep(min(45,2**(attempt+1))); continue
            if r.status_code==404: raise RuntimeError(f"Missing Binance archive: {label}")
            r.raise_for_status(); return r.content
        except RuntimeError: raise
        except Exception as exc:
            error=exc
            if attempt<5: time.sleep(min(30,2**(attempt+1)))
    raise RuntimeError(f"Download failed for {label}: {error}")

def archive_url(market,period,symbol,date):
    m="spot" if market=="spot" else "futures/um"
    return f"{BASE}/{m}/{period}/klines/{symbol}/1m/{symbol}-1m-{date}.zip"

def read_archive(url,label):
    raw=download(url,label); checksum=download(url+".CHECKSUM",label+" checksum").decode("utf8","replace")
    match=re.search(r"([0-9a-fA-F]{64})",checksum)
    if not match: raise RuntimeError(f"Missing SHA256 checksum: {label}")
    digest=hashlib.sha256(raw).hexdigest()
    if digest.lower()!=match.group(1).lower(): raise RuntimeError(f"SHA256 mismatch: {label}")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=[n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(names)!=1: raise RuntimeError(f"Expected one CSV in {label}, got {len(names)}")
        with z.open(names[0]) as f:
            df=pd.read_csv(f,header=None,usecols=[0,1,2,3,4,5],
                names=["open_time","open","high","low","close","volume"])
    df["open_time"]=pd.to_numeric(df.open_time,errors="coerce")
    if df.open_time.dropna().median()>100000000000000: df["open_time"]//=1000
    for c in ("open","high","low","close","volume"): df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna().drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=df.open_time.astype("int64")
    if df.empty: raise RuntimeError(f"Empty archive: {label}")
    return df

def month_iter(a,b):
    cur=datetime(a.year,a.month,1,tzinfo=timezone.utc); last=datetime(b.year,b.month,1,tzinfo=timezone.utc)
    while cur<=last:
        yield cur
        cur=(datetime(cur.year+1,1,1,tzinfo=timezone.utc) if cur.month==12
             else datetime(cur.year,cur.month+1,1,tzinfo=timezone.utc))

def load_data(market,symbol):
    path=CACHE/f"{market}_{symbol}_{DAYS}d.csv"
    if path.exists():
        df=pd.read_csv(path); df["open_time"]=pd.to_numeric(df.open_time,errors="coerce")
        if df.open_time.dropna().median()>100000000000000: df["open_time"]//=1000
        df=df.dropna().drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        try: validate(df,market,symbol); print(f"[CACHE] {market} {symbol}: {len(df):,}",flush=True); return df
        except Exception: print(f"[CACHE] rejected incomplete cache {market} {symbol}",flush=True)
    a=datetime.fromtimestamp(START/1000,timezone.utc); b=datetime.fromtimestamp(END/1000,timezone.utc)
    current=datetime.now(timezone.utc).replace(day=1,hour=0,minute=0,second=0,microsecond=0); frames=[]
    for m in month_iter(a,b):
        if m<current:
            part=m.strftime("%Y-%m")
            frames.append(read_archive(archive_url(market,"monthly",symbol,part),f"{market}/{symbol}/{part}"))
        else:
            day=max(a.date(),m.date()); last=min(b.date(),m.replace(day=calendar.monthrange(m.year,m.month)[1]).date())
            while day<=last:
                part=day.strftime("%Y-%m-%d")
                frames.append(read_archive(archive_url(market,"daily",symbol,part),f"{market}/{symbol}/{part}"))
                day+=timedelta(days=1)
    if not frames: raise RuntimeError(f"No archives selected for {market} {symbol}")
    df=pd.concat(frames,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df=df[(df.open_time>=START)&(df.open_time<=END)][["open_time","open","high","low","close","volume"]].reset_index(drop=True)
    validate(df,market,symbol); df.to_csv(path,index=False)
    return df

def validate(df,market,symbol):
    expected=DAYS*1440; minimum=math.ceil(expected*MIN_COVERAGE)
    if df.empty or len(df)<minimum or int(df.open_time.iloc[0])>START+60000 or int(df.open_time.iloc[-1])<END-60000:
        raise RuntimeError(f"INCOMPLETE {market} {symbol}: {len(df):,}/{expected:,} candles; minimum {minimum:,}")
    return expected

def indicators(x):
    x=x.copy()
    for p in (9,20,21,50,200): x[f"EMA_{p}"]=EMAIndicator(x.close,p).ema_indicator()
    x["RSI"]=RSIIndicator(x.close,14).rsi(); x["MACD_hist"]=MACD(x.close).macd_diff()
    x["ATR"]=AverageTrueRange(x.high,x.low,x.close).average_true_range()
    x["ADX"]=ADXIndicator(x.high,x.low,x.close).adx()
    bb=BollingerBands(x.close); x["BB_lower"]=bb.bollinger_lband(); x["BB_upper"]=bb.bollinger_hband()
    return x

def resample(raw,rule):
    x=raw.set_index("dt").resample(rule,label="right",closed="right").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    return indicators(x)

def align(one,frame,prefix):
    cols=["close","EMA_21","EMA_50","EMA_200","ATR","ADX"]; right=frame[[c for c in cols if c in frame]].copy()
    right.columns=[prefix+c for c in right.columns]
    return pd.merge_asof(one.sort_index(),right.sort_index(),left_index=True,right_index=True,direction="backward")

def build_signals(raw,market):
    one=indicators(raw.set_index("dt")); five=resample(raw,"5min"); fifteen=resample(raw,"15min")
    one=align(align(one,five,"5_"),fifteen,"15_")
    if market=="spot": one=align(align(one,resample(raw,"1h"),"1h_"),resample(raw,"4h"),"4h_")
    one["candle_loc"]=(one.close-one.low)/(one.high-one.low).replace(0,np.nan)
    one["body"]=(one.close-one.open).abs()/one.ATR
    one["vr"]=one.volume/one.volume.shift(1).rolling(30).mean()
    one["prior_high"]=one.high.shift(1).rolling(10).max(); one["prior_low"]=one.low.shift(1).rolling(10).min()
    bull=(one["5_close"]>one["5_EMA_50"])&((one["5_close"]>=one["5_EMA_21"])|(one["15_EMA_21"]>one["15_EMA_50"]))
    bear=(one["15_close"]<one["15_EMA_50"])&((one["5_close"]<=one["5_EMA_21"])|(one["15_EMA_21"]<one["15_EMA_50"]))
    pl=bull&(one.low<=one.EMA_20+.18*one.ATR)&(one.close>one.EMA_9)&(one.close>one.open)&one.RSI.between(42,69)&(one.MACD_hist>=one.MACD_hist.shift(1))&(one.candle_loc>=.55)&(one.body>=.08)&(one.vr>=.72)
    ps=bear&(one.high>=one.EMA_20-.18*one.ATR)&(one.close<one.EMA_9)&(one.close<one.open)&one.RSI.between(31,58)&(one.MACD_hist<=one.MACD_hist.shift(1))&(one.candle_loc<=.45)&(one.body>=.08)&(one.vr>=.72)
    bl=bull&(one.close>one.prior_high+.015*one.ATR)&(one.close>one.open)&one.RSI.between(48,76)&(one.candle_loc>=.62)&(one.body>=.12)&(one.vr>=.85)
    bs=bear&(one.close<one.prior_low-.015*one.ATR)&(one.close<one.open)&one.RSI.between(24,52)&(one.candle_loc<=.38)&(one.body>=.12)&(one.vr>=.85)
    if market=="futures":
        rl=(one["5_ADX"]<=20)&(one.candle_loc>=.62)&(one.close>one.open)&(one.RSI<=29)&(one.close<=one.BB_lower*1.003)
        rs=(one["5_ADX"]<=20)&(one.candle_loc<=.38)&(one.close<one.open)&(one.RSI>=71)&(one.close>=one.BB_upper*.997)
        masks=[pl,ps,bl,bs,rl,rs]; dirs=["LONG","SHORT","LONG","SHORT","LONG","SHORT"]
        setups=["MICRO_PULLBACK","MICRO_PULLBACK","MICRO_BREAKOUT","MICRO_BREAKOUT","RANGE_REJECTION","RANGE_REJECTION"]
        base=[1.35,1.35,1.45,1.45,1.25,1.25]
    else:
        bullish=bull|((one["1h_close"]>one["1h_EMA_200"])&(one["5_close"]>=one["5_EMA_21"]))
        deep=(one["4h_close"]<one["4h_EMA_200"]*.975)&(one["1h_close"]<one["1h_EMA_200"])
        pull=bullish&~deep&(one.low<=one.EMA_20+.20*one.ATR)&(one.close>one.EMA_9)&(one.close>one.open)&one.RSI.between(40,70)&(one.MACD_hist>=one.MACD_hist.shift(1))&(one.candle_loc>=.54)&(one.body>=.07)&(one.vr>=.68)
        breakout=bullish&~deep&(one.close>one.prior_high+.01*one.ATR)&(one.close>one.open)&one.RSI.between(48,76)&(one.candle_loc>=.60)&(one.body>=.11)&(one.vr>=.82)
        bounce=(one["5_ADX"]<=19)&(one.close<=one.BB_lower*1.004)&(one.RSI<=27)&(one.close>one.open)&(one.candle_loc>=.60)&(one.vr>=.65)
        masks=[pull,breakout,bounce]; dirs=["LONG"]*3; setups=["MICRO_PULLBACK","MICRO_BREAKOUT","RANGE_BOUNCE"]; base=[1.45,1.55,1.35]
    one["direction"]=""; one["setup"]=""; one["base_r"]=0.0
    for mask,direction,setup,r in zip(masks,dirs,setups,base):
        choose=mask&(one.direction=="")
        one.loc[choose,"direction"]=direction; one.loc[choose,"setup"]=setup; one.loc[choose,"base_r"]=r
    return one

def simulate(market,symbol,raw):
    sig=build_signals(raw,market).reset_index()
    # 5m candles are formed from data strictly before the current 1m candle.
    five=raw.set_index("dt").resample("5min",label="right",closed="right").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()
    trades=[]; active=None; fee=FEE[market]; cost=2*(fee+ADVERSE)
    for i in range(250,len(sig)-1):
        row=sig.iloc[i]; nxt=sig.iloc[i+1]
        price=float(row.close); hi=float(row.high); lo=float(row.low)
        if active:
            side=active["side"]; entry=active["entry"]; initial_risk=active["initial_risk"]
            # Conservative OHLC ordering: existing stop is checked before any new arm.
            stop_hit=(lo<=active["stop"]) if side=="LONG" else (hi>=active["stop"])
            if stop_hit:
                exit_price=active["stop"]; reason="TRAILING_STOP" if active["armed"] else "STOP_LOSS"
            else:
                favorable=(hi-entry if side=="LONG" else entry-lo)
                if not active["armed"] and favorable>=initial_risk:
                    active["armed"]=True; active["armed_at"]=str(row["dt"])
                # Only candles fully closed before this 1m candle may update the trail.
                if active["armed"]:
                    closed=five.loc[five.index < pd.Timestamp(row["dt"])]
                    if len(closed)>=16:
                        update=calculate_trailing_stop(closed.tail(40),side,active["stop"],
                            active["best_price"],price)
                        if update:
                            candidate=float(update["stop"])
                            active["stop"]=max(candidate,entry) if side=="LONG" else min(candidate,entry)
                            active["best_price"]=float(update["best_price"])
                            active["trail_updates"]+=1
                # A newly calculated stop becomes active from the next 1m candle.
                # This avoids look-ahead fills from lows/highs that may have occurred
                # before the current candle close used to calculate the new stop.
                exit_price=None; reason=None
            if reason:
                gross=((exit_price-entry)/entry if side=="LONG" else (entry-exit_price)/entry)
                net=gross-cost
                active.update(exit_time=str(row["dt"]),exit_price=exit_price,reason=reason,
                    gross_pct=gross*100,net_pct=net*100,net_usdt=active["notional"]*net,
                    hold_minutes=i-active["entry_i"],stop_final=active["stop"])
                trades.append(active); active=None
        if active is None and str(row.direction) in ("LONG","SHORT"):
            entry=float(nxt.open); atr=float(row["5_ATR"])
            if not np.isfinite(entry) or not np.isfinite(atr) or entry<=0 or atr<=0: continue
            distance=max(.90*atr,entry*.0008); risk_rate=distance/entry
            required=1.5+2.5*cost/risk_rate
            target_r=max(float(row.base_r),required)
            net_reward=target_r*risk_rate-cost; net_rr=net_reward/(risk_rate+cost)
            if net_reward<.00035 or net_rr<1.5: continue
            side=str(row.direction); stop=entry-distance if side=="LONG" else entry+distance
            active={"market":market,"symbol":symbol,"side":side,"setup":str(row.setup),
                "signal_time":str(row["dt"]),"entry_time":str(nxt["dt"]),"entry":entry,
                "initial_stop":stop,"stop":stop,"initial_risk":distance,"best_price":entry,
                "armed":False,"armed_at":None,"trail_updates":0,"entry_i":i+1,
                "notional":100.0,"target_r_reference":target_r,"net_rr_at_entry":net_rr}
    if active:
        row=sig.iloc[-1]; exit_price=float(row.close); entry=active["entry"]; side=active["side"]
        gross=((exit_price-entry)/entry if side=="LONG" else (entry-exit_price)/entry); net=gross-cost
        active.update(exit_time=str(row["dt"]),exit_price=exit_price,reason="END_OF_DATA",
            gross_pct=gross*100,net_pct=net*100,net_usdt=active["notional"]*net,
            hold_minutes=len(sig)-1-active["entry_i"],stop_final=active["stop"])
        trades.append(active)
    return trades,len(sig)

def main():
    all_trades=[]; summaries=[]; audit=[]; errors=[]
    expected_pairs=len(MARKETS)*len(SYMBOLS)
    print(f"[START] V3 TRAILING historical simulation: {DAYS} days; {MARKETS}; {SYMBOLS}",flush=True)
    print("[EXIT] Initial SL; arm at +1R; production ATR 1.5x + 5-candle structure; breakeven clamp; NO TIME EXIT",flush=True)
    for market in MARKETS:
        for symbol in SYMBOLS:
            print(f"[RUN] {market.upper()} {symbol}",flush=True)
            try:
                df=load_data(market,symbol); expected=validate(df,market,symbol)
                df["dt"]=pd.to_datetime(df.open_time,unit="ms",utc=True)
                raw=df[["dt","open","high","low","close","volume"]].copy()
                trades,candle_count=simulate(market,symbol,raw); all_trades.extend(trades)
                t=pd.DataFrame(trades)
                wins=int((t.net_usdt>0).sum()) if len(t) else 0
                gross=float(t.net_usdt.sum()) if len(t) else 0.0
                losses=t.loc[t.net_usdt<0,"net_usdt"].sum() if len(t) else 0.0
                profits=t.loc[t.net_usdt>0,"net_usdt"].sum() if len(t) else 0.0
                equity=100+t.net_usdt.cumsum() if len(t) else pd.Series([100])
                peak=equity.cummax(); dd=float(((peak-equity)/peak).max()*100)
                summary={"market":market,"symbol":symbol,"candles":len(df),"expected_candles":expected,
                    "coverage_pct":round(len(df)/expected*100,4),"trades":len(t),"wins":wins,
                    "losses":len(t)-wins,"win_rate_pct":round(wins/len(t)*100,2) if len(t) else 0,
                    "net_pnl_usdt_per_100_notional":round(gross,4),
                    "profit_factor":round(float(profits/abs(losses)),3) if losses<0 else None,
                    "max_drawdown_pct":round(dd,3),
                    "armed_trades":int(t.armed.sum()) if len(t) else 0,
                    "trailing_stop_exits":int((t.reason=="TRAILING_STOP").sum()) if len(t) else 0,
                    "initial_stop_exits":int((t.reason=="STOP_LOSS").sum()) if len(t) else 0,
                    "end_of_data":int((t.reason=="END_OF_DATA").sum()) if len(t) else 0,
                    "avg_hold_minutes":round(float(t.hold_minutes.mean()),2) if len(t) else 0}
                summaries.append(summary)
                audit.append({"market":market,"symbol":symbol,"status":"VALID","candles":len(df),
                    "expected":expected,"coverage_pct":round(len(df)/expected*100,4)})
                print("[DONE]",json.dumps(summary),flush=True)
            except Exception as exc:
                msg=f"{market} {symbol}: {type(exc).__name__}: {exc}"; errors.append(msg)
                audit.append({"market":market,"symbol":symbol,"status":"ERROR","error":msg})
                print("[ERROR]",msg,flush=True)
    pd.DataFrame(summaries).to_csv(OUT/"summary.csv",index=False)
    pd.DataFrame(all_trades).drop(columns=["entry_i"],errors="ignore").to_csv(OUT/"trades.csv",index=False)
    status="VALID" if not errors and len(summaries)==expected_pairs else "FAILED"
    methodology={"strategy":"Scalping V3 signals + production calculate_trailing_stop",
        "days":DAYS,"start_utc":datetime.fromtimestamp(START/1000,timezone.utc).isoformat(),
        "end_utc":datetime.fromtimestamp(END/1000,timezone.utc).isoformat(),
        "markets":MARKETS,"symbols":SYMBOLS,"data_source":"Binance Vision public archives",
        "archive_sha256_verified":True,"minimum_coverage_pct":MIN_COVERAGE*100,
        "fees_round_trip":{"spot":2*FEE["spot"],"futures":2*FEE["futures"]},
        "adverse_cost_round_trip":2*ADVERSE,"entry":"next 1m open",
        "trailing_activation":"favorable excursion reaches initial 1R",
        "trailing_calculation":"core.trailing_stop.calculate_trailing_stop; ATR 1.5x plus 5-candle structure",
        "trailing_update_data":"only fully closed 5m candles before current 1m candle",
        "breakeven_clamp":"stop cannot move behind entry after activation",
        "intrabar_policy":"existing stop checked first; no fixed take-profit; no time exit",
        "end_of_data":"mark remaining position at final close; reported separately",
        "expected_pairs":expected_pairs,"successful_pairs":len(summaries),
        "trade_count":len(all_trades),"errors":errors,"status":status}
    (OUT/"methodology.json").write_text(json.dumps(methodology,indent=2))
    (OUT/"data_audit.json").write_text(json.dumps({"status":status,"pairs":audit,"errors":errors},indent=2))
    print(f"[FINISHED] status={status} pairs={len(summaries)}/{expected_pairs} trades={len(all_trades)} errors={len(errors)}",flush=True)
    if status!="VALID": raise SystemExit("Backtest invalid: one or more symbol datasets or simulations failed.")

if __name__=="__main__": main()
