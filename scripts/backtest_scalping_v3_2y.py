#!/usr/bin/env python3
"""Two-year, candle-by-candle Scalping V3 backtest. Analysis only; never places orders."""
import os, time, zipfile, io, json
from pathlib import Path
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT").split(",")
DAYS = int(os.getenv("DAYS", "730"))
MARKETS = os.getenv("MARKETS", "spot,futures").split(",")
OUT = Path("backtest-results")
CACHE = Path(".backtest-cache")
OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
NOW = int(time.time() * 1000)
START = NOW - DAYS * 86400000
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AI-Trader-V3-Historical-Backtest/1.0"})

def fetch_1m(market, symbol):
    path = CACHE / f"{market}_{symbol}_{DAYS}d.csv"
    if path.exists():
        df = pd.read_csv(path)
        if len(df) and int(df.open_time.iloc[0]) <= START + 60000 and int(df.open_time.iloc[-1]) >= NOW - 180000:
            return df
    base = "https://api.binance.com/api/v3/klines" if market == "spot" else "https://fapi.binance.com/fapi/v1/klines"
    rows, cursor = [], START
    while cursor < NOW:
        for attempt in range(8):
            try:
                response = SESSION.get(base, params={"symbol": symbol, "interval": "1m",
                    "startTime": cursor, "endTime": NOW, "limit": 1000}, timeout=25)
                if response.status_code == 429 or response.status_code == 418:
                    time.sleep(min(60, 3 ** (attempt + 1))); continue
                response.raise_for_status()
                batch = response.json()
                break
            except Exception:
                if attempt == 7: raise
                time.sleep(2 + attempt * 2)
        if not batch: break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + 60000
        if nxt <= cursor: break
        cursor = nxt
        if len(batch) < 1000: break
        time.sleep(0.07)
        if len(rows) % 100000 < 1000:
            print(f"[DATA] {market} {symbol}: {len(rows):,} one-minute candles", flush=True)
    if not rows: raise RuntimeError(f"No Binance data for {market} {symbol}")
    cols = ["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df = df[["open_time","open","high","low","close","volume"]]
    for c in cols[1:6]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_numeric(df.open_time, errors="coerce").astype("int64")
    df = df.dropna().drop_duplicates("open_time").sort_values("open_time")
    df.to_csv(path, index=False)
    return df

def indicators(raw):
    x = raw.copy()
    for p in (9,20,21,50,200):
        x[f"EMA_{p}"] = EMAIndicator(x.close, p).ema_indicator()
    x["RSI"] = RSIIndicator(x.close, 14).rsi()
    x["MACD_hist"] = MACD(x.close).macd_diff()
    x["ATR"] = AverageTrueRange(x.high, x.low, x.close).average_true_range()
    x["ADX"] = ADXIndicator(x.high, x.low, x.close).adx()
    bb = BollingerBands(x.close); x["BB_lower"] = bb.bollinger_lband(); x["BB_upper"] = bb.bollinger_hband()
    return x

def resample(raw, rule):
    x = raw.set_index("dt").resample(rule, label="right", closed="right").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    return indicators(x)

def align(one, frame, prefix):
    cols = ["close","EMA_21","EMA_50","EMA_200","ATR","ADX"]
    keep = [c for c in cols if c in frame.columns]
    right = frame[keep].copy()
    right.columns = [prefix+c for c in right.columns]
    return pd.merge_asof(one.sort_index(), right.sort_index(), left_index=True, right_index=True, direction="backward")

def build_signals(raw, market):
    one = indicators(raw.set_index("dt"))
    five, fifteen = resample(raw, "5min"), resample(raw, "15min")
    one = align(one, five, "5_")
    one = align(one, fifteen, "15_")
    if market == "spot":
        hourly, four = resample(raw, "1h"), resample(raw, "4h")
        one = align(one, hourly, "1h_")
        one = align(one, four, "4h_")
    one["loc"] = (one.close-one.low)/(one.high-one.low).replace(0,np.nan)
    one["body"] = (one.close-one.open).abs()/one.ATR
    one["vr"] = one.volume / one.volume.shift(1).rolling(30).mean()
    one["prior_high"] = one.high.shift(1).rolling(10).max()
    one["prior_low"] = one.low.shift(1).rolling(10).min()
    bull = (one["5_close"] > one["5_EMA_50"]) & ((one["5_close"] >= one["5_EMA_21"]) | (one["15_EMA_21"] > one["15_EMA_50"]))
    bear = (one["15_close"] < one["15_EMA_50"]) & ((one["5_close"] <= one["5_EMA_21"]) | (one["15_EMA_21"] < one["15_EMA_50"]))
    pull_long = bull & (one.low <= one.EMA_20 + .18*one.ATR) & (one.close > one.EMA_9) & (one.close > one.open) & one.RSI.between(42,69) & (one["MACD_hist"] >= one["MACD_hist"].shift(1)) & (one.loc >= .55) & (one.body >= .08) & (one.vr >= .72)
    pull_short = bear & (one.high >= one.EMA_20 - .18*one.ATR) & (one.close < one.EMA_9) & (one.close < one.open) & one.RSI.between(31,58) & (one["MACD_hist"] <= one["MACD_hist"].shift(1)) & (one.loc <= .45) & (one.body >= .08) & (one.vr >= .72)
    break_long = bull & (one.close > one.prior_high + .015*one.ATR) & (one.close > one.open) & one.RSI.between(48,76) & (one.loc >= .62) & (one.body >= .12) & (one.vr >= .85)
    break_short = bear & (one.close < one.prior_low - .015*one.ATR) & (one.close < one.open) & one.RSI.between(24,52) & (one.loc <= .38) & (one.body >= .12) & (one.vr >= .85)
    if market == "futures":
        range_long = (one["5_ADX"] <= 20) & (one.loc >= .62) & (one.close > one.open) & (one.RSI <= 29) & (one.close <= one.BB_lower*1.003)
        range_short = (one["5_ADX"] <= 20) & (one.loc <= .38) & (one.close < one.open) & (one.RSI >= 71) & (one.close >= one.BB_upper*.997)
        direction = np.select([pull_long, pull_short, break_long, break_short, range_long, range_short],
                              ["LONG","SHORT","LONG","SHORT","LONG","SHORT"], default="")
        setup = np.select([pull_long,pull_short,break_long,break_short,range_long,range_short],
                          ["MICRO_PULLBACK","MICRO_PULLBACK","MICRO_BREAKOUT","MICRO_BREAKOUT","RANGE_REJECTION","RANGE_REJECTION"],default="")
        base_r = np.select([pull_long|pull_short,break_long|break_short],[1.35,1.45],default=1.25)
        max_hold, fee = 12, .0005
    else:
        bullish = bull | ((one["1h_close"] > one["1h_EMA_200"]) & (one["5_close"] >= one["5_EMA_21"]))
        deep_bear = (one["4h_close"] < one["4h_EMA_200"]*.975) & (one["1h_close"] < one["1h_EMA_200"])
        pull = bullish & ~deep_bear & (one.low <= one.EMA_20+.20*one.ATR) & (one.close > one.EMA_9) & (one.close > one.open) & one.RSI.between(40,70) & (one["MACD_hist"] >= one["MACD_hist"].shift(1)) & (one.loc >= .54) & (one.body >= .07) & (one.vr >= .68)
        breakout = bullish & ~deep_bear & (one.close > one.prior_high+.01*one.ATR) & (one.close > one.open) & one.RSI.between(48,76) & (one.loc >= .60) & (one.body >= .11) & (one.vr >= .82)
        bounce = (one["5_ADX"] <= 19) & (one.close <= one.BB_lower*1.004) & (one.RSI <= 27) & (one.close > one.open) & (one.loc >= .60) & (one.vr >= .65)
        direction = np.where(pull,"LONG",np.where(breakout,"LONG",np.where(bounce,"LONG","")))
        setup = np.where(pull,"MICRO_PULLBACK",np.where(breakout,"MICRO_BREAKOUT",np.where(bounce,"RANGE_BOUNCE","")))
        base_r = np.where(pull,1.45,np.where(breakout,1.55,1.35))
        max_hold, fee = 15, .001
    one["direction"], one["setup"], one["base_r"] = direction, setup, base_r
    return one, max_hold, fee

def simulate(market, symbol, raw):
    sig, max_hold, fee = build_signals(raw, market)
    arr = sig.reset_index()
    trades=[]; active=None; adverse=.0005
    for i in range(250, len(arr)-1):
        row=arr.iloc[i]; nxt=arr.iloc[i+1]
        if active:
            held=i-active["entry_i"]
            hi,lo=float(row.high),float(row.low)
            stop_hit = lo <= active["sl"] if active["side"]=="LONG" else hi >= active["sl"]
            tp_hit = hi >= active["tp"] if active["side"]=="LONG" else lo <= active["tp"]
            if stop_hit or tp_hit or held >= max_hold:
                # Conservative same-candle assumption: stop wins if both touched.
                if stop_hit: exit_price=active["sl"]; reason="SL"
                elif tp_hit: exit_price=active["tp"]; reason="TP"
                else: exit_price=float(row.close); reason="TIME"
                gross=(exit_price-active["entry"])/active["entry"] if active["side"]=="LONG" else (active["entry"]-exit_price)/active["entry"]
                net=gross-2*(fee+adverse)
                active.update(exit_time=str(row.dt),exit_price=exit_price,reason=reason,gross_pct=gross*100,net_pct=net*100,
                              net_usdt=active["notional"]*net,hold_minutes=held)
                trades.append(active); active=None
        if active is None and str(row.direction):
            entry=float(nxt.open)
            atr=float(row["5_ATR"])
            if not np.isfinite(entry) or not np.isfinite(atr) or entry<=0 or atr<=0: continue
            distance=max(.90*atr,entry*.0008); risk=distance/entry
            required=1.5+2.5*(fee+adverse)/risk
            target_r=max(float(row.base_r),required)
            net_reward=target_r*risk-2*(fee+adverse)
            net_rr=net_reward/(risk+2*(fee+adverse))
            if net_reward < .00035 or net_rr < 1.5: continue
            side=str(row.direction)
            sl=entry-distance if side=="LONG" else entry+distance
            tp=entry+target_r*distance if side=="LONG" else entry-target_r*distance
            active={"market":market,"symbol":symbol,"side":side,"setup":str(row.setup),
                    "signal_time":str(row.dt),"entry_time":str(nxt.dt),"entry":entry,"sl":sl,"tp":tp,
                    "target_r":target_r,"net_rr":net_rr,"atr":atr,"entry_i":i+1,"notional":100.0}
    if active:
        row=arr.iloc[-1]; exit_price=float(row.close)
        gross=(exit_price-active["entry"])/active["entry"] if active["side"]=="LONG" else (active["entry"]-exit_price)/active["entry"]
        net=gross-2*(fee+adverse)
        active.update(exit_time=str(row.dt),exit_price=exit_price,reason="END",gross_pct=gross*100,net_pct=net*100,
                      net_usdt=active["notional"]*net,hold_minutes=len(arr)-1-active["entry_i"])
        trades.append(active)
    return trades, arr

def chart_trade(trade, arr, path):
    t=pd.to_datetime(trade["entry_time"],utc=True)
    et=pd.to_datetime(trade["exit_time"],utc=True)
    a=arr.copy(); a["dt"]=pd.to_datetime(a.dt,utc=True)
    subset=a[(a.dt>=t-pd.Timedelta(minutes=35))&(a.dt<=et+pd.Timedelta(minutes=10))].tail(180)
    if subset.empty:return
    fig,ax=plt.subplots(figsize=(13,6))
    ax.plot(subset.dt,subset.close,linewidth=1.0,label="Close")
    ax.fill_between(subset.dt,trade["entry"],trade["tp"],alpha=.10,label="Reward zone")
    ax.fill_between(subset.dt,trade["entry"],trade["sl"],alpha=.10,label="Risk zone")
    ax.axhline(trade["entry"],linestyle="--",label=f'ENTRY {trade["entry"]:.6g}')
    ax.axhline(trade["sl"],linestyle="--",label=f'STOP {trade["sl"]:.6g}')
    ax.axhline(trade["tp"],linestyle="--",label=f'TAKE PROFIT {trade["tp"]:.6g}')
    ax.axhline(trade["exit_price"],linestyle=":",label=f'EXIT {trade["exit_price"]:.6g}')
    ax.axvline(t,linestyle=":",label="Entry time"); ax.axvline(et,linestyle=":",label="Exit time")
    ax.set_title(f'{trade["market"].upper()} {trade["symbol"]} {trade["side"]} | {trade["reason"]} | Net {trade["net_pct"]:.3f}%')
    ax.set_xlabel("UTC"); ax.set_ylabel("Price"); ax.grid(alpha=.2); ax.legend(loc="best",fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path,dpi=130); plt.close(fig)

def main():
    all_trades=[]; summaries=[]
    for market in MARKETS:
        for symbol in SYMBOLS:
            print(f"[RUN] {market.upper()} {symbol} downloading {DAYS} days",flush=True)
            try:
                df=fetch_1m(market,symbol)
                df["dt"]=pd.to_datetime(df.open_time,unit="ms",utc=True)
                raw=df[["dt","open","high","low","close","volume"]].copy()
                trades,arr=simulate(market,symbol,raw)
                all_trades.extend(trades)
                t=pd.DataFrame(trades)
                if len(t):
                    wins=int((t.net_usdt>0).sum()); net=float(t.net_usdt.sum())
                    equity=100+ t.net_usdt.cumsum()
                    peak=equity.cummax(); dd=float(((peak-equity)/peak).max()*100)
                    summary={"market":market,"symbol":symbol,"candles":len(df),"trades":len(t),"wins":wins,
                        "losses":len(t)-wins,"win_rate_pct":round(wins/len(t)*100,2),
                        "net_pnl_usdt_per_100_notional":round(net,4),"profit_factor":round(float(t.loc[t.net_usdt>0,"net_usdt"].sum()/abs(t.loc[t.net_usdt<0,"net_usdt"].sum())),3) if (t.net_usdt<0).any() else None,
                        "max_drawdown_pct":round(dd,3),"avg_net_rr":round(float(t.net_rr.mean()),3)}
                else:
                    summary={"market":market,"symbol":symbol,"candles":len(df),"trades":0,"wins":0,"losses":0,"win_rate_pct":0,"net_pnl_usdt_per_100_notional":0,"profit_factor":None,"max_drawdown_pct":0,"avg_net_rr":0}
                summaries.append(summary); print("[DONE]",summary,flush=True)
                # Up to 5 representative trade charts per market/symbol, spread through the test.
                if trades:
                    indices=np.linspace(0,len(trades)-1,min(5,len(trades)),dtype=int)
                    for j in sorted(set(indices)):
                        tr=trades[j]
                        name=f'{market}_{symbol}_{j+1}_{tr["reason"]}.png'
                        chart_trade(tr,arr,OUT/name)
            except Exception as e:
                summaries.append({"market":market,"symbol":symbol,"error":str(e)})
                print(f"[ERROR] {market} {symbol}: {e}",flush=True)
    pd.DataFrame(summaries).to_csv(OUT/"summary.csv",index=False)
    pd.DataFrame(all_trades).drop(columns=["entry_i"],errors="ignore").to_csv(OUT/"trades.csv",index=False)
    (OUT/"methodology.json").write_text(json.dumps({"strategy":"Scalping V3 vectorized rule-equivalent","days":DAYS,
        "start_utc":datetime.fromtimestamp(START/1000,timezone.utc).isoformat(),
        "end_utc":datetime.fromtimestamp(NOW/1000,timezone.utc).isoformat(),
        "symbols":SYMBOLS,"markets":MARKETS,"fees":{"spot_round_trip":0.002,"futures_round_trip":0.001},
        "adverse_cost_per_side":0.0005,"entry":"next 1m candle open after closed signal candle",
        "intrabar_collision":"SL first (conservative)","position":"one at a time per symbol, fixed 100 USDT notional",
        "charts":"up to 5 representative trades per market/symbol; CSV contains all trades"},indent=2))
    with zipfile.ZipFile("AI-Trader-V3-Backtest-Charts.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in OUT.glob("*.png"): z.write(p,p.name)
    print(f"[FINISHED] trades={len(all_trades)} summaries={len(summaries)}",flush=True)

if __name__=="__main__": main()
