"""Fetch public Binance klines and produce a reproducible research report."""
import json, time, urllib.request
from datetime import datetime, timedelta, timezone
from core.edge_backtest import stress_test

def fetch(symbol, market, days=7):
    base = "https://fapi.binance.com/fapi/v1/klines" if market=="futures" else "https://api.binance.com/api/v3/klines"
    start=int((datetime.now(timezone.utc)-timedelta(days=days)).timestamp()*1000)
    rows=[]
    while True:
        url=f"{base}?symbol={symbol}&interval=5m&startTime={start}&limit=1000"
        req=urllib.request.Request(url,headers={"User-Agent":"edge-research/1.0"})
        with urllib.request.urlopen(req,timeout=30) as response: batch=json.load(response)
        if not batch: break
        rows.extend(batch)
        start=int(batch[-1][0])+1
        if len(batch)<1000 or len(rows)>=days*288: break
        time.sleep(.15)
    out=[]
    for k in rows:
        out.append({"open_time":k[0],"open":float(k[1]),"high":float(k[2]),"low":float(k[3]),"close":float(k[4]),"volume":float(k[5])})
    # exclude the still-forming candle
    now=int(datetime.now(timezone.utc).timestamp()*1000)
    return [x for x in out if x["open_time"]+300000<=now]

def main():
    report={"disclaimer":"Fixed ATR research simulation only. No execution; futures funding not included; no HTF filter; TIMEOUT exits marked-to-market net of costs.","data_days":7,"results":{}}
    for market in ("spot","futures"):
      for symbol in ("BTCUSDT","ETHUSDT"):
        try:
          bars=fetch(symbol,market)
          report["results"][market+"/"+symbol]={"bars":len(bars),"start":bars[0]["open_time"] if bars else None,"end":bars[-1]["open_time"] if bars else None,"stress":stress_test(bars,market=market)}
        except Exception as e:
          report["results"][market+"/"+symbol]={"error":str(e)}
    with open("edge-report.json","w") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
