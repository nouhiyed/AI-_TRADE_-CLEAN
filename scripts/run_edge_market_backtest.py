"""Fetch official Binance Vision daily archives and run edge research."""
import csv, io, json, urllib.request, zipfile
from datetime import datetime, timedelta, timezone
from core.edge_backtest import stress_test\nfrom core.edge_portfolio import non_overlapping_replay

COLUMNS = ["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base_volume","taker_buy_quote_volume","ignore"]

def fetch(symbol, market, days=30):
    kind = "spot" if market == "spot" else "futures/um"
    today = datetime.now(timezone.utc).date()
    out = []
    failures = []
    for offset in range(days, 0, -1):
        day = today - timedelta(days=offset)
        ds = day.strftime("%Y-%m-%d")
        name = f"{symbol}-5m-{ds}.zip"
        url = f"https://data.binance.vision/data/{kind}/daily/klines/{symbol}/5m/{name}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"edge-research/1.0"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                payload = resp.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                csvname = next(n for n in archive.namelist() if n.endswith(".csv"))
                text = io.TextIOWrapper(archive.open(csvname), encoding="utf-8")
                reader = csv.reader(text)
                for row in reader:
                    if not row or row[0].lower().startswith("open"):
                        continue
                    if len(row) < 6: continue
                    stamp = int(row[0])
                    # Binance Spot archive timestamps switched to microseconds;
                    # USD-M futures archives remain in milliseconds.
                    if stamp >= 100_000_000_000_000:
                        stamp //= 1000
                    out.append({"open_time":stamp,"open":float(row[1]),"high":float(row[2]),"low":float(row[3]),"close":float(row[4]),"volume":float(row[5])})
        except Exception as exc:
            failures.append({"date":ds,"error":str(exc)})
    # dedupe and sort
    out = sorted({r["open_time"]:r for r in out}.values(),key=lambda r:r["open_time"])
    now=int(datetime.now(timezone.utc).timestamp()*1000)
    out=[r for r in out if r["open_time"]+300000<=now]
    if len(out)<100: raise RuntimeError(f"Insufficient archive bars ({len(out)}); download errors: {failures[-3:]}")
    return out

def main():
    report={"source":"Binance Vision daily kline ZIP archives","data_days":30,"results":{}}
    for market in ("spot","futures"):
      for symbol in ("BTCUSDT","ETHUSDT"):
        try:
          bars=fetch(symbol,market,30)
          report["results"][market+"/"+symbol]={"bars":len(bars),"start":bars[0]["open_time"],"end":bars[-1]["open_time"],"stress":stress_test(bars,market=market),"non_overlap_replay":non_overlapping_replay(bars,market=market)}
        except Exception as e:
          report["results"][market+"/"+symbol]={"error":str(e)}
    with open("edge-report.json","w") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
