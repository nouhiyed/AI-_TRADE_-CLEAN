"""Offline attribution of the production V3 Spot entry gates.

Research-only: reads public Binance candles, never places orders, and never
changes production strategy code. The per-gate predicates mirror Spot V3 so
we can distinguish signal scarcity from downstream plan vetoes.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from core.scalping_v3 import SpotScalpingStrategyV3

BASE = "https://data-api.binance.vision"
INTERVAL_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def fetch(market: str, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    endpoint = "/api/v3/klines"
    start = int(time.time() * 1000) - int(limit) * INTERVAL_MS[interval]
    raw = []
    while len(raw) < limit:
        size = min(1000, limit - len(raw))
        response = requests.get(
            BASE + endpoint,
            params={"symbol": symbol, "interval": interval,
                    "limit": size, "startTime": start},
            timeout=25,
            headers={"User-Agent": "AI-Trader-V3-Gate-Research/1.0"},
        )
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list) or not batch:
            break
        raw.extend(batch)
        next_start = int(batch[-1][6]) + 1
        if next_start <= start or len(batch) < size:
            break
        start = next_start
        time.sleep(0.08)
    if len(raw) < min(300, limit):
        raise RuntimeError(f"Insufficient {interval} history: {len(raw)} bars")
    frame = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    frame = frame[frame["close_time"] < pd.Timestamp.now(tz="UTC")]
    return frame.reset_index(drop=True)


def indicators(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    x = frame.copy()
    c, h, lo = x["close"], x["high"], x["low"]
    for span in (9, 20, 21, 50, 200):
        x[f"EMA_{span}"] = c.ewm(span=span, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = (100 - 100/(1+rs)).fillna(50)
    prev = c.shift(1)
    tr = pd.concat([h-lo, (h-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    fast, slow = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    x["MACD_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    up, down = h.diff(), -lo.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    pdi = 100 * plus.ewm(alpha=1/14, adjust=False).mean() / x["ATR"].replace(0, np.nan)
    mdi = 100 * minus.ewm(alpha=1/14, adjust=False).mean() / x["ATR"].replace(0, np.nan)
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    x["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)
    mid, std = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    x["BB_lower"], x["BB_upper"] = mid-2*std, mid+2*std
    x["is_closed"] = True
    x.attrs["timeframe"] = timeframe
    x.attrs["interval"] = timeframe
    return x


def candle_metrics(frame: pd.DataFrame) -> dict:
    r = frame.iloc[-1]
    span = float(r.high-r.low)
    if span <= 0:
        return {}
    body = abs(float(r.close-r.open))
    return {
        "close": float(r.close), "open": float(r.open), "high": float(r.high),
        "low": float(r.low), "ema9": float(r.EMA_9), "ema20": float(r.EMA_20),
        "atr": float(r.ATR), "rsi": float(r.RSI),
        "hist": float(r.MACD_hist), "prev_hist": float(frame.iloc[-2].MACD_hist),
        "body_atr": body/float(r.ATR) if float(r.ATR) > 0 else 0.0,
        "close_location": (float(r.close-r.low))/span,
        "lower_wick": max(0.0, min(float(r.open), float(r.close))-float(r.low))/span,
        "upper_wick": max(0.0, float(r.high)-max(float(r.open), float(r.close)))/span,
        "volume_ratio": float(r.volume)/max(float(frame.volume.iloc[-25:-1].mean()), 1e-12),
    }


def gate_sets(five: pd.DataFrame, fifteen: pd.DataFrame, hourly: pd.DataFrame) -> dict:
    m = candle_metrics(five)
    if not m or not np.isfinite(m["atr"]) or m["atr"] <= 0:
        return {}
    c, o, h, low = m["close"], m["open"], m["high"], m["low"]
    atr, rsi = m["atr"], m["rsi"]
    hist, prev = m["hist"], m["prev_hist"]
    loc, body, vr = m["close_location"], m["body_atr"], m["volume_ratio"]
    try:
        prior_high = float(five.high.iloc[-13:-1].max())
        prior_low = float(five.low.iloc[-13:-1].min())
    except (ValueError, TypeError):
        return {}
    c1, e200 = float(hourly.close.iloc[-1]), float(hourly.EMA_200.iloc[-1])
    c15, e21_15, e50_15 = (
        float(fifteen.close.iloc[-1]), float(fifteen.EMA_21.iloc[-1]),
        float(fifteen.EMA_50.iloc[-1]),
    )
    e21_5 = float(five.EMA_21.iloc[-1])
    bullish = (c15 > e50_15 and (c >= e21_5 or e21_15 > e50_15)) or (
        c1 > e200 and c15 >= e21_15
    )
    return {
        "SUPPORT_RECLAIM_LONG": {
            "bullish_context": bullish, "support_touch": low <= prior_low + .14*atr,
            "green_candle": c > o, "close_above_ema9": c > m["ema9"],
            "not_extended": c <= m["ema9"] + .45*atr, "close_location": loc >= .66,
            "lower_wick": m["lower_wick"] >= .28, "rsi": 35 <= rsi <= 62,
            "macd_momentum": hist >= prev, "volume": vr >= .82,
        },
        "PULLBACK_LONG": {
            "bullish_context": bullish, "ema20_touch": low <= m["ema20"]+.14*atr,
            "green_candle": c > o, "close_above_ema9": c > m["ema9"],
            "rsi": 44 <= rsi <= 68, "macd_momentum": hist >= prev,
            "close_location": loc >= .62, "body": body >= .10,
            "volume": vr >= .82, "lower_wick": m["lower_wick"] >= .16,
        },
        "BREAKOUT_LONG": {
            "bullish_context": bullish, "breakout": c > prior_high+.05*atr,
            "green_candle": c > o, "rsi": 52 <= rsi <= 74,
            "close_location": loc >= .75, "body": body >= .18,
            "volume": vr >= 1.05, "opposing_wick": m["upper_wick"] <= .18,
        },
        "RANGE_BOUNCE_LONG": {
            "range_regime": float(fifteen.ADX.iloc[-1]) <= 18,
            "lower_band_touch": c <= float(five.BB_lower.iloc[-1])*1.002,
            "oversold_rsi": rsi <= 30, "green_candle": c > o,
            "close_location": loc >= .72, "volume": vr >= .80,
            "lower_wick": m["lower_wick"] >= .35,
        },
    }


def evaluate(symbol: str, limit: int) -> dict:
    five = indicators(fetch("spot", symbol, "5m", limit), "5m")
    fifteen = indicators(fetch("spot", symbol, "15m", max(1000, (limit+2)//3)), "15m")
    hourly = indicators(fetch("spot", symbol, "1h", max(1000, (limit+11)//12)), "1h")
    strategy = SpotScalpingStrategyV3()
    totals = {}
    for setup in ("SUPPORT_RECLAIM_LONG", "PULLBACK_LONG", "BREAKOUT_LONG", "RANGE_BOUNCE_LONG"):
        totals[setup] = {"bars": 0, "individual_pass": Counter(),
                         "sequential_pass": Counter(), "all_gates_pass": 0}
    reasons, actual_success, valid_setups = Counter(), 0, 0
    skipped_context = 0
    for i in range(220, len(five)-1):
        row = five.iloc[i]
        asof = row.close_time
        f15 = fifteen[fifteen.close_time < asof].copy()
        f1h = hourly[hourly.close_time < asof].copy()
        if len(f15) < 60 or len(f1h) < 210:
            skipped_context += 1
            continue
        f5 = five.iloc[:i+1].copy()
        f5.attrs.update({"timeframe": "5m", "interval": "5m"})
        f15.attrs.update({"timeframe": "15m", "interval": "15m"})
        f1h.attrs.update({"timeframe": "1h", "interval": "1h"})
        gates = gate_sets(f5, f15, f1h)
        for name, checks in gates.items():
            item = totals[name]
            item["bars"] += 1
            ordered = list(checks.items())
            passed_so_far = True
            for key, passed in ordered:
                if passed:
                    item["individual_pass"][key] += 1
                passed_so_far = passed_so_far and bool(passed)
                if passed_so_far:
                    item["sequential_pass"][key] += 1
            if all(checks.values()):
                item["all_gates_pass"] += 1
                valid_setups += 1
        result = strategy.analyze({"5m": f5, "15m": f15, "1h": f1h})
        if result.get("success"):
            actual_success += 1
            reasons["V3_SIGNAL_AND_PLAN_SUCCESS"] += 1
        else:
            reasons[str(result.get("reason") or "unspecified")] += 1

    packed = {}
    for name, item in totals.items():
        bars = max(1, item["bars"])
        packed[name] = {
            "bars_evaluated": item["bars"],
            "individual_gate_pass_pct": {
                key: round(value/bars*100, 2)
                for key, value in item["individual_pass"].items()
            },
            "sequential_gate_survival_pct": {
                key: round(value/bars*100, 2)
                for key, value in item["sequential_pass"].items()
            },
            "all_gates_pass_bars": item["all_gates_pass"],
        }
    return {
        "market": "spot", "symbol": symbol,
        "start_utc": five.open_time.iloc[0].isoformat(),
        "end_utc": five.close_time.iloc[-1].isoformat(),
        "five_minute_bars": len(five), "skipped_for_context_warmup": skipped_context,
        "actual_v3_successful_signal_and_plan_count": actual_success,
        "mirrored_gate_candidate_bar_count": valid_setups,
        "v3_rejection_reasons": reasons.most_common(12),
        "setup_gate_attribution": packed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=[
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
        "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT",
    ])
    parser.add_argument("--limit", type=int, default=8640)
    parser.add_argument("--output", default="v3_gate_attribution.json")
    args = parser.parse_args()
    results = []
    for symbol in args.symbols:
        try:
            results.append(evaluate(symbol, args.limit))
        except Exception as error:
            results.append({"market": "spot", "symbol": symbol, "error": str(error)})
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_research_only_no_orders_no_production_changes",
        "note": "Gate predicates are a research mirror of Spot V3; actual V3 decisions are counted separately.",
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if not any("actual_v3_successful_signal_and_plan_count" in x for x in results):
        raise SystemExit("No usable V3 attribution results; inspect per-symbol errors")


if __name__ == "__main__":
    main()
