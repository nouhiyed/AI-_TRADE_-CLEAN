"""Isolated historical research for a 5m trend-continuation-after-pullback setup.

This script is deliberately separate from the production entry pipeline. It
fetches public Binance candles, builds closed-candle indicators, compares the
candidate with the existing V3 strategy, and simulates exits conservatively.
It never places orders and never changes production strategy code.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from core.scalping_v3 import FuturesScalpingStrategyV3, SpotScalpingStrategyV3


BASE_URL = {"spot": "https://data-api.binance.vision", "futures": "https://fapi.binance.com"}
FEE = {"spot": 0.0010, "futures": 0.0005}
ADVERSE = 0.0005
STOP_ATR = 1.15
TARGET_R = 1.60
MAX_HOLD_BARS = 12


def fetch_klines(market: str, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    endpoint = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"
    response = requests.get(
        BASE_URL[market] + endpoint,
        params={"symbol": symbol, "interval": interval, "limit": min(1000, limit)},
        timeout=20,
        headers={"User-Agent": "AI-Trader-Offline-Research/1.0"},
    )
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list) or len(raw) < 300:
        raise RuntimeError(f"Insufficient Binance history for {market} {symbol} {interval}")
    frame = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    now = pd.Timestamp.now(tz="UTC")
    frame = frame[frame["close_time"] < now].copy().reset_index(drop=True)
    return frame


def add_indicators(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    x = frame.copy()
    close, high, low = x["close"], x["high"], x["low"]
    x["EMA_9"] = close.ewm(span=9, adjust=False).mean()
    x["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    x["EMA_21"] = close.ewm(span=21, adjust=False).mean()
    x["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    x["EMA_200"] = close.ewm(span=200, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = (100 - 100 / (1 + rs)).fillna(50)
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / x["ATR"].replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / x["ATR"].replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    x["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    x["MACD_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    mid = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    x["BB_lower"] = mid - 2 * std
    x["BB_upper"] = mid + 2 * std
    x["is_closed"] = True
    x.attrs["timeframe"] = timeframe
    x.attrs["interval"] = timeframe
    return x


def resample_closed(five: pd.DataFrame, timeframe: str, asof: pd.Timestamp) -> pd.DataFrame:
    rule = timeframe
    raw = five.set_index("open_time").resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "close_time": "last",
    }).dropna(subset=["open", "high", "low", "close"])
    # A higher-timeframe candle is usable only after its scheduled close.
    duration = pd.Timedelta(rule)
    raw = raw[(raw.index + duration) <= asof].copy().reset_index()
    if len(raw) < 220:
        raise RuntimeError(f"Not enough closed {timeframe} context")
    return add_indicators(raw, timeframe)


def candidate_signal(five: pd.DataFrame, fifteen: pd.DataFrame,
                     hourly: pd.DataFrame, market: str) -> tuple[str, str] | None:
    """A separate, intentionally testable trend-continuation hypothesis."""
    c = five.iloc[-1]
    p = five.iloc[-2]
    prior = five.iloc[-4:-1]
    atr = float(c["ATR"])
    if not np.isfinite(atr) or atr <= 0:
        return None
    vr = float(c["volume"]) / max(float(five["volume"].iloc[-25:-1].mean()), 1e-12)
    body_atr = abs(float(c["close"])-float(c["open"])) / atr
    span = float(c["high"])-float(c["low"])
    if span <= 0:
        return None
    close_loc = (float(c["close"])-float(c["low"])) / span
    # Pullback must have occurred recently, followed by a fresh 5m resumption.
    touched_ema = bool((prior["low"] <= prior["EMA_20"] + 0.20*prior["ATR"]).any())
    bullish_context = (
        float(fifteen.iloc[-1]["close"]) > float(fifteen.iloc[-1]["EMA_50"])
        and float(hourly.iloc[-1]["close"]) > float(hourly.iloc[-1]["EMA_50"])
    )
    bearish_context = (
        float(fifteen.iloc[-1]["close"]) < float(fifteen.iloc[-1]["EMA_50"])
        and float(hourly.iloc[-1]["close"]) < float(hourly.iloc[-1]["EMA_50"])
    )
    long_ok = (
        bullish_context and touched_ema
        and float(c["close"]) > float(c["open"])
        and float(c["close"]) > float(p["high"])
        and float(c["close"]) > float(c["EMA_9"])
        and 48 <= float(c["RSI"]) <= 68
        and float(c["MACD_hist"]) >= float(p["MACD_hist"])
        and close_loc >= 0.60 and body_atr >= 0.08 and vr >= 0.85
    )
    short_ok = (
        market == "futures" and bearish_context and
        bool((prior["high"] >= prior["EMA_20"] - 0.20*prior["ATR"]).any())
        and float(c["close"]) < float(c["open"])
        and float(c["close"]) < float(p["low"])
        and float(c["close"]) < float(c["EMA_9"])
        and 32 <= float(c["RSI"]) <= 52
        and float(c["MACD_hist"]) <= float(p["MACD_hist"])
        and close_loc <= 0.40 and body_atr >= 0.08 and vr >= 0.85
    )
    if long_ok:
        return "LONG", "TREND_CONTINUATION_PULLBACK_5M"
    if short_ok:
        return "SHORT", "TREND_CONTINUATION_PULLBACK_5M"
    return None


def simulate_exit(frame: pd.DataFrame, signal_index: int, direction: str,
                  entry: float, stop: float, target: float, market: str) -> dict:
    end = min(len(frame)-1, signal_index + MAX_HOLD_BARS)
    exit_price, reason, exit_index = float(frame.iloc[end]["close"]), "TIME", end
    for j in range(signal_index + 1, end + 1):
        bar = frame.iloc[j]
        hit_stop = float(bar["low"]) <= stop if direction == "LONG" else float(bar["high"]) >= stop
        hit_target = float(bar["high"]) >= target if direction == "LONG" else float(bar["low"]) <= target
        # If both levels trade inside one candle, assume the adverse stop happened first.
        if hit_stop:
            exit_price, reason, exit_index = stop, "SL", j
            break
        if hit_target:
            exit_price, reason, exit_index = target, "TP", j
            break
    gross = (exit_price-entry)/entry if direction == "LONG" else (entry-exit_price)/entry
    net = gross - 2*(FEE[market]+ADVERSE)
    risk = abs(entry-stop)/entry + 2*(FEE[market]+ADVERSE)
    return {
        "entry": entry, "exit": exit_price, "reason": reason,
        "bars_held": exit_index-signal_index, "exit_index": exit_index, "net_return_pct": net*100,
        "net_r": net/(risk if risk else 1),
    }


def evaluate_market(market: str, symbol: str, limit: int) -> dict:
    five_raw = fetch_klines(market, symbol, "5m", limit)
    five = add_indicators(five_raw, "5m")
    # Build context strictly from completed 5m bars available at each signal.
    fifteen_raw = fetch_klines(market, symbol, "15m", 1000)
    hourly_raw = fetch_klines(market, symbol, "1h", 1000)
    fifteen = add_indicators(fifteen_raw, "15m")
    hourly = add_indicators(hourly_raw, "1h")
    strategy = FuturesScalpingStrategyV3() if market == "futures" else SpotScalpingStrategyV3()
    baseline, experimental = [], []
    baseline_free_until = -1
    experimental_free_until = -1
    i = 220
    while i < len(five)-MAX_HOLD_BARS-1:
        current = five.iloc[i]
        asof = current["close_time"]
        f15 = fifteen[fifteen["close_time"] < asof].copy()
        f1h = hourly[hourly["close_time"] < asof].copy()
        if len(f15) < 210 or len(f1h) < 210:
            i += 1
            continue
        f5 = five.iloc[:i+1].copy()
        f5.attrs.update({"timeframe": "5m", "interval": "5m"})
        f15.attrs.update({"timeframe": "15m", "interval": "15m"})
        f1h.attrs.update({"timeframe": "1h", "interval": "1h"})
        frames = {"5m": f5, "15m": f15, "1h": f1h}
        data = {**frames, "profile": "standard"}
        next_open = float(five.iloc[i+1]["open"])

        # Existing production V3 is measured without modifying its rules.
        result = strategy.analyze(data) if i > baseline_free_until else {"success": False}
        if result.get("success"):
            direction = result["direction"]
            plan = result
            entry = next_open
            # Preserve the plan's absolute SL/TP distances when entering at next open.
            shift = entry - float(plan["entry"])
            stop = float(plan["stop_loss"]) + shift
            target = float(plan["take_profit"]) + shift
            trade = simulate_exit(five, i, direction, entry, stop, target, market)
            trade["setup"] = plan.get("setup", "V3")
            baseline.append(trade)
            baseline_free_until = trade["exit_index"]

        # Candidate has its own signal, but shares V3's plan/risk/structure guard.
        signal = candidate_signal(f5, f15, f1h, market) if i > experimental_free_until else None
        if signal:
            direction, setup = signal
            plan = strategy.plan_with_structure(
                market, direction, f5, frames, setup,
                "BULL_TREND" if direction == "LONG" else "BEAR_TREND",
                TARGET_R, float(current["close"]), "standard",
            )
            if plan.get("success"):
                entry = next_open
                shift = entry - float(plan["entry"])
                trade = simulate_exit(
                    five, i, direction, entry,
                    float(plan["stop_loss"])+shift,
                    float(plan["take_profit"])+shift, market,
                )
                trade["setup"] = setup
                experimental.append(trade)
                experimental_free_until = trade["exit_index"]
        i += 1

    def stats(trades):
        if not trades:
            return {"trades": 0, "win_rate_pct": 0.0, "net_return_pct_sum": 0.0,
                    "avg_net_r": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0}
        returns = np.array([t["net_return_pct"] for t in trades], dtype=float)
        gains = float(returns[returns > 0].sum())
        losses = abs(float(returns[returns < 0].sum()))
        equity = np.cumprod(1 + returns/100)
        peaks = np.maximum.accumulate(equity)
        dd = (equity/peaks - 1)*100
        return {
            "trades": len(trades),
            "win_rate_pct": round(float((returns > 0).mean()*100), 2),
            "net_return_pct_sum": round(float(returns.sum()), 3),
            "avg_net_r": round(float(np.mean([t["net_r"] for t in trades])), 3),
            "profit_factor": round(gains/losses, 3) if losses else None,
            "max_drawdown_pct": round(abs(float(dd.min())), 3),
        }

    return {
        "market": market, "symbol": symbol,
        "history_start_utc": five_raw.iloc[0]["open_time"].isoformat(),
        "history_end_utc": five_raw.iloc[-1]["close_time"].isoformat(),
        "bars_5m": len(five_raw),
        "assumptions": {
            "entry": "next 5m candle open", "stop": f"{STOP_ATR} ATR",
            "target": f"{TARGET_R}R", "max_hold_bars": MAX_HOLD_BARS,
            "fees_per_side": FEE[market], "adverse_per_side": ADVERSE,
            "same_bar_sl_tp": "stop first", "overlapping_signals": "one open trade per strategy per symbol; new signals skipped until exit",
        },
        "v3_baseline": stats(baseline),
        "trend_continuation_experiment": stats(experimental),
        "sample_trades": experimental[:5],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", choices=("spot", "futures"), default=("spot", "futures"))
    parser.add_argument("--symbols", nargs="+", default=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", default="trend_continuation_research.json")
    args = parser.parse_args()
    results = []
    for market in args.markets:
        for symbol in args.symbols:
            try:
                results.append(evaluate_market(market, symbol, args.limit))
                time.sleep(0.25)
            except Exception as exc:
                results.append({"market": market, "symbol": symbol, "error": str(exc)})
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_research_only_no_orders",
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if not any("v3_baseline" in item for item in results):
        raise SystemExit("No usable backtest results; see errors in report")


if __name__ == "__main__":
    main()
