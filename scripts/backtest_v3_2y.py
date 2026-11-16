#!/usr/bin/env python3
"""Two-year historical replay of the exact in-app V3 Futures/Spot strategies.

Entries: confirmed 5m close signal, next 5m candle open, with the same
0.10-ATR / 0.02%-of-price drift and net-RR validation used by DEMO.
Exits: initial strategy stop, then the app's +1R activation and shared
ATR(14)*1.5 + five-candle structure trailing stop. No fixed TP and no time exit.

The replay uses completed Binance 5m candles and resamples only completed
15m/1h/4h candles, avoiding higher-timeframe lookahead.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core.scalping_v3 import FuturesScalpingStrategyV3, SpotScalpingStrategyV3
from core.technical import analyze as analyze_technical
from core.trailing_stop import (
    apply_cost_aware_trailing_floor,
    calculate_trailing_stop,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
           "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"]
INTERVAL_MS = 5 * 60 * 1000
START_UTC = "2024-11-16T00:00:00+00:00"
CACHE = Path("data/backtest_cache")
OUT = Path("data/backtest_results")
BASE = {"spot": "https://api.binance.com", "futures": "https://fapi.binance.com"}
FEE = {"spot": 0.0010, "futures": 0.0005}
SLIPPAGE = {"spot": 0.0005, "futures": 0.0005}


def fetch_5m(market: str, symbol: str, start_ms: int, end_ms: int,
             refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{market}_{symbol}_5m.csv.gz"
    if path.exists() and not refresh:
        frame = pd.read_csv(path)
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        return frame[(frame["open_time"].astype("int64") // 1_000_000 >= start_ms)
                     & (frame["open_time"].astype("int64") // 1_000_000 < end_ms)].reset_index(drop=True)

    endpoint = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"
    session = requests.Session()
    session.headers.update({"User-Agent": "AI-Trader-V3-Backtest/1.0"})
    rows, cursor = [], start_ms
    last_report = time.monotonic()
    while cursor < end_ms:
        response = session.get(
            BASE[market] + endpoint,
            params={"symbol": symbol, "interval": "5m", "startTime": cursor,
                    "endTime": end_ms - 1, "limit": 1000},
            timeout=25,
        )
        if response.status_code in (418, 429):
            raise RuntimeError(f"Binance rate limit for {market} {symbol}: HTTP {response.status_code}")
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + INTERVAL_MS
        if next_cursor <= cursor:
            raise RuntimeError("Binance pagination did not advance")
        cursor = next_cursor
        if time.monotonic() - last_report > 20:
            print(f"  downloading {market} {symbol}: {len(rows):,} candles", flush=True)
            last_report = time.monotonic()
        time.sleep(0.18)

    if not rows:
        raise RuntimeError(f"No Binance 5m history returned for {market} {symbol}")
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"]
    frame = pd.DataFrame(rows, columns=cols)
    frame = frame[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame.dropna().drop_duplicates("open_time").sort_values("open_time")
    frame = frame[(frame["open_time"].astype("int64") // 1_000_000 >= start_ms)
                  & (frame["close_time"].astype("int64") // 1_000_000 < end_ms)]
    frame.to_csv(path, index=False, compression="gzip")
    return frame.reset_index(drop=True)


def prepare(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    result = analyze_technical(frame, timeframe=timeframe)
    result["is_closed"] = True
    result.attrs["timeframe"] = timeframe
    result.attrs["interval"] = timeframe
    return result


def resample_ohlcv(raw: pd.DataFrame, minutes: int) -> pd.DataFrame:
    indexed = raw.set_index("open_time")
    rule = f"{minutes}min"
    grouped = indexed.resample(rule, label="left", closed="left")
    result = grouped.agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        count=("close", "count"),
    )
    expected = minutes // 5
    result = result[result["count"] == expected].drop(columns=["count"]).dropna()
    result = result.reset_index()
    result["is_closed"] = True
    return result


def set_frame_attrs(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    frame.attrs["timeframe"] = timeframe
    frame.attrs["interval"] = timeframe
    return frame


def htf_end_indices(frame: pd.DataFrame, minutes: int) -> np.ndarray:
    opens = pd.to_datetime(frame["open_time"], utc=True).astype("int64").to_numpy() // 1_000_000
    closes = opens + minutes * 60_000
    return closes


def entry_is_valid(plan: dict, direction: str, market: str,
                   entry: float) -> bool:
    try:
        planned = float(plan["entry"])
        stop = float(plan["stop_loss"])
        target = float(plan["take_profit"])
        atr = float(plan["atr"])
        if not all(np.isfinite(x) and x > 0 for x in (planned, stop, target, atr, entry)):
            return False
        max_drift = max(atr * 0.10, planned * 0.0002)
        if abs(entry - planned) > max_drift:
            return False
        if direction == "SHORT":
            if not target < entry < stop:
                return False
            risk, reward = stop - entry, entry - target
        else:
            if not stop < entry < target:
                return False
            risk, reward = entry - stop, target - entry
        cost = 2.0 * (FEE[market] + SLIPPAGE[market])
        net_reward = reward / entry - cost
        net_risk = risk / entry + cost
        return net_reward > 0 and net_reward / net_risk >= 1.5
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def build_indicators(raw: pd.DataFrame, market: str):
    five = prepare(raw, "5m")
    fifteen = prepare(resample_ohlcv(raw, 15), "15m")
    hourly = prepare(resample_ohlcv(raw, 60), "1h")
    four = prepare(resample_ohlcv(raw, 240), "4h") if market == "spot" else None
    return five, fifteen, hourly, four


def replay_symbol(market: str, symbol: str, raw: pd.DataFrame) -> tuple[list, dict]:
    strategy = FuturesScalpingStrategyV3() if market == "futures" else SpotScalpingStrategyV3()
    five, fifteen, hourly, four = build_indicators(raw, market)
    five = five.reset_index(drop=True)
    times = pd.to_datetime(five["open_time"], utc=True).astype("int64").to_numpy() // 1_000_000
    closes = times + INTERVAL_MS
    fifteen_close = htf_end_indices(fifteen, 15)
    hourly_close = htf_end_indices(hourly, 60)
    four_close = htf_end_indices(four, 240) if four is not None else None

    trades = []
    position = None
    last_entry_index = -1
    # The next 5m open is the only permitted execution point.
    for i in range(59, len(five) - 1):
        bar = five.iloc[i]
        if position is not None:
            j = i
            row = five.iloc[j]
            o, h, lo, c = map(float, (row.open, row.high, row.low, row.close))
            direction = position["direction"]
            stop = position["stop"]
            reason = None
            raw_exit = None

            # Existing stop has priority if the candle also reaches +1R.
            if direction == "LONG":
                if o <= stop:
                    reason, raw_exit = ("TRAILING_STOP" if position["armed"] else "STOP_LOSS"), o
                elif lo <= stop:
                    reason, raw_exit = ("TRAILING_STOP" if position["armed"] else "STOP_LOSS"), stop
                elif not position["armed"] and h >= position["entry"] + position["initial_risk"]:
                    position["armed"] = True
                    position["best"] = h
                    hist = five.iloc[max(0, j - 40):j]
                    update = calculate_trailing_stop(
                        hist, "LONG", position["stop"], position["best"], h)
                    if update:
                        candidate = update["stop"]
                        position["stop"] = apply_cost_aware_trailing_floor(
                            candidate, position["entry"], "LONG", update["best_price"],
                            FEE[market], SLIPPAGE[market])
                        position["best"] = update["best_price"]
                    if lo <= position["stop"]:
                        reason, raw_exit = "TRAILING_STOP", position["stop"]
                elif position["armed"] and h > position["best"]:
                    position["best"] = h
                    hist = five.iloc[max(0, j - 40):j]
                    update = calculate_trailing_stop(
                        hist, "LONG", position["stop"], position["best"], h)
                    if update:
                        candidate = update["stop"]
                        position["stop"] = apply_cost_aware_trailing_floor(
                            candidate, position["entry"], "LONG", update["best_price"],
                            FEE[market], SLIPPAGE[market])
                        position["best"] = update["best_price"]
                    if lo <= position["stop"]:
                        reason, raw_exit = "TRAILING_STOP", position["stop"]
            else:
                if o >= stop:
                    reason, raw_exit = ("TRAILING_STOP" if position["armed"] else "STOP_LOSS"), o
                elif h >= stop:
                    reason, raw_exit = ("TRAILING_STOP" if position["armed"] else "STOP_LOSS"), stop
                elif not position["armed"] and lo <= position["entry"] - position["initial_risk"]:
                    position["armed"] = True
                    position["best"] = lo
                    hist = five.iloc[max(0, j - 40):j]
                    update = calculate_trailing_stop(
                        hist, "SHORT", position["stop"], position["best"], lo)
                    if update:
                        candidate = update["stop"]
                        position["stop"] = apply_cost_aware_trailing_floor(
                            candidate, position["entry"], "SHORT", update["best_price"],
                            FEE[market], SLIPPAGE[market])
                        position["best"] = update["best_price"]
                    if h >= position["stop"]:
                        reason, raw_exit = "TRAILING_STOP", position["stop"]
                elif position["armed"] and lo < position["best"]:
                    position["best"] = lo
                    hist = five.iloc[max(0, j - 40):j]
                    update = calculate_trailing_stop(
                        hist, "SHORT", position["stop"], position["best"], lo)
                    if update:
                        candidate = update["stop"]
                        position["stop"] = apply_cost_aware_trailing_floor(
                            candidate, position["entry"], "SHORT", update["best_price"],
                            FEE[market], SLIPPAGE[market])
                        position["best"] = update["best_price"]
                    if h >= position["stop"]:
                        reason, raw_exit = "TRAILING_STOP", position["stop"]

            # At candle close, refresh the trailing level from the now-confirmed
            # candle for the next bar, exactly as the live monitor uses closed 5m data.
            if reason is None and position["armed"]:
                hist = five.iloc[max(0, j - 39):j + 1]
                update = calculate_trailing_stop(
                    hist, direction, position["stop"], position["best"], c)
                if update:
                    position["stop"] = apply_cost_aware_trailing_floor(
                        update["stop"], position["entry"], direction,
                        update["best_price"], FEE[market], SLIPPAGE[market])
                    position["best"] = update["best_price"]

            if reason:
                # Exit slippage is adverse; fee charged on both entry and exit.
                exit_price = raw_exit * (1 - SLIPPAGE[market]) if direction == "LONG" else raw_exit * (1 + SLIPPAGE[market])
                entry_price = position["entry"]
                qty = 1.0 / position["initial_risk"]
                gross = (exit_price - entry_price) * qty if direction == "LONG" else (entry_price - exit_price) * qty
                fees = (entry_price * qty + exit_price * qty) * FEE[market]
                net = gross - fees
                trades.append({
                    "market": market, "symbol": symbol, "direction": direction,
                    "setup": position["setup"],
                    "entry_time": position["entry_time"],
                    "exit_time": pd.Timestamp(int(closes[j]), unit="ms", tz="UTC").isoformat(),
                    "entry_price": entry_price, "exit_price": exit_price,
                    "initial_stop": position["initial_stop"], "final_stop": position["stop"],
                    "exit_reason": reason, "bars_held": j - position["entry_index"] + 1,
                    "net_pnl_per_1R": net, "net_R": net,
                    "fees_in_R": fees,
                })
                position = None
                continue

        if position is not None or i <= last_entry_index:
            continue

        # Closed 5m prefilter is an exact fast gate from the app. Only candidates
        # proceed to the authoritative V3 strategy, preserving all HTF conditions.
        trigger = set_frame_attrs(five.iloc[i - 59:i + 1].copy(), "5m")
        candidate = strategy.prefilter_5m(trigger)
        if not candidate.get("candidate"):
            continue

        close_ms = int(closes[i])
        idx15 = int(np.searchsorted(fifteen_close, close_ms, side="right"))
        idx1h = int(np.searchsorted(hourly_close, close_ms, side="right"))
        if idx15 < 60 or idx1h < 60:
            continue
        data = {
            "5m": trigger,
            "15m": set_frame_attrs(fifteen.iloc[max(0, idx15 - 60):idx15].copy(), "15m"),
            "1h": set_frame_attrs(hourly.iloc[max(0, idx1h - 60):idx1h].copy(), "1h"),
        }
        if market == "spot" and four is not None:
            idx4h = int(np.searchsorted(four_close, close_ms, side="right"))
            if idx4h:
                data["4h"] = set_frame_attrs(four.iloc[max(0, idx4h - 240):idx4h].copy(), "4h")

        plan = strategy.analyze(data)
        if plan.get("success") is not True:
            continue

        direction = str(plan["direction"]).upper()
        if market == "spot" and direction != "LONG":
            continue
        next_row = five.iloc[i + 1]
        raw_entry = float(next_row.open)
        entry = raw_entry * (1 + SLIPPAGE[market]) if direction == "LONG" else raw_entry * (1 - SLIPPAGE[market])
        # DEMO validates the raw next-candle open before the execution engine applies slippage.
        if not entry_is_valid(plan, direction, market, raw_entry):
            continue
        initial_stop = float(plan["stop_loss"])
        initial_risk = abs(entry - initial_stop)
        if initial_risk <= 0 or (direction == "LONG" and initial_stop >= entry) or (direction == "SHORT" and initial_stop <= entry):
            continue

        position = {
            "direction": direction, "entry": entry, "stop": initial_stop,
            "initial_stop": initial_stop, "initial_risk": initial_risk,
            "armed": False, "best": entry, "setup": plan.get("setup", ""),
            "entry_index": i + 1,
            "entry_time": pd.Timestamp(int(times[i + 1]), unit="ms", tz="UTC").isoformat(),
        }
        last_entry_index = i + 1

    open_at_end = None
    if position is not None:
        last = five.iloc[-1]
        mark = float(last.close)
        direction = position["direction"]
        mark_exit = mark * (1 - SLIPPAGE[market]) if direction == "LONG" else mark * (1 + SLIPPAGE[market])
        qty = 1.0 / position["initial_risk"]
        gross = (mark_exit - position["entry"]) * qty if direction == "LONG" else (position["entry"] - mark_exit) * qty
        fees = (position["entry"] * qty + mark_exit * qty) * FEE[market]
        open_at_end = {
            "market": market, "symbol": symbol, "direction": direction,
            "entry_time": position["entry_time"], "entry_price": position["entry"],
            "mark_price": mark, "stop_price": position["stop"],
            "trailing_armed": position["armed"], "unrealized_net_R": gross - fees,
        }
    meta = {
        "market": market, "symbol": symbol, "candles": len(raw),
        "first_candle": raw.iloc[0]["open_time"].isoformat(),
        "last_candle": raw.iloc[-1]["open_time"].isoformat(),
        "trades": len(trades), "open_at_end": open_at_end,
    }
    return trades, meta


def summarize(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0,
                "net_R": 0.0, "profit_factor": None, "max_drawdown_R": 0.0}
    values = np.array([float(t["net_R"]) for t in trades], dtype=float)
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    drawdown = peaks[1:] - equity
    return {
        "trades": int(len(values)), "wins": int((values > 0).sum()),
        "losses": int((values <= 0).sum()),
        "win_rate_pct": round(float((values > 0).mean() * 100), 2),
        "net_R": round(float(values.sum()), 3),
        "profit_factor": round(gains / losses, 3) if losses > 0 else None,
        "average_R": round(float(values.mean()), 4),
        "max_drawdown_R": round(float(drawdown.max()) if len(drawdown) else 0.0, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="AI-Trader V3 two-year trailing-exit backtest")
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    parser.add_argument("--markets", default="futures,spot")
    parser.add_argument("--start", default=START_UTC)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    start_ms = int(pd.Timestamp(args.start).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    markets = [x.strip().lower() for x in args.markets.split(",") if x.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    all_trades, metadata, summaries = [], [], []
    start_label = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date().isoformat()
    end_label = datetime.now(timezone.utc).date().isoformat()
    print(f"AI-Trader V3 | {start_label} -> {end_label} UTC | {symbols}", flush=True)
    print("Entry: V3 5m close -> next 5m open; exits: initial SL + ATR/structure trailing; no fixed TP/time exit.", flush=True)

    for market in markets:
        if market not in ("futures", "spot"):
            raise ValueError(f"Unsupported market: {market}")
        market_trades = []
        for symbol in symbols:
            print(f"\n[{market.upper()}] {symbol}: downloading/reading 5m candles", flush=True)
            raw = fetch_5m(market, symbol, start_ms, end_ms, args.refresh)
            trades, meta = replay_symbol(market, symbol, raw)
            market_trades.extend(trades)
            metadata.append(meta)
            print(f"  candles={meta['candles']:,} trades={meta['trades']}", flush=True)
            if meta["open_at_end"]:
                print(f"  still open at end; unrealized={meta['open_at_end']['unrealized_net_R']:.3f}R (not force-closed)", flush=True)
        market_trades.sort(key=lambda x: x["exit_time"])
        all_trades.extend(market_trades)
        summaries.append({"market": market, **summarize(market_trades)})
        for symbol in symbols:
            one = [t for t in market_trades if t["symbol"] == symbol]
            summaries.append({"market": market, "symbol": symbol, **summarize(one)})

    pd.DataFrame(all_trades).to_csv(OUT / "trades.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print("\n=== SUMMARY ===")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nSaved: {OUT / 'summary.csv'}, {OUT / 'trades.csv'}, {OUT / 'metadata.json'}")
    print("Note: results are trade-level 1R-normalized, per-symbol sequential replays; portfolio-wide risk caps/3-position concurrency are not simulated.")


if __name__ == "__main__":
    main()
