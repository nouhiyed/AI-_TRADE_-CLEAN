#!/usr/bin/env python3
"""Research the exact DEMO entry-drift allowance without changing V3 signals."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_v3_2y as bt


def demo_aligned_entry_is_valid(plan, direction, market, entry, counters=None):
    """Mirror core.coordinator._validate_demo_entry_price's 0.30 ATR drift gate."""
    try:
        planned = float(plan["entry"])
        stop = float(plan["stop_loss"])
        target = float(plan["take_profit"])
        atr = float(plan["atr"])
        if not all(np.isfinite(x) and x > 0 for x in (planned, stop, target, atr, entry)):
            if counters is not None:
                counters["invalid_plan"] += 1
            return False
        drift = abs(entry - planned)
        max_drift = max(atr * 0.30, planned * 0.0002)
        if counters is not None:
            counters["checked"] += 1
            counters["drift_atr_sum"] += drift / atr
            counters["max_drift_atr_sum"] += max_drift / atr
        if drift > max_drift:
            if counters is not None:
                counters["rejected_drift"] += 1
            return False
        if direction == "SHORT":
            if not target < entry < stop:
                if counters is not None:
                    counters["rejected_geometry"] += 1
                return False
            risk, reward = stop - entry, entry - target
        else:
            if not stop < entry < target:
                if counters is not None:
                    counters["rejected_geometry"] += 1
                return False
            risk, reward = entry - stop, target - entry
        cost = 2.0 * (bt.FEE[market] + bt.SLIPPAGE[market])
        net_reward = reward / entry - cost
        net_risk = risk / entry + cost
        valid = net_reward > 0 and net_reward / net_risk >= 1.5
        if counters is not None:
            counters["accepted" if valid else "rejected_net_rr"] += 1
        return valid
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        if counters is not None:
            counters["invalid_plan"] += 1
        return False


def stats(rows):
    if not rows:
        return {"trades": 0, "wins": 0, "win_rate_pct": 0.0, "net_R": 0.0,
                "average_R": None, "profit_factor": None, "max_drawdown_R": 0.0}
    return bt.summarize(rows)


def grouped_stats(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(k, "")) for k in keys)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        item = dict(zip(keys, key))
        item.update(stats(values))
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(bt.SYMBOLS))
    parser.add_argument("--start", default=bt.START_UTC)
    parser.add_argument("--out", default="data/v3_demo_alignment")
    args = parser.parse_args()

    start_ms = int(pd.Timestamp(args.start).timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    all_trades = []
    per_symbol = []
    validation = {}

    original_validator = bt.entry_is_valid
    try:
        for symbol in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
            counters = defaultdict(float)
            def validator(plan, direction, market, entry, _counters=counters):
                return demo_aligned_entry_is_valid(
                    plan, direction, market, entry, _counters
                )
            bt.entry_is_valid = validator

            print(f"[FUTURES] {symbol}: DEMO-aligned 0.30 ATR replay", flush=True)
            raw = bt.fetch_5m("futures", symbol, start_ms, end_ms)
            trades, meta = bt.replay_symbol("futures", symbol, raw)
            all_trades.extend(trades)
            row = {"market": "futures", "symbol": symbol, **stats(trades),
                   "candles": meta["candles"], "first_candle": meta["first_candle"],
                   "last_candle": meta["last_candle"],
                   "open_at_end": bool(meta["open_at_end"])}
            per_symbol.append(row)
            validation[symbol] = {
                "checked_signals": int(counters["checked"]),
                "accepted": int(counters["accepted"]),
                "rejected_drift": int(counters["rejected_drift"]),
                "rejected_geometry": int(counters["rejected_geometry"]),
                "rejected_net_rr": int(counters["rejected_net_rr"]),
                "invalid_plan": int(counters["invalid_plan"]),
                "mean_entry_drift_atr": round(
                    counters["drift_atr_sum"] / counters["checked"], 5
                ) if counters["checked"] else None,
            }
            print(f"  trades={len(trades)} accepted={validation[symbol]['accepted']} "
                  f"drift_rejected={validation[symbol]['rejected_drift']}", flush=True)
    finally:
        bt.entry_is_valid = original_validator

    all_trades.sort(key=lambda x: (x.get("exit_time", ""), x.get("symbol", "")))
    pooled = stats(all_trades)
    split_date = pd.Timestamp("2025-11-16T00:00:00Z")
    for trade in all_trades:
        trade["period"] = (
            "OOS_SECOND_YEAR"
            if pd.Timestamp(trade["entry_time"]) >= split_date
            else "IS_FIRST_YEAR"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_trades).to_csv(out / "trades_demo_aligned.csv", index=False)
    pd.DataFrame(per_symbol).to_csv(out / "summary_by_symbol.csv", index=False)
    pd.DataFrame(grouped_stats(all_trades, ["setup", "direction"])).to_csv(
        out / "summary_by_setup.csv", index=False
    )
    pd.DataFrame(grouped_stats(all_trades, ["period"])).to_csv(
        out / "summary_by_period.csv", index=False
    )
    pd.DataFrame(grouped_stats(all_trades, ["symbol", "setup", "direction"])).to_csv(
        out / "summary_by_symbol_setup.csv", index=False
    )

    report = {
        "experiment": "V3 current signals + exact DEMO entry drift allowance",
        "entry_drift_rule": "max(0.30 * ATR, planned_entry * 0.0002)",
        "unchanged": [
            "signal conditions and setup ordering",
            "15m/1h closed-candle context",
            "late-entry and structure guards",
            "net reward/risk >= 1.5 after fees and slippage",
            "initial stop and ATR/structure trailing exit simulation",
        ],
        "baseline_reference_from_read_only_diagnostics": {
            "entry_drift_rule": "max(0.10 * ATR, planned_entry * 0.0002)",
            "trades": 87, "wins": 31, "win_rate_pct": 35.63,
            "net_R": -60.215, "average_R": -0.6921,
            "note": "Reference result from run 45379557941; pooled across independently replayed symbols, not a shared-account portfolio simulation."
        },
        "demo_aligned_pooled_result": pooled,
        "per_symbol_entry_validation": validation,
        "per_symbol_results": per_symbol,
        "period_split": "First year through 2025-11-15 = IS; from 2025-11-16 = OOS second year.",
        "limitations": [
            "Each symbol is replayed independently; simultaneous cross-symbol capital/risk contention is not modeled.",
            "This is a historical DEMO-parity experiment, not proof of future profitability.",
            "No strategy rule is changed and no LIVE execution is enabled."
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print("Saved DEMO-aligned research report to", out, flush=True)
    print(json.dumps(pooled, indent=2), flush=True)


if __name__ == "__main__":
    main()
