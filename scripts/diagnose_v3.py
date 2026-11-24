#!/usr/bin/env python3
"""Read-only V3 signal diagnostics. Does not alter strategy rules or trade execution."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import backtest_v3_2y as bt
from core.scalping_v3 import FuturesScalpingStrategyV3, SpotScalpingStrategyV3


def trade_stats(rows):
    values = [float(row["net_R"]) for row in rows]
    wins = [value for value in values if value > 0]
    losses = [-value for value in values if value < 0]
    return {
        "trades": len(values),
        "wins": len(wins),
        "win_rate_pct": round(100 * len(wins) / len(values), 2) if values else 0.0,
        "net_R": round(sum(values), 4),
        "average_R": round(sum(values) / len(values), 4) if values else None,
        "profit_factor": round(sum(wins) / sum(losses), 4) if losses else (None if not wins else "infinite"),
    }



def measurable_gate_failures(market, metrics, failed_setups):
    """Count observable failed clauses on candles where no setup passed.
    Missing metrics are deliberately excluded rather than guessed.
    """
    specs = {
        "PULLBACK_LONG": [
            ("bullish_bias", "bullish_15m_1h_bias", lambda x: bool(x)),
            ("ema20_touch", "low_vs_ema20_atr", lambda x: x <= 0.12),
            ("close_above_ema9", "close_vs_ema9_atr", lambda x: x > 0),
            ("rsi_45_67", "rsi_5m", lambda x: 45 <= x <= 67),
            ("volume_min", "volume_ratio", lambda x: x >= 0.85),
            ("body_min", "body_atr", lambda x: x >= 0.10),
            ("close_location", "close_location", lambda x: x >= 0.62),
            ("lower_wick", "lower_wick_ratio", lambda x: x >= 0.16),
        ],
        "PULLBACK_SHORT": [
            ("bearish_bias", "bearish_15m_1h_bias", lambda x: bool(x)),
            ("ema20_touch", "high_vs_ema20_atr", lambda x: x >= -0.12),
            ("close_below_ema9", "close_vs_ema9_atr", lambda x: x < 0),
            ("rsi_33_55", "rsi_5m", lambda x: 33 <= x <= 55),
            ("volume_min", "volume_ratio", lambda x: x >= 0.85),
            ("body_min", "body_atr", lambda x: x >= 0.10),
            ("close_location", "close_location", lambda x: x <= 0.38),
            ("upper_wick", "upper_wick_ratio", lambda x: x >= 0.16),
        ],
        "BREAKOUT_LONG": [
            ("bullish_bias", "bullish_15m_1h_bias", lambda x: bool(x)),
            ("prior_high_break", "close_vs_prior_high_atr", lambda x: x > 0.05),
            ("rsi_52_74", "rsi_5m", lambda x: 52 <= x <= 74),
            ("close_location", "close_location", lambda x: x >= 0.75),
            ("body_min", "body_atr", lambda x: x >= 0.18),
            ("volume_min", "volume_ratio", lambda x: x >= 1.05),
            ("opposing_wick_max", "upper_wick_ratio", lambda x: x <= 0.18),
        ],
        "BREAKOUT_SHORT": [
            ("bearish_bias", "bearish_15m_1h_bias", lambda x: bool(x)),
            ("prior_low_break", "close_vs_prior_low_atr", lambda x: x < -0.05),
            ("rsi_26_48", "rsi_5m", lambda x: 26 <= x <= 48),
            ("close_location", "close_location", lambda x: x <= 0.25),
            ("body_min", "body_atr", lambda x: x >= 0.18),
            ("volume_min", "volume_ratio", lambda x: x >= 1.05),
            ("opposing_wick_max", "lower_wick_ratio", lambda x: x <= 0.18),
        ],
        "SUPPORT_RECLAIM_LONG": [
            ("bullish_bias", "bullish_15m_1h_bias", lambda x: bool(x)),
            ("close_above_ema9", "close_vs_ema9_atr", lambda x: x > 0),
            ("rsi_35_62", "rsi_5m", lambda x: 35 <= x <= 62),
            ("volume_min", "volume_ratio", lambda x: x >= 0.82),
            ("close_location", "close_location", lambda x: x >= 0.66),
            ("lower_wick", "lower_wick_ratio", lambda x: x >= 0.28),
        ],
        "RANGE_REJECTION_LONG": [
            ("range_adx", "adx_15m", lambda x: x <= 18),
            ("rsi_max", "rsi_5m", lambda x: x <= 30),
            ("volume_min", "volume_ratio", lambda x: x >= 0.80),
            ("close_location", "close_location", lambda x: x >= 0.72),
            ("lower_wick", "lower_wick_ratio", lambda x: x >= 0.35),
        ],
        "RANGE_REJECTION_SHORT": [
            ("range_adx", "adx_15m", lambda x: x <= 18),
            ("rsi_min", "rsi_5m", lambda x: x >= 70),
            ("volume_min", "volume_ratio", lambda x: x >= 0.80),
            ("close_location", "close_location", lambda x: x <= 0.28),
            ("upper_wick", "upper_wick_ratio", lambda x: x >= 0.35),
        ],
    }
    failures, observed, unavailable = Counter(), Counter(), Counter()
    for setup in failed_setups:
        for label, metric, predicate in specs.get(setup, []):
            if metric not in metrics or metrics[metric] is None:
                unavailable[f"{setup}:{label}"] += 1
                continue
            observed[f"{setup}:{label}"] += 1
            if not predicate(metrics[metric]):
                failures[f"{setup}:{label}"] += 1
    return failures, observed, unavailable


def scan_symbol(market, symbol, raw):
    strategy = FuturesScalpingStrategyV3() if market == "futures" else SpotScalpingStrategyV3()
    five, fifteen, hourly, four = bt.build_indicators(raw, market)
    five = five.reset_index(drop=True)
    times = bt.to_ms(five["open_time"])
    closes = times + bt.INTERVAL_MS
    close15, close1h = bt.htf_end_indices(fifteen, 15), bt.htf_end_indices(hourly, 60)
    close4 = bt.htf_end_indices(four, 240) if four is not None else None

    counts = Counter()
    reject_reasons = Counter()
    prefilter_reasons = Counter()
    failed_setups = Counter()
    gate_failures = Counter()
    gate_observations = Counter()
    gate_unavailable = Counter()
    setup_metric_samples = defaultdict(Counter)
    plan_rejects = Counter()
    gate_samples = defaultdict(list)
    successful_plans = []
    for i in range(59, len(five) - 1):
        trigger = bt.set_frame_attrs(five.iloc[i - 59:i + 1].copy(), "5m")
        pre = strategy.prefilter_5m(trigger)
        counts["candles_scanned"] += 1
        counts["prefilter_candidates"] += int(bool(pre.get("candidate")))
        if not pre.get("candidate"):
            prefilter_reasons[pre.get("reason") or "unspecified"] += 1

        close_ms = int(closes[i])
        idx15 = int(pd.Series(close15).searchsorted(close_ms, side="right"))
        idx1h = int(pd.Series(close1h).searchsorted(close_ms, side="right"))
        if idx15 < 60 or idx1h < 60:
            counts["missing_htf_context"] += 1
            continue
        data = {
            "5m": trigger,
            "15m": bt.set_frame_attrs(fifteen.iloc[max(0, idx15 - 60):idx15].copy(), "15m"),
            "1h": bt.set_frame_attrs(hourly.iloc[max(0, idx1h - 60):idx1h].copy(), "1h"),
        }
        if market == "spot" and four is not None:
            idx4 = int(pd.Series(close4).searchsorted(close_ms, side="right"))
            if idx4:
                data["4h"] = bt.set_frame_attrs(four.iloc[max(0, idx4 - 240):idx4].copy(), "4h")

        # Run the authoritative analyzer on every valid closed candle, not only
        # prefilter candidates, so the prefilter can itself be audited.
        plan = strategy.analyze(data)
        if plan.get("success") is True:
            counts["analyzer_plans"] += 1
            setup = str(plan.get("setup", "UNKNOWN"))
            direction = str(plan.get("direction", "UNKNOWN"))
            successful_plans.append({"setup": setup, "direction": direction})
            counts[f"plan:{setup}:{direction}"] += 1
            counts["analyzer_success_without_prefilter"] += int(not pre.get("candidate"))
            next_open = float(five.iloc[i + 1].open)
            counts["next_open_valid"] += int(bt.entry_is_valid(plan, direction, market, next_open))
            counts["next_open_rejected"] += int(not bt.entry_is_valid(plan, direction, market, next_open))
        else:
            reason = str(plan.get("reason", "unspecified"))
            reject_reasons[reason] += 1
            diagnostics = plan.get("diagnostics") or {}
            for setup in diagnostics.get("failed_setups", []):
                failed_setups[setup] += 1
            metrics = diagnostics.get("metrics") or {}
            failed, observed, unavailable = measurable_gate_failures(
                market, metrics, diagnostics.get("failed_setups", []))
            gate_failures.update(failed)
            gate_observations.update(observed)
            gate_unavailable.update(unavailable)
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    gate_samples[key].append(float(value))
            if "structure" in reason.lower():
                plan_rejects["structure"] += 1
            elif "late-entry" in reason.lower() or "entry window" in reason.lower():
                plan_rejects["late_entry"] += 1
            elif "rr" in reason.lower() or "reward" in reason.lower():
                plan_rejects["risk_reward"] += 1
            elif "closed" in reason.lower() or "incomplete" in reason.lower():
                plan_rejects["data_or_context"] += 1
            else:
                plan_rejects["other"] += 1

    # Actual trade outcomes use the untouched replay implementation and its
    # existing entry/exit rules. The scan above is observational only.
    trades, meta = bt.replay_symbol(market, symbol, raw)
    by_setup_direction = defaultdict(list)
    for trade in trades:
        by_setup_direction[f'{trade.get("setup", "UNKNOWN")}|{trade.get("direction", "UNKNOWN")}'].append(trade)
    outcomes = {key: trade_stats(rows) for key, rows in sorted(by_setup_direction.items())}
    metric_summary = {
        key: {
            "observations": len(values),
            "mean": round(sum(values) / len(values), 5),
            "min": round(min(values), 5),
            "max": round(max(values), 5),
        }
        for key, values in sorted(gate_samples.items()) if values
    }
    return {
        "market": market, "symbol": symbol, "scan": dict(counts),
        "prefilter_reasons": dict(prefilter_reasons),
        "analyzer_reject_reasons": dict(reject_reasons),
        "failed_setup_counts": dict(failed_setups),
        "measurable_gate_failures": dict(gate_failures),
        "measurable_gate_observations": dict(gate_observations),
        "unmeasured_gate_observations": dict(gate_unavailable),
        "plan_reject_categories": dict(plan_rejects),
        "rejected_signal_metric_distribution": metric_summary,
        "executed_trade_outcomes_by_setup_direction": outcomes,
        "executed_trade_overall": trade_stats(trades),
        "backtest_metadata": meta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(bt.SYMBOLS))
    parser.add_argument("--markets", default="futures")
    parser.add_argument("--start", default=bt.START_UTC)
    parser.add_argument("--out", default="data/v3_diagnostics")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    start_ms = int(pd.Timestamp(args.start).timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    reports = []
    for market in [item.strip().lower() for item in args.markets.split(",") if item.strip()]:
        if market not in ("futures", "spot"):
            raise ValueError(f"Unsupported market: {market}")
        for symbol in [item.strip().upper() for item in args.symbols.split(",") if item.strip()]:
            print(f"[{market.upper()}] {symbol}: scanning closed 5m candles", flush=True)
            raw = bt.fetch_5m(market, symbol, start_ms, end_ms, args.refresh)
            reports.append(scan_symbol(market, symbol, raw))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.json").write_text(json.dumps(reports, indent=2, default=str))
    flat = []
    for report in reports:
        for key, value in report["executed_trade_outcomes_by_setup_direction"].items():
            flat.append({"market": report["market"], "symbol": report["symbol"],
                         "setup_direction": key, **value})
    pd.DataFrame(flat).to_csv(out / "trade_outcomes_by_setup.csv", index=False)
    pd.DataFrame([{
        "market": r["market"], "symbol": r["symbol"],
        **r["scan"],
        "executed_trades": r["executed_trade_overall"]["trades"],
        "executed_net_R": r["executed_trade_overall"]["net_R"],
        "executed_average_R": r["executed_trade_overall"]["average_R"],
    } for r in reports]).to_csv(out / "scan_summary.csv", index=False)
    print(f"Saved read-only diagnostics to {out}", flush=True)


if __name__ == "__main__":
    main()
