#!/usr/bin/env python3
"""Read-only V3 entry-quality research. The original replay/exit engine is untouched."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_v3_2y as bt

SPLIT = pd.Timestamp("2025-11-16T00:00:00Z")


def val(frame, name, index=-1):
    return bt.FuturesScalpingStrategyV3.value(frame, name, index)


def safe_ratio(a, b):
    try:
        a, b = float(a), float(b)
        return a / b if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def capture_features(strategy, data, plan, symbol):
    five, fifteen, hourly = (data.get(k) for k in ("5m", "15m", "1h"))
    direction = str(plan.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return None
    sign = 1.0 if direction == "LONG" else -1.0
    c, o, h, lo = (val(five, k) for k in ("close", "open", "high", "low"))
    ema9, ema20, atr = (val(five, k) for k in ("EMA_9", "EMA_20", "ATR"))
    rsi, adx5 = val(five, "RSI"), val(five, "ADX")
    hist, hist_prev = val(five, "MACD_hist"), val(five, "MACD_hist", -2)
    loc = safe_ratio(c - lo, h - lo)
    body_atr = safe_ratio(abs(c - o), atr)
    ext = safe_ratio(sign * (c - ema9), atr)
    ema_stack = safe_ratio(sign * (ema9 - ema20), atr)
    adx15 = val(fifteen, "ADX")
    atr15, atr1h = val(fifteen, "ATR"), val(hourly, "ATR")
    ema21_15, ema50_15 = val(fifteen, "EMA_21"), val(fifteen, "EMA_50")
    ema21_1h, ema50_1h = val(hourly, "EMA_21"), val(hourly, "EMA_50")
    gap15 = safe_ratio(sign * (ema21_15 - ema50_15), atr15)
    gap1h = safe_ratio(sign * (ema21_1h - ema50_1h), atr1h)
    slope15 = safe_ratio(sign * (ema21_15 - val(fifteen, "EMA_21", -4)), atr15)
    slope1h = safe_ratio(sign * (ema21_1h - val(hourly, "EMA_21", -4)), atr1h)
    guard = plan.get("structure_guard") or {}
    clearance = guard.get("distance_atr")
    if clearance is None and guard.get("distance") is not None:
        clearance = safe_ratio(guard.get("distance"), atr)
    signal_time = pd.to_datetime(five["open_time"].iloc[-1], utc=True) + pd.Timedelta(minutes=5)
    return {
        "symbol": symbol, "entry_time": signal_time.isoformat(),
        "direction": direction, "setup": plan.get("setup", ""),
        "rsi_directional": rsi if direction == "LONG" else (100 - rsi if rsi is not None else np.nan),
        "adx5": adx5, "adx15": adx15, "volume_ratio": strategy.volume_ratio(five),
        "body_atr": body_atr,
        "close_location_directional": loc if direction == "LONG" else (1 - loc if np.isfinite(loc) else np.nan),
        "macd_momentum_atr": safe_ratio(sign * (hist - hist_prev), atr),
        "ema9_extension_atr": ext, "ema9_20_alignment_atr": ema_stack,
        "ema21_50_gap_15m_atr": gap15, "ema21_50_gap_1h_atr": gap1h,
        "ema21_slope_15m_atr": slope15, "ema21_slope_1h_atr": slope1h,
        "structure_clearance_atr": clearance,
    }


def metrics(frame):
    if len(frame) == 0:
        return {"trades": 0, "wins": 0, "win_rate_pct": 0.0, "net_R": 0.0,
                "average_R": None, "profit_factor": None}
    values = frame.net_R.astype(float)
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    return {"trades": int(len(values)), "wins": int((values > 0).sum()),
            "win_rate_pct": round(float((values > 0).mean() * 100), 2),
            "net_R": round(float(values.sum()), 3), "average_R": round(float(values.mean()), 4),
            "profit_factor": round(float(gains / losses), 3) if losses > 0 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(bt.SYMBOLS))
    parser.add_argument("--start", default=bt.START_UTC)
    parser.add_argument("--out", default="data/v3_entry_quality")
    args = parser.parse_args()

    original_analyze = bt.FuturesScalpingStrategyV3.analyze
    original_validator = bt.entry_is_valid
    all_trades = []
    current_symbol = [None]
    feature_rows = []

    def wrapped_analyze(self, data, costs=None):
        plan = original_analyze(self, data, costs)
        if plan.get("success") is True:
            feature = capture_features(self, data, plan, current_symbol[0])
            if feature:
                feature_rows.append(feature)
        return plan

    def demo_entry_is_valid(plan, direction, market, entry):
        try:
            planned, stop, target, atr = map(float, (
                plan["entry"], plan["stop_loss"], plan["take_profit"], plan["atr"]))
            if min(planned, stop, target, atr, float(entry)) <= 0:
                return False
            if abs(float(entry) - planned) > max(atr * 0.30, planned * 0.0002):
                return False
            if direction == "SHORT":
                if not target < entry < stop:
                    return False
                risk, reward = stop - entry, entry - target
            else:
                if not stop < entry < target:
                    return False
                risk, reward = entry - stop, target - entry
            cost = 2.0 * (bt.FEE[market] + bt.SLIPPAGE[market])
            net_reward, net_risk = reward / entry - cost, risk / entry + cost
            return net_reward > 0 and net_reward / net_risk >= 1.5
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False

    bt.FuturesScalpingStrategyV3.analyze = wrapped_analyze
    bt.entry_is_valid = demo_entry_is_valid
    try:
        start_ms = int(pd.Timestamp(args.start).timestamp() * 1000)
        end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        for symbol in [x.strip().upper() for x in args.symbols.split(",") if x.strip()]:
            current_symbol[0] = symbol
            feature_rows = []
            raw = bt.fetch_5m("futures", symbol, start_ms, end_ms)
            trades, meta = bt.replay_symbol("futures", symbol, raw)
            feature_map = {row["entry_time"]: row for row in feature_rows}
            matched = 0
            for trade in trades:
                feature = feature_map.get(pd.Timestamp(trade["entry_time"]).isoformat())
                if feature:
                    matched += 1
                    trade.update({k: v for k, v in feature.items()
                                  if k not in ("symbol", "entry_time", "direction", "setup")})
                else:
                    trade["feature_capture_missing"] = True
            all_trades.extend(trades)
            print(f"{symbol}: trades={len(trades)} features={len(feature_rows)} matched={matched}", flush=True)
    finally:
        bt.FuturesScalpingStrategyV3.analyze = original_analyze
        bt.entry_is_valid = original_validator

    frame = pd.DataFrame(all_trades)
    frame["entry_time"] = pd.to_datetime(frame.entry_time, utc=True)
    frame["period"] = np.where(frame.entry_time >= SPLIT, "OOS_SECOND_YEAR", "IS_FIRST_YEAR")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "entry_quality_trades.csv", index=False)

    features = [
        "rsi_directional", "adx5", "adx15", "volume_ratio", "body_atr",
        "close_location_directional", "macd_momentum_atr", "ema9_extension_atr",
        "ema9_20_alignment_atr", "ema21_50_gap_15m_atr", "ema21_50_gap_1h_atr",
        "ema21_slope_15m_atr", "ema21_slope_1h_atr", "structure_clearance_atr",
    ]
    distributions = []
    for feature in features:
        for period in ("IS_FIRST_YEAR", "OOS_SECOND_YEAR"):
            for outcome, subset in (("WIN", frame[frame.net_R > 0]), ("LOSS", frame[frame.net_R <= 0])):
                values = subset[subset.period == period][feature].dropna()
                if len(values):
                    distributions.append({
                        "feature": feature, "period": period, "outcome": outcome, "n": len(values),
                        "mean": round(float(values.mean()), 4), "median": round(float(values.median()), 4),
                        "q25": round(float(values.quantile(.25)), 4),
                        "q75": round(float(values.quantile(.75)), 4),
                    })
    pd.DataFrame(distributions).to_csv(output / "feature_distributions.csv", index=False)

    # Thresholds are derived only from IS. Each one-factor filter is frozen before OOS evaluation.
    sweep = []
    train = frame[frame.period == "IS_FIRST_YEAR"]
    for feature in features:
        values = train[feature].dropna()
        if len(values) < 20:
            continue
        for quantile in (.25, .40, .50, .60, .75):
            threshold = float(values.quantile(quantile))
            for operator in (">=", "<="):
                mask = frame[feature] >= threshold if operator == ">=" else frame[feature] <= threshold
                row = {"feature": feature, "operator": operator, "threshold": round(threshold, 6)}
                for period in ("IS_FIRST_YEAR", "OOS_SECOND_YEAR"):
                    row.update({f"{period}_{key}": value
                                for key, value in metrics(frame[(frame.period == period) & mask]).items()})
                sweep.append(row)
    sweep_frame = pd.DataFrame(sweep)
    sweep_frame.to_csv(output / "filter_sweep.csv", index=False)

    selected = []
    for feature, group in sweep_frame.groupby("feature"):
        eligible = group[group.IS_FIRST_YEAR_trades >= 15].copy()
        if len(eligible):
            best = eligible.sort_values(
                ["IS_FIRST_YEAR_average_R", "IS_FIRST_YEAR_net_R"], ascending=False
            ).iloc[0]
            selected.append(best.to_dict())
    pd.DataFrame(selected).to_csv(output / "is_selected_filters_oos.csv", index=False)

    report = {
        "experiment": "Entry-quality feature diagnostics and IS-only one-factor threshold selection",
        "baseline_all": metrics(frame),
        "IS": metrics(frame[frame.period == "IS_FIRST_YEAR"]),
        "OOS": metrics(frame[frame.period == "OOS_SECOND_YEAR"]),
        "matched_feature_rows": int(frame.get(
            "feature_capture_missing", pd.Series(False, index=frame.index)).eq(False).sum()),
        "trades": len(frame),
        "exit_engine_unchanged": True,
        "exit_engine_note": "Original bt.replay_symbol is called without modification. Initial stop, +1R activation, trailing calculation, cost-aware floor, candle order, and update sequence are untouched.",
        "entry_validation": "Same 0.30 ATR / 0.02% drift and net RR >= 1.5 as the prior entry-alignment replay.",
        "selection_protocol": "Thresholds computed only from first-year IS feature quantiles; each one-factor filter requires at least 15 IS trades. OOS is frozen second-year evaluation. Diagnostic only, not production approval.",
        "features": features,
        "scope": "Eight Futures symbols, independent sequential per-symbol replays, not portfolio simulation.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
