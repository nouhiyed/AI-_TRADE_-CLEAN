#!/usr/bin/env python3
"""Collect V3 rejection samples from the running server and show which single
gate blocks the most setups ("near misses"). Read-only: it only calls GET
endpoints. Usage (Termux, from ~/AI-Trader):
  python scripts/near_miss.py collect futures     # run every ~5 minutes, or --loop 300
  python scripts/near_miss.py report futures
"""
import argparse, json, sys, time, urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = "http://localhost:8001"
STORE = Path("data/near_miss_{market}.jsonl")
MARK = "V3 diagnostics="


# Gate definitions copied from scripts/diagnose_v3.py so this file is standalone.
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


def parse(reason):
    if not isinstance(reason, str) or MARK not in reason:
        return None
    text = reason.split(MARK, 1)[1]
    try:
        return json.JSONDecoder().raw_decode(text)[0]
    except ValueError:
        return None


def collect(market):
    path = Path(str(STORE).format(market=market)); path.parent.mkdir(exist_ok=True)
    seen = set()
    if path.exists():
        for line in path.open():
            row = json.loads(line); seen.add((row["symbol"], row["key"]))
    url = f"{BASE}/diagnostics/strategy?market={market}&limit=100"
    data = json.load(urllib.request.urlopen(url, timeout=20))
    added = 0
    with path.open("a") as out:
        for rec in data["markets"][market]["symbols"]:
            diag = None
            for value in rec.values():
                diag = parse(value)
                if diag:
                    break
            if not diag:
                continue
            key = rec.get("last_close_time_ms") or rec.get("updated_at_ms")
            if (rec["symbol"], key) in seen:
                continue
            out.write(json.dumps({"symbol": rec["symbol"], "key": key, "diag": diag}) + "\n")
            added += 1
    print(f"added {added} samples -> {path}")


def report(market):
    path = Path(str(STORE).format(market=market))
    rows = [json.loads(x) for x in path.open()]
    print(f"{len(rows)} rejected samples")
    single, double, clause = Counter(), Counter(), Counter()
    for row in rows:
        metrics = row["diag"]["metrics"]
        for setup in row["diag"]["failed_setups"]:
            fails, _, _ = measurable_gate_failures(market, metrics, [setup])
            labels = sorted(k.split(":", 1)[1] for k in fails)
            if len(labels) == 1:
                single[f"{setup}: {labels[0]}"] += 1
            elif len(labels) == 2:
                double[f"{setup}: {' + '.join(labels)}"] += 1
            for label in labels:
                clause[label] += 1
    print("\nSetups failing exactly ONE gate (relax this first):")
    for k, v in single.most_common(12): print(f"  {v:4d}  {k}")
    print("\nSetups failing exactly TWO gates:")
    for k, v in double.most_common(10): print(f"  {v:4d}  {k}")
    print("\nMost frequent failing gates overall:")
    for k, v in clause.most_common(10): print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["collect", "report"])
    ap.add_argument("market", choices=["futures", "spot"])
    ap.add_argument("--loop", type=int, default=0, help="seconds between collections")
    a = ap.parse_args()
    if a.action == "report":
        report(a.market)
    else:
        while True:
            collect(a.market)
            if not a.loop:
                break
            time.sleep(a.loop)
