#!/usr/bin/env python3
"""Methodical, leak-resistant relaxation sweep for the V3 5m strategy.

Why this exists
---------------
Relaxing entry conditions always produces *more* trades. Whether those extra
trades are *worth taking* is a statistical question that a single backtest
cannot answer honestly, because trying many settings and keeping the best one
manufactures edge out of noise (selection bias / multiple testing).

Protocol enforced here
----------------------
1. One factor at a time, from the live baseline, in small steps.
2. Chronological split of the history: DESIGN (first 50%) | VALIDATION
   (next 25%) | HOLDOUT (last 25%). Holdout statistics are written to a sealed
   file and are only shown through ``--show-holdout`` (access is logged).
3. A relaxation is ACCEPTED only if, against the baseline:
     - it adds at least 25% more trades on DESIGN, and
     - average R is not worse by more than 0.03R on DESIGN and on VALIDATION, and
     - the *marginal* trades (those the baseline would not have taken) are not
       worse than the baseline by more than 0.03R (design + validation).
4. "Edge" is a separate, stricter claim: validation mean R must be positive with
   a one-sided t-test that is Bonferroni-adjusted for every configuration tried.
5. ``--random-baseline`` replays random-direction entries with the same stop
   distance, trailing exit and costs, to show how much of the result comes from
   the entry signal rather than from exits/costs.

Every trade is normalised to R (1R = initial stop distance) and costs (fees +
slippage both ways) are included by the shared replay engine in
``backtest_v3_2y.py``. Trades are per-symbol sequential; portfolio concurrency,
margin and cooldowns are NOT simulated, so absolute numbers are optimistic.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import backtest_v3_2y as bt  # noqa: E402
from core.scalping_v3 import FuturesScalpingStrategyV3, SpotScalpingStrategyV3  # noqa: E402

STRATEGY_CLASS = {"futures": ("FuturesScalpingStrategyV3", FuturesScalpingStrategyV3),
                  "spot": ("SpotScalpingStrategyV3", SpotScalpingStrategyV3)}

# Baseline first, then progressively more relaxed levels (small steps).
GRID = {
    "PULLBACK_VOLUME": [0.85, 0.75, 0.65, 0.55],
    "BREAKOUT_VOLUME": [1.05, 0.95, 0.85, 0.75],
    "PULLBACK_CLOSE_LOCATION": [0.62, 0.57, 0.52],
    "BREAKOUT_CLOSE_LOCATION": [0.75, 0.70, 0.65],
    "PULLBACK_BODY_ATR": [0.10, 0.06, 0.03],
    "BREAKOUT_BODY_ATR": [0.18, 0.12, 0.08],
    "BREAKOUT_MAX_OPPOSING_WICK": [0.18, 0.25, 0.32],
}
SPLITS = (0.50, 0.75)  # design | validation | holdout
MIN_TRADE_GAIN = 1.25
NON_INFERIORITY_R = 0.03
MIN_VALIDATION_TRADES = 30
MIN_MARGINAL_TRADES = 20
ALPHA = 0.05
RANDOM_RATE = 0.004  # probability a flat bar becomes a random-entry candidate
_MISSING = object()


# ----------------------------------------------------------------- helpers
class patched:
    """Temporarily override class attributes (restores even on error)."""

    def __init__(self, cls, values: dict):
        self.cls, self.values, self.old = cls, values, {}

    def __enter__(self):
        for key, value in self.values.items():
            self.old[key] = self.cls.__dict__.get(key, _MISSING)
            setattr(self.cls, key, value)
        return self

    def __exit__(self, *exc):
        for key, value in self.old.items():
            if value is _MISSING:
                delattr(self.cls, key)
            else:
                setattr(self.cls, key, value)


def config_id(params: dict) -> str:
    return "baseline" if not params else ",".join(f"{k}={v}" for k, v in sorted(params.items()))


def make_random_class(base_cls, rate: float, seed: int):
    rng = np.random.default_rng(seed)

    class RandomEntry(base_cls):
        @classmethod
        def prefilter_5m(cls, frame):
            return {"candidate": bool(rng.random() < rate), "setups": ["RANDOM"]}

        def analyze(self, data, costs=None):
            frame = data["5m"]
            close = float(frame["close"].iloc[-1])
            direction = "LONG" if rng.random() < 0.5 else "SHORT"
            # Same stop/target/cost construction as the real strategy (no structure veto).
            return self.plan(self.MARKET, direction, frame, "RANDOM", "RANDOM", 1.6, close)

    return RandomEntry


def bootstrap_ci(values: np.ndarray, resamples: int = 1000, seed: int = 0):
    n = len(values)
    if n < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    chunk = max(1, min(resamples, 4_000_000 // n))
    for start in range(0, resamples, chunk):
        stop = min(resamples, start + chunk)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def period_stats(values, days: float) -> dict:
    values = np.asarray(values, dtype=float)
    n = int(len(values))
    if n == 0:
        return {"n": 0, "per_day": 0.0, "avg_R": None, "win_rate_pct": None,
                "profit_factor": None, "total_R": 0.0, "max_dd_R": 0.0,
                "t_stat": None, "ci95_lo": None, "ci95_hi": None}
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])[1:]
    sd = float(values.std(ddof=1)) if n > 1 else float("nan")
    t_stat = float(values.mean() / (sd / math.sqrt(n))) if n > 1 and sd > 0 else None
    lo, hi = bootstrap_ci(values)
    return {
        "n": n, "per_day": round(n / max(days, 1e-9), 3),
        "avg_R": round(float(values.mean()), 4),
        "win_rate_pct": round(float((values > 0).mean() * 100), 2),
        "profit_factor": round(gains / losses, 3) if losses > 0 else None,
        "total_R": round(float(values.sum()), 2),
        "max_dd_R": round(float((peaks - equity).max()), 2),
        "t_stat": round(t_stat, 3) if t_stat is not None else None,
        "ci95_lo": round(lo, 4) if lo is not None else None,
        "ci95_hi": round(hi, 4) if hi is not None else None,
    }


def critical_z(trials: int, alpha: float = ALPHA) -> float:
    """One-sided Bonferroni-adjusted critical z for ``trials`` configurations."""
    return NormalDist().inv_cdf(1.0 - alpha / max(1, trials))


def _minus(a, b):
    return None if a is None or b is None else a - b


# ------------------------------------------------------------------ worker
def _run_symbol(task: dict) -> dict:
    market, symbol = task["market"], task["symbol"]
    raw = bt.fetch_5m(market, symbol, task["start_ms"], task["end_ms"], False)
    if raw is None or len(raw) < 500:
        return {"symbol": symbol, "first_ms": None, "last_ms": None, "runs": {}}
    built = bt.build_indicators(raw, market)          # heavy; computed once per symbol
    bt.build_indicators = lambda _raw, _market, _built=built: _built
    class_name, strategy_cls = STRATEGY_CLASS[market]
    runs = {}
    for cfg in task["configs"]:
        if cfg.get("random"):
            seed = int(cfg["random"]["seed"]) + zlib.crc32(symbol.encode())
            replacement = make_random_class(strategy_cls, RANDOM_RATE, seed)
            previous = getattr(bt, class_name)
            setattr(bt, class_name, replacement)
            try:
                trades, _ = bt.replay_symbol(market, symbol, raw)
            finally:
                setattr(bt, class_name, previous)
        else:
            with patched(strategy_cls, cfg["params"]):
                trades, _ = bt.replay_symbol(market, symbol, raw)
        runs[cfg["id"]] = [
            {"symbol": symbol, "entry_ms": int(pd.Timestamp(t["entry_time"]).timestamp() * 1000),
             "direction": t["direction"], "setup": t.get("setup", ""),
             "exit_reason": t["exit_reason"], "net_R": float(t["net_R"])}
            for t in trades
        ]
    first = int(pd.Timestamp(raw.iloc[0]["open_time"]).timestamp() * 1000)
    last = int(pd.Timestamp(raw.iloc[-1]["open_time"]).timestamp() * 1000)
    return {"symbol": symbol, "first_ms": first, "last_ms": last, "runs": runs}


# -------------------------------------------------------------- evaluation
def split_bounds(first_ms: int, last_ms: int):
    span = last_ms - first_ms
    return first_ms, first_ms + SPLITS[0] * span, first_ms + SPLITS[1] * span, last_ms


def assign_period(entry_ms: int, bounds) -> str:
    _, b1, b2, _ = bounds
    return "design" if entry_ms < b1 else ("validation" if entry_ms < b2 else "holdout")


def evaluate(all_runs: dict, first_ms: int, last_ms: int, trials: int) -> tuple[list, dict]:
    bounds = split_bounds(first_ms, last_ms)
    day_ms = 86_400_000
    days = {"design": (bounds[1] - bounds[0]) / day_ms,
            "validation": (bounds[2] - bounds[1]) / day_ms,
            "holdout": (bounds[3] - bounds[2]) / day_ms}
    z_crit = critical_z(trials)

    def by_period(trades):
        groups = {"design": [], "validation": [], "holdout": []}
        for trade in sorted(trades, key=lambda t: t["entry_ms"]):
            groups[assign_period(trade["entry_ms"], bounds)].append(trade)
        return groups

    baseline_groups = by_period(all_runs["baseline"])
    baseline_keys = {(t["symbol"], t["entry_ms"]) for t in all_runs["baseline"]}
    baseline_stats = {p: period_stats([t["net_R"] for t in baseline_groups[p]], days[p])
                      for p in ("design", "validation")}
    baseline_dv = [t["net_R"] for p in ("design", "validation") for t in baseline_groups[p]]
    baseline_dv_avg = float(np.mean(baseline_dv)) if baseline_dv else None

    rows, sealed = [], {}
    for cid, trades in all_runs.items():
        groups = by_period(trades)
        stats = {p: period_stats([t["net_R"] for t in groups[p]], days[p])
                 for p in ("design", "validation")}
        sealed[cid] = period_stats([t["net_R"] for t in groups["holdout"]], days["holdout"])
        row = {"id": cid, "design": stats["design"], "validation": stats["validation"]}
        if cid.startswith("random"):
            row["verdict"] = "CONTROL"
            rows.append(row)
            continue
        val_t = stats["validation"]["t_stat"]
        row["edge_proven"] = bool(
            val_t is not None and stats["validation"]["avg_R"] is not None
            and stats["validation"]["avg_R"] > 0 and val_t >= z_crit)
        if cid == "baseline":
            row["verdict"] = "BASELINE"
            rows.append(row)
            continue
        marginal = [t["net_R"] for p in ("design", "validation") for t in groups[p]
                    if (t["symbol"], t["entry_ms"]) not in baseline_keys]
        row["marginal_dv"] = period_stats(marginal, days["design"] + days["validation"])
        reasons = []
        base_d, base_v = baseline_stats["design"], baseline_stats["validation"]
        if stats["design"]["n"] < MIN_TRADE_GAIN * max(base_d["n"], 1):
            reasons.append(f"trade gain < {int((MIN_TRADE_GAIN - 1) * 100)}% on design")
        d_gap = _minus(stats["design"]["avg_R"], base_d["avg_R"])
        if d_gap is None or d_gap < -NON_INFERIORITY_R:
            reasons.append(f"design avgR worse than baseline by > {NON_INFERIORITY_R}R")
        v_gap = _minus(stats["validation"]["avg_R"], base_v["avg_R"])
        if stats["validation"]["n"] < MIN_VALIDATION_TRADES:
            reasons.append(f"validation trades < {MIN_VALIDATION_TRADES}")
        elif v_gap is None or v_gap < -NON_INFERIORITY_R:
            reasons.append(f"validation avgR worse than baseline by > {NON_INFERIORITY_R}R")
        m = row["marginal_dv"]
        if m["n"] < MIN_MARGINAL_TRADES:
            reasons.append(f"marginal trades < {MIN_MARGINAL_TRADES}")
        elif baseline_dv_avg is not None and m["avg_R"] < baseline_dv_avg - NON_INFERIORITY_R:
            reasons.append("marginal trades are worse than the baseline's trades")
        row["verdict"] = "ACCEPT" if not reasons else "REJECT"
        row["reasons"] = reasons
        rows.append(row)
    meta = {"first_ms": first_ms, "last_ms": last_ms, "days": days, "trials": trials,
            "z_crit": round(z_crit, 3)}
    return rows, {"sealed": sealed, "meta": meta}


def render_report(rows: list, meta: dict, title: str) -> str:
    def cell(s):
        if s["n"] == 0:
            return "0 | - | - | -"
        return f"{s['n']} | {s['per_day']} | {s['avg_R']} | {s['win_rate_pct']}%"

    lines = [f"# {title}", "",
             f"Trials (configurations) counted for multiple-testing: **{meta['trials']}**; "
             f"Bonferroni one-sided critical z = **{meta['z_crit']}**.",
             "Holdout is sealed. Costs included. Per-symbol sequential replay (optimistic).", "",
             "| config | verdict | design n | /day | avgR | win | valid n | /day | avgR | win | marginal n | marginal avgR | edge |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        marg = row.get("marginal_dv")
        lines.append(
            f"| {row['id']} | {row['verdict']} | {cell(row['design'])} | {cell(row['validation'])} | "
            f"{marg['n'] if marg else '-'} | {marg['avg_R'] if marg else '-'} | "
            f"{'YES' if row.get('edge_proven') else 'no'} |")
    notes = [r for r in rows if r.get("reasons")]
    if notes:
        lines += ["", "## Why rejected", ""]
        lines += [f"- `{r['id']}`: " + "; ".join(r["reasons"]) for r in notes]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- CLI
def parse_combo(text: str) -> dict:
    params = {}
    for part in text.split(","):
        key, value = part.split("=")
        key = key.strip()
        if key not in GRID:
            raise SystemExit(f"Unknown parameter: {key}")
        params[key] = float(value)
    return params


def run_sweep(args) -> int:
    market = args.market
    if market == "spot":
        print("WARNING: Spot analyze() hard-codes most thresholds; these class constants "
              "only affect the Spot prefilter. Sweep Futures, or refactor Spot first.", flush=True)
    if args.params.strip().lower() in ("-", "none"):
        names = []
    else:
        names = [p.strip() for p in args.params.split(",") if p.strip()] if args.params else list(GRID)
    configs = [{"id": "baseline", "params": {}}]
    for name in names:
        if name not in GRID:
            raise SystemExit(f"Unknown parameter: {name} (choose from {', '.join(GRID)})")
        for value in GRID[name][1:]:
            params = {name: value}
            configs.append({"id": config_id(params), "params": params})
    for combo in args.combo or []:
        params = parse_combo(combo)
        configs.append({"id": config_id(params), "params": params})
    for seed in range(args.random_baseline):
        configs.append({"id": f"random_seed{seed}", "params": {}, "random": {"seed": seed}})

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start_ms = int(pd.Timestamp(args.start).timestamp() * 1000)
    end_ms = int((pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC")).timestamp() * 1000)
    tasks = [{"market": market, "symbol": s, "start_ms": start_ms, "end_ms": end_ms,
              "configs": configs} for s in symbols]
    print(f"[{market}] {len(symbols)} symbols x {len(configs)} configs, workers={args.workers}", flush=True)

    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for done in pool.map(_run_symbol, tasks):
                results.append(done)
                print(f"  done {done['symbol']}", flush=True)
    else:
        for task in tasks:
            results.append(_run_symbol(task))
            print(f"  done {results[-1]['symbol']}", flush=True)

    firsts = [r["first_ms"] for r in results if r["first_ms"]]
    lasts = [r["last_ms"] for r in results if r["last_ms"]]
    if not firsts:
        print("No data available.", file=sys.stderr)
        return 2
    all_runs = {c["id"]: [] for c in configs}
    for r in results:
        for cid, trades in r["runs"].items():
            all_runs[cid].extend(trades)
    non_control = [c for c in configs if not c.get("random")]
    trials = len(non_control) - 1 + args.prior_trials  # baseline is not a trial
    rows, extra = evaluate(all_runs, min(firsts), max(lasts), max(1, trials))

    out = Path(args.out) if args.out else Path("data/sweep_results") / f"{market}_{args.label}"
    out.mkdir(parents=True, exist_ok=True)
    title = f"V3 {market} relaxation sweep - {args.label}"
    (out / "report.md").write_text(render_report(rows, extra["meta"], title))
    (out / "results.json").write_text(json.dumps({"rows": rows, "meta": extra["meta"]}, indent=2))
    (out / "holdout_sealed.json").write_text(json.dumps(extra["sealed"], indent=2))
    flat = [{"config": cid, **t} for cid, trades in all_runs.items() for t in trades]
    pd.DataFrame(flat).to_csv(out / "trades.csv", index=False)
    print(render_report(rows, extra["meta"], title))
    print(f"Saved to {out} (holdout sealed: use --show-holdout {out} --config ID once, at the end)")
    return 0


def show_holdout(args) -> int:
    folder = Path(args.show_holdout)
    sealed = json.loads((folder / "holdout_sealed.json").read_text())
    wanted = ["baseline", args.config] if args.config else None
    if not wanted or args.config not in sealed:
        raise SystemExit("--config must name exactly one configuration present in the sealed file")
    with (folder / "holdout_access.log").open("a") as log:
        log.write(f"{datetime.now(timezone.utc).isoformat()} opened holdout for {args.config}\n")
    print("HOLDOUT (final, one-shot look - do not iterate after seeing this):")
    for cid in wanted:
        print(cid, json.dumps(sealed[cid]))
    return 0


def aggregate(args) -> int:
    merged, meta = {}, None
    for folder in args.aggregate:
        data = json.loads((Path(folder) / "results.json").read_text())
        meta = meta or data["meta"]
        for row in data["rows"]:
            merged.setdefault(row["id"], row)
    non_control = [r for r in merged.values() if r["verdict"] not in ("BASELINE", "CONTROL")]
    trials = len(non_control)
    z_crit = critical_z(max(1, trials))
    for row in merged.values():
        if row["verdict"] in ("BASELINE", "CONTROL"):
            continue
        val = row["validation"]
        row["edge_proven"] = bool(val["t_stat"] is not None and val["avg_R"] and val["avg_R"] > 0
                                  and val["t_stat"] >= z_crit)
    meta = {**meta, "trials": trials, "z_crit": round(z_crit, 3)}
    rows = list(merged.values())
    out = Path(args.out or "data/sweep_results/combined")
    out.mkdir(parents=True, exist_ok=True)
    report = render_report(rows, meta, "V3 relaxation sweep - combined")
    (out / "report.md").write_text(report)
    print(report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--market", default="futures", choices=["futures", "spot"])
    parser.add_argument("--symbols", default=",".join(bt.SYMBOLS))
    parser.add_argument("--params", default="", help="comma list; default = all in GRID; 'none' = baseline only")
    parser.add_argument("--combo", action="append", help="e.g. PULLBACK_VOLUME=0.65,BREAKOUT_VOLUME=0.95")
    parser.add_argument("--random-baseline", type=int, default=0, help="number of random-entry seeds")
    parser.add_argument("--start", default=bt.START_UTC)
    parser.add_argument("--end", default="", help="fixed end date for reproducibility")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default="")
    parser.add_argument("--prior-trials", type=int, default=0,
                        help="configurations already tried before this run (multiple-testing count)")
    parser.add_argument("--aggregate", nargs="+", help="merge results.json folders into one report")
    parser.add_argument("--show-holdout", default="", help="results folder; requires --config")
    parser.add_argument("--config", default="")
    args = parser.parse_args()
    if args.aggregate:
        return aggregate(args)
    if args.show_holdout:
        return show_holdout(args)
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
