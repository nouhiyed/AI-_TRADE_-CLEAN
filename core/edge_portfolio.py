"""Greedy non-overlapping per-strategy replay. Research-only approximation."""
from collections import defaultdict
from core.edge_backtest import run_chronological_backtest, _stats

def non_overlapping_replay(rows, market="futures", fee_rate=0.001,
                          slippage_rate=0.0005, horizon=48,
                          warmup_fraction=0.60):
    raw = run_chronological_backtest(rows, market=market, fee_rate=fee_rate,
        slippage_rate=slippage_rate, horizon=horizon,
        warmup_fraction=warmup_fraction)
    grouped = defaultdict(list)
    for trade in raw.get("trades", []):
        grouped[trade["strategy"]].append(trade)
    accepted, all_trades = defaultdict(list), []
    for name in raw.get("by_strategy", {}):
        busy_until = -1
        for trade in sorted(grouped[name], key=lambda t:t["entry_time"]):
            if trade["entry_time"] < busy_until:
                continue
            accepted[name].append(trade)
            busy_until = trade["entry_time"] + max(1,trade["bars"])*300000
            all_trades.append(trade)
    raw["replay"] = "NON_OVERLAP_PER_STRATEGY"
    raw["portfolio_overall"] = _stats(all_trades)
    raw["portfolio_by_strategy"] = {n:_stats(accepted.get(n,[])) for n in raw.get("by_strategy",{})}
    raw["portfolio_trades"] = all_trades
    raw["warning"] = raw.get("warning","") + " This greedy filter permits concurrency across strategies and does not model shared capital, margin, or funding."
    return raw
