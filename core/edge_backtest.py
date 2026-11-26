"""Chronological OHLCV backtest for observational edge strategies.

Signals are generated only from closed candles available at decision time.
Execution is conservatively simulated at the NEXT candle open. Fixed ATR
brackets are research assumptions, not the production V3 exit engine.
"""
from collections import defaultdict
from math import isfinite
from statistics import mean
from core.edge_strategies import EdgeStrategyEngine
from core.research_outcomes import evaluate_forward_path

STRATEGIES = ("PULLBACK", "BREAKOUT", "RECLAIM", "RANGE")


class _Frame(list):
    def __init__(self, rows, timeframe):
        super().__init__(rows)
        self.attrs = {"timeframe": timeframe}

    def to_dict(self, orient="records"):
        return list(self)


def _valid_rows(rows):
    out = []
    last_stamp = None
    for source in rows:
        try:
            row = {k: float(source[k]) for k in ("open", "high", "low", "close")}
            row["volume"] = float(source.get("volume", 0) or 0)
            stamp = int(float(source.get("open_time", source.get("timestamp"))))
            if stamp > 10_000_000_000:
                stamp //= 1
            if row["high"] < max(row["open"], row["close"], row["low"]):
                continue
            if row["low"] > min(row["open"], row["close"], row["high"]):
                continue
            if last_stamp is not None and stamp <= last_stamp:
                continue
            row.update({"open_time": stamp, "is_closed": True})
            out.append(row)
            last_stamp = stamp
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return out


def _atr(rows, period=14):
    if len(rows) < period + 1:
        return None
    values = []
    for i in range(len(rows) - period, len(rows)):
        previous = rows[i - 1]["close"]
        bar = rows[i]
        values.append(max(bar["high"] - bar["low"],
                          abs(bar["high"] - previous),
                          abs(bar["low"] - previous)))
    value = mean(values)
    return value if isfinite(value) and value > 0 else None


def _stats(trades):
    closed = [x for x in trades if x["status"] in ("STOP_LOSS", "TAKE_PROFIT", "TIMEOUT")]
    values = [x["result_r"] for x in closed if x["result_r"] is not None]
    wins = [x for x in values if x > 0]
    losses = [x for x in values if x < 0]
    gross_win, gross_loss = sum(wins), abs(sum(losses))
    return {
        "signals": len(trades), "closed": len(closed),
        "timeouts": sum(x["status"] == "TIMEOUT" for x in trades),\n        "resolved": len(closed),
        "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(100 * len(wins) / len(closed), 2) if closed else None,
        "net_r": round(sum(values), 4) if values else None,
        "avg_r": round(mean(values), 4) if values else None,
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss else (None if not gross_win else "INF"),
        "max_drawdown_r": _max_drawdown(values),
    }


def _max_drawdown(values):
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return round(drawdown, 4)


def run_chronological_backtest(rows, market="futures", fee_rate=0.001,
                               slippage_rate=0.0005, horizon=48,
                               warmup_fraction=0.60):
    """Evaluate fixed rules only in the chronological holdout after warmup.

    A signal on candle i is filled at candle i+1 open. The future path begins
    at that entry candle. Same-bar SL/TP ambiguity is stop-first.
    """
    clean = _valid_rows(rows)
    if len(clean) < 100:
        empty = _stats([])
        return {"status": "INSUFFICIENT_DATA", "bars": len(clean),
                "minimum_bars": 100, "overall": empty,
                "by_strategy": {name: _stats([]) for name in STRATEGIES},
                "oos_bars": 0, "trades": [],
                "warning": "Not enough historical bars to evaluate."}
    split = max(25, min(len(clean) - horizon - 1,
                        int(len(clean) * float(warmup_fraction))))
    engine = EdgeStrategyEngine()
    results = defaultdict(list)
    all_trades = []
    for index in range(max(25, split), len(clean) - horizon):
        history = clean[:index + 1]
        atr = _atr(history)
        if atr is None:
            continue
        frame = _Frame(history, "5m")
        detected = engine.analyze(market, {"5m": frame})
        if not detected.get("success"):
            continue
        next_bar = clean[index + 1]
        entry = next_bar["open"]
        future = clean[index + 1:index + 1 + horizon]
        for candidate in detected.get("candidates", []):
            strategy = candidate.get("strategy")
            direction = candidate.get("direction")
            if strategy not in STRATEGIES:
                continue
            risk = 1.5 * atr
            stop = entry - risk if direction == "LONG" else entry + risk
            target = entry + 2.25 * atr if direction == "LONG" else entry - 2.25 * atr
            outcome = evaluate_forward_path(
                direction, entry, stop, target, future, max_bars=horizon,
                fee_rate=fee_rate, slippage_rate=slippage_rate)
            trade = {
                "strategy": strategy, "market": market, "direction": direction,
                "signal_time": history[-1]["open_time"],
                "entry_time": next_bar["open_time"], "entry": entry,
                "stop": stop, "target": target, "status": outcome.status,
                "bars": outcome.bars, "result_r": outcome.result_r,
                "mfe_r": outcome.mfe_r, "mae_r": outcome.mae_r,
            }
            results[strategy].append(trade)
            all_trades.append(trade)
    summary = {name: _stats(results.get(name, [])) for name in STRATEGIES}
    total = _stats(all_trades)
    return {
        "status": "RESEARCH_ONLY" if total["closed"] >= 100 else "INSUFFICIENT_DATA",
        "market": market, "timeframe": "5m", "bars": len(clean),
        "warmup_bars": split, "oos_bars": len(clean) - split,
        "execution": "NEXT_BAR_OPEN", "stop_atr": 1.5, "target_atr": 2.25,
        "horizon_bars": horizon, "fee_rate": fee_rate,
        "slippage_rate": slippage_rate, "same_bar_policy": "STOP_FIRST",
        "overall": total, "by_strategy": summary, "trades": all_trades,
        "warning": "Fixed-rule historical simulation; not proof of future profitability.",
    }


def stress_test(rows, market="futures", horizon=48):
    """Re-run identical rules under base and adverse transaction-cost assumptions."""
    scenarios = {
        "base": (0.001, 0.0005),
        "double_costs": (0.002, 0.001),
        "severe_costs": (0.003, 0.002),
    }
    return {name: run_chronological_backtest(
        rows, market=market, fee_rate=cost[0], slippage_rate=cost[1],
        horizon=horizon) for name, cost in scenarios.items()}
