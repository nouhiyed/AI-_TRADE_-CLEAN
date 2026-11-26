"""Offline, conservative forward-outcome labelling for research candidates.

This module does not create orders or alter V3 exits. Callers supply only bars
that occur strictly after a candidate candle. If stop and target touch in the
same OHLC bar, STOP is assumed first (conservative ambiguity handling).
"""
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Outcome:
    status: str
    bars: int
    exit_price: float | None
    result_r: float | None
    mfe_r: float
    mae_r: float
    reason: str


def evaluate_forward_path(direction, entry, stop, target, bars: Iterable[dict],
                          max_bars=48, fee_rate=0.001, slippage_rate=0.0005):
    direction = str(direction).upper()
    direction = {"BUY": "LONG", "SELL": "SHORT"}.get(direction, direction)
    if direction not in ("LONG", "SHORT"):
        raise ValueError("direction must be LONG or SHORT")
    entry, stop, target = float(entry), float(stop), float(target)
    risk = abs(entry - stop)
    if risk <= 0 or (direction == "LONG" and not stop < entry < target) or (
            direction == "SHORT" and not target < entry < stop):
        raise ValueError("invalid entry/stop/target geometry")
    if max_bars < 1:
        raise ValueError("max_bars must be positive")

    bars = list(bars)[:max_bars]
    if len(bars) < max_bars:
        return Outcome(\"INCOMPLETE\", len(bars), None, None, 0.0, 0.0, \"INSUFFICIENT_FORWARD_HORIZON\")
    mfe = 0.0
    mae = 0.0
    for index, bar in enumerate(bars):
        if index >= max_bars:
            break
        high, low = float(bar["high"]), float(bar["low"])
        if high < low:
            raise ValueError("bar high is below low")
        if direction == "LONG":
            mfe = max(mfe, (high - entry) / risk)
            mae = min(mae, (low - entry) / risk)
            stop_hit, target_hit = low <= stop, high >= target
        else:
            mfe = max(mfe, (entry - low) / risk)
            mae = min(mae, (entry - high) / risk)
            stop_hit, target_hit = high >= stop, low <= target
        if stop_hit or target_hit:
            # Conservative ordering when OHLC cannot reveal intrabar sequence.
            reason = "STOP_LOSS" if stop_hit else "TAKE_PROFIT"
            raw_exit = stop if stop_hit else target
            exit_slippage = raw_exit * slippage_rate
            exit_price = raw_exit - exit_slippage if direction == "LONG" else raw_exit + exit_slippage
            effective_entry = (entry + entry * slippage_rate if direction == "LONG"
                               else entry - entry * slippage_rate)
            gross = ((exit_price - effective_entry) if direction == "LONG"
                     else (effective_entry - exit_price))
            fees = (effective_entry + exit_price) * fee_rate
            net_r = (gross - fees) / risk
            return Outcome(reason, index + 1, exit_price, net_r, mfe, mae, reason)

    bars_seen = len(bars)
    if bars_seen == 0:
        return Outcome("TIMEOUT", 0, None, None, mfe, mae, "NO_HORIZON_BARS")
    raw_exit = float(bars[-1]["close"])
    effective_entry = (entry + entry * slippage_rate if direction == "LONG"
                       else entry - entry * slippage_rate)
    exit_price = (raw_exit - raw_exit * slippage_rate if direction == "LONG"
                  else raw_exit + raw_exit * slippage_rate)
    gross = ((exit_price - effective_entry) if direction == "LONG"
             else (effective_entry - exit_price))
    fees = (effective_entry + exit_price) * fee_rate
    net_r = (gross - fees) / risk
    return Outcome("TIMEOUT", bars_seen, exit_price, net_r, mfe, mae,
                   "HORIZON_MARK_TO_MARKET")


def classify_market_regime(features):
    """Transparent descriptive classifier; not a trade approval model."""
    f = features or {}
    adx = f.get("adx_15m")
    atr_pct = f.get("atr_pct_5m")
    trend_up = f.get("ema20_slope_1h", 0) > 0
    trend_down = f.get("ema20_slope_1h", 0) < 0
    if atr_pct is not None and atr_pct >= 2.0:
        return "HIGH_VOLATILITY"
    if adx is None:
        return "UNKNOWN"
    if adx >= 25 and trend_up:
        return "TREND_UP"
    if adx >= 25 and trend_down:
        return "TREND_DOWN"
    if adx <= 18:
        return "RANGE"
    return "TRANSITION"
