"""ATR + confirmed candle-structure trailing stop calculations.

The stop only tightens. It never moves away from the market or increases
the risk on an already-open position.
"""
from math import isfinite


def calculate_trailing_stop(frame, direction, previous_stop, best_price,
                            current_price, atr_period=14,
                            atr_multiple=1.5, swing_lookback=5):
    if frame is None or len(frame) < atr_period + 2:
        return None
    columns = {str(c).strip().lower(): c for c in frame.columns}
    if not all(k in columns for k in ("high", "low", "close")):
        return None
    try:
        highs = [float(v) for v in frame[columns["high"]].tail(atr_period + 1)]
        lows = [float(v) for v in frame[columns["low"]].tail(atr_period + 1)]
        closes = [float(v) for v in frame[columns["close"]].tail(atr_period + 1)]
        if not all(isfinite(v) and v > 0 for v in highs + lows + closes):
            return None
        ranges = []
        for i in range(1, len(closes)):
            ranges.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))
        atr = sum(ranges[-atr_period:]) / min(atr_period, len(ranges))
        if not isfinite(atr) or atr <= 0:
            return None
        direction = str(direction).upper()
        previous = float(previous_stop)
        current = float(current_price)
        best = float(best_price or current)
        recent_highs = highs[-swing_lookback:]
        recent_lows = lows[-swing_lookback:]
        if direction == "LONG":
            best = max(best, current, max(recent_highs))
            structure = min(recent_lows) - 0.20 * atr
            candidate = max(structure, best - atr_multiple * atr)
            tightened = max(previous, candidate)
        elif direction == "SHORT":
            best = min(best, current, min(recent_lows))
            structure = max(recent_highs) + 0.20 * atr
            candidate = min(structure, best + atr_multiple * atr)
            tightened = min(previous, candidate)
        else:
            return None
        if not isfinite(tightened) or tightened <= 0:
            return None
        return {
            "stop": tightened,
            "best_price": best,
            "atr": atr,
            "structure_level": structure,
            "method": "ATR_1.5x_PLUS_5_CANDLE_STRUCTURE",
        }
    except (TypeError, ValueError, ArithmeticError, IndexError, KeyError):
        return None


def net_break_even_price(entry_price, direction, fee_rate, slippage_rate):
    """Estimated exit price that offsets entry/exit fees and slippage."""
    entry = float(entry_price)
    side_cost = float(fee_rate) + float(slippage_rate)
    if not isfinite(entry) or entry <= 0 or not isfinite(side_cost) or not 0 <= side_cost < 1:
        raise ValueError("Invalid entry price or execution cost")
    direction = str(direction).upper()
    if direction == "LONG":
        return entry * (1.0 + side_cost) / (1.0 - side_cost)
    if direction == "SHORT":
        return entry * (1.0 - side_cost) / (1.0 + side_cost)
    raise ValueError("Direction must be LONG or SHORT")


def apply_cost_aware_trailing_floor(candidate_stop, entry_price, direction, best_price, fee_rate, slippage_rate):
    """Apply gross break-even until best price covers costs, then net break-even.
    Never put the stop beyond the best favorable price.
    """
    candidate = float(candidate_stop)
    entry = float(entry_price)
    best = float(best_price)
    net_be = net_break_even_price(entry, direction, fee_rate, slippage_rate)
    if str(direction).upper() == "LONG":
        floor = net_be if best >= net_be else entry
        return min(best, max(candidate, floor))
    ceiling = net_be if best <= net_be else entry
    return max(best, min(candidate, ceiling))