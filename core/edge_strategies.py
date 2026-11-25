"""Independent, closed-candle strategy detectors for offline/DEMO evaluation.

These detectors propose research candidates only. They do not create orders,
modify V3, calculate protective exits, or authorize LIVE execution.
"""
from math import isfinite


def _num(value):
    try:
        value = float(value)
        return value if isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _closed_rows(frame, timeframe):
    if frame is None:
        return []
    attrs = getattr(frame, "attrs", {}) or {}
    actual = str(attrs.get("timeframe") or attrs.get("interval") or "").lower()
    if actual and actual != timeframe:
        return []
    try:
        rows = frame.to_dict("records") if hasattr(frame, "to_dict") else list(frame)
    except Exception:
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict) or not bool(row.get("is_closed", False)):
            continue
        clean = {str(k).lower(): _num(v) for k, v in row.items()}
        if all(clean.get(k) is not None for k in ("open", "high", "low", "close")):
            if clean["high"] >= max(clean["open"], clean["close"], clean["low"]) and clean["low"] <= min(clean["open"], clean["close"], clean["high"]):
                out.append(clean)
    return out


def _ema(values, period):
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    value = sum(values[:period]) / period
    for item in values[period:]:
        value = alpha * item + (1.0 - alpha) * value
    return value


def _rsi(values, period=14):
    if len(values) <= period:
        return None
    changes = [values[i] - values[i-1] for i in range(len(values)-period, len(values))]
    gains = sum(max(x, 0) for x in changes) / period
    losses = sum(max(-x, 0) for x in changes) / period
    return 100.0 if losses == 0 else 100.0 - 100.0 / (1.0 + gains / losses)


def _features(frames):
    rows5 = _closed_rows((frames or {}).get("5m"), "5m")
    rows15 = _closed_rows((frames or {}).get("15m"), "15m")
    rows1h = _closed_rows((frames or {}).get("1h"), "1h")
    if len(rows5) < 25:
        return None
    c = [x["close"] for x in rows5]
    h = [x["high"] for x in rows5]
    l = [x["low"] for x in rows5]
    v = [_num(x.get("volume")) for x in rows5]
    last, prev = rows5[-1], rows5[-2]
    span = last["high"] - last["low"]
    if span <= 0:
        return None
    avg_volume = [x for x in v[-21:-1] if x is not None]
    volume_ratio = (v[-1] / (sum(avg_volume)/len(avg_volume))) if v[-1] is not None and avg_volume and sum(avg_volume) > 0 else None
    ema20 = _ema(c[-40:], 20)
    ema20_prev = _ema(c[-41:-1], 20) if len(c) >= 41 else None
    ema9 = _ema(c[-20:], 9)
    atr_ranges = [max(rows5[i]["high"]-rows5[i]["low"], abs(rows5[i]["high"]-c[i-1]), abs(rows5[i]["low"]-c[i-1])) for i in range(1,len(rows5))]
    atr = sum(atr_ranges[-14:])/min(14,len(atr_ranges)) if atr_ranges else None
    ema15 = _ema([x["close"] for x in rows15], 20) if len(rows15)>=20 else None
    ema1h = _ema([x["close"] for x in rows1h], 20) if len(rows1h)>=20 else None
    close_location = (last["close"]-last["low"])/span
    body_fraction = abs(last["close"]-last["open"])/span
    upper_wick = (last["high"]-max(last["open"],last["close"]))/span
    lower_wick = (min(last["open"],last["close"])-last["low"])/span
    return {
        "rows5":rows5,"close":c[-1],"previous_close":c[-2],"high":h,"low":l,
        "ema20":ema20,"ema20_prev":ema20_prev,"ema9":ema9,"atr":atr,
        "ema15":ema15,"ema1h":ema1h,"rsi":_rsi(c),"volume_ratio":volume_ratio,
        "close_location":close_location,"body_fraction":body_fraction,
        "upper_wick":upper_wick,"lower_wick":lower_wick,
        "prior_high":max(h[-21:-1]),"prior_low":min(l[-21:-1]),
        "range_high":max(h[-30:-1]),"range_low":min(l[-30:-1]),
    }


class EdgeStrategyEngine:
    """Four independent archetypes; candidate generation is intentionally non-executing."""

    VERSION = "edge-archetypes-v1"

    def analyze(self, market, frames):
        market = str(market or "").lower()
        if market not in ("futures", "spot", "alpha"):
            return {"success": False, "reason": "unsupported market", "candidates": []}
        f = _features(frames)
        if not f:
            return {"success": False, "reason": "insufficient closed 5m candles", "candidates": []}
        if None in (f["ema20"], f["ema9"], f["atr"], f["rsi"], f["volume_ratio"]):
            return {"success": False, "reason": "incomplete closed-candle features", "candidates": []}
        long_only = market in ("spot", "alpha")
        up = f["ema20_prev"] is not None and f["close"] > f["ema20"] and f["ema20"] > f["ema20_prev"]
        down = f["ema20_prev"] is not None and f["close"] < f["ema20"] and f["ema20"] < f["ema20_prev"]
        candidates = []

        def add(name, direction, reason, strength):
            if direction == "SHORT" and long_only:
                return
            candidates.append({
                "strategy": name, "strategy_version": self.VERSION,
                "market": market, "direction": direction,
                "timestamp": f["rows5"][-1].get("open_time", f["rows5"][-1].get("timestamp")),
                "strength": round(max(0.0, min(1.0, strength)), 4),
                "reason": reason, "execution_authorized": False,
            })

        # Pullback continuation: established EMA trend, retest of EMA20, close resumes.
        if up and f["low"][-1] <= f["ema20"] + .20*f["atr"] and f["close"] > f["ema9"] and f["close"] > f["rows5"][-1]["open"] and 42 <= f["rsi"] <= 68 and f["close_location"] >= .60:
            add("PULLBACK", "LONG", "uptrend retest and bullish 5m reclaim", .55 + min(.2, f["volume_ratio"]*.05))
        if down and f["high"][-1] >= f["ema20"] - .20*f["atr"] and f["close"] < f["ema9"] and f["close"] < f["rows5"][-1]["open"] and 32 <= f["rsi"] <= 58 and f["close_location"] <= .40:
            add("PULLBACK", "SHORT", "downtrend retest and bearish 5m rejection", .55 + min(.2, f["volume_ratio"]*.05))

        # Breakout: closed candle clears the previous 20-candle boundary with expansion.
        if f["close"] > f["prior_high"] and f["close"] > f["rows5"][-1]["open"] and f["volume_ratio"] >= 1.15 and f["close_location"] >= .72:
            add("BREAKOUT", "LONG", "closed 5m close above prior 20-bar high with volume expansion", .58 + min(.2,(f["volume_ratio"]-1)*.12))
        if f["close"] < f["prior_low"] and f["close"] < f["rows5"][-1]["open"] and f["volume_ratio"] >= 1.15 and f["close_location"] <= .28:
            add("BREAKOUT", "SHORT", "closed 5m close below prior 20-bar low with volume expansion", .58 + min(.2,(f["volume_ratio"]-1)*.12))

        # Reclaim: sweep the prior extreme, then close back through it with rejection wick.
        if f["low"][-1] < f["prior_low"] and f["close"] > f["prior_low"] and f["close"] > f["rows5"][-1]["open"] and f["lower_wick"] >= .30:
            add("RECLAIM", "LONG", "liquidity sweep below prior low reclaimed on close", .60 + min(.15,f["lower_wick"]*.2))
        if f["high"][-1] > f["prior_high"] and f["close"] < f["prior_high"] and f["close"] < f["rows5"][-1]["open"] and f["upper_wick"] >= .30:
            add("RECLAIM", "SHORT", "liquidity sweep above prior high rejected on close", .60 + min(.15,f["upper_wick"]*.2))

        # Range rejection: only when higher timeframe trend is weak/flat.
        trend_slope = abs((f["ema20"]/f["ema20_prev"]-1)) if f["ema20_prev"] else 1
        ranging = trend_slope < .002 and (f["ema15"] is None or abs(f["close"]/f["ema15"]-1)<.012)
        if ranging and f["close"] <= f["range_low"] + .15*f["atr"] and f["close"] > f["rows5"][-1]["open"] and f["rsi"] <= 38 and f["lower_wick"] >= .30:
            add("RANGE", "LONG", "range-low rejection with oversold momentum", .52 + min(.18,f["lower_wick"]*.2))
        if ranging and f["close"] >= f["range_high"] - .15*f["atr"] and f["close"] < f["rows5"][-1]["open"] and f["rsi"] >= 62 and f["upper_wick"] >= .30:
            add("RANGE", "SHORT", "range-high rejection with overbought momentum", .52 + min(.18,f["upper_wick"]*.2))

        # Deterministic ordering only; no learned score is implied by strength.
        candidates.sort(key=lambda x: (x["strength"], x["strategy"]), reverse=True)
        return {"success": True, "strategy_version": self.VERSION,
                "market": market, "candidate_count": len(candidates),
                "candidates": candidates,
                "execution_authorized": False,
                "note": "Research/DEMO candidate detector only; existing risk and exit engines remain authoritative."}
