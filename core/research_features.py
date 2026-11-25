"""Deterministic, causal feature extraction for completed OHLCV candles."""
from math import isfinite


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) else None


def _ema(values, period):
    if not values:
        return None
    alpha = 2.0 / (period + 1.0)
    value = values[0]
    for item in values[1:]:
        value = alpha * item + (1.0 - alpha) * value
    return value


def extract_timeframe_features(frame, prefix):
    """Use only the supplied frame's rows; caller must supply closed candles."""
    if frame is None:
        return {}
    try:
        rows = frame.to_dict("records") if hasattr(frame, "to_dict") else list(frame)
    except Exception:
        return {}
    rows = [row for row in rows if isinstance(row, dict)
            and bool(row.get("is_closed", True))]
    clean = []
    for row in rows:
        values = {key: _finite(row.get(key)) for key in
                  ("open", "high", "low", "close", "volume")}
        if all(values[key] is not None for key in ("high", "low", "close")):
            clean.append(values)
    if len(clean) < 3:
        return {}
    closes = [r["close"] for r in clean]
    highs = [r["high"] for r in clean]
    lows = [r["low"] for r in clean]
    volumes = [r["volume"] for r in clean if r["volume"] is not None]
    true_ranges, plus_dm, minus_dm = [], [], []
    for i in range(1, len(clean)):
        prev_close = closes[i - 1]
        true_ranges.append(max(highs[i] - lows[i],
                               abs(highs[i] - prev_close),
                               abs(lows[i] - prev_close)))
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    last = clean[-1]
    close = closes[-1]
    if close <= 0:
        return {}
    features = {}
    def put(name, value):
        value = _finite(value)
        if value is not None:
            features[prefix + "__" + name] = value
    for lag in (1, 3, 6, 12):
        if len(closes) > lag and closes[-lag-1] != 0:
            put("return_%s_pct" % lag,
                (close / closes[-lag-1] - 1.0) * 100.0)
    for period in (7, 14, 20):
        if len(true_ranges) >= period:
            atr = sum(true_ranges[-period:]) / period
            put("atr_pct_%s" % period, atr / close * 100.0)
    if len(closes) >= 22:
        ema_now = _ema(closes[-20:], 20)
        ema_prev = _ema(closes[-21:-1], 20)
        if ema_prev:
            put("ema20_slope_pct", (ema_now / ema_prev - 1.0) * 100.0)
        put("close_vs_ema20_pct", (close / ema_now - 1.0) * 100.0)
    if len(closes) >= 15:
        changes = [closes[i] - closes[i-1] for i in range(len(closes)-14, len(closes))]
        gains = sum(max(x, 0.0) for x in changes) / 14.0
        losses = sum(max(-x, 0.0) for x in changes) / 14.0
        put("rsi14", 100.0 if losses == 0 else 100.0 - 100.0 / (1.0 + gains / losses))
        if len(true_ranges) >= 14:
            tr = sum(true_ranges[-14:])
            pdi = 100.0 * sum(plus_dm[-14:]) / tr if tr else 0.0
            mdi = 100.0 * sum(minus_dm[-14:]) / tr if tr else 0.0
            dx = 100.0 * abs(pdi - mdi) / (pdi + mdi) if pdi + mdi else 0.0
            put("adx14_proxy", dx)
    if len(volumes) >= 20 and volumes[-1] is not None:
        average = sum(volumes[-20:]) / 20.0
        if average > 0:
            put("volume_ratio20", volumes[-1] / average)
    candle_range = last["high"] - last["low"]
    if candle_range > 0 and last["open"] is not None:
        put("body_fraction", abs(last["close"] - last["open"]) / candle_range)
        put("upper_wick_fraction",
            (last["high"] - max(last["open"], last["close"])) / candle_range)
        put("lower_wick_fraction",
            (min(last["open"], last["close"]) - last["low"]) / candle_range)
        put("close_location",
            (last["close"] - last["low"]) / candle_range)
    return features


def extract_research_features(frames):
    result = {}
    for timeframe, frame in (frames or {}).items():
        result.update(extract_timeframe_features(frame, str(timeframe)))
    return result


def classify_market_regime(features):
    """Descriptive regime label from normalized extracted features, not a signal."""
    f = features or {}
    adx = f.get("15m__adx14_proxy", f.get("adx_15m"))
    atr_pct = f.get("5m__atr_pct_14", f.get("atr_pct_5m"))
    slope = f.get("1h__ema20_slope_pct", f.get("ema20_slope_1h"))
    if atr_pct is not None and atr_pct >= 2.0:
        return "HIGH_VOLATILITY"
    if adx is None or slope is None:
        return "UNKNOWN"
    if adx >= 25 and slope > 0:
        return "TREND_UP"
    if adx >= 25 and slope < 0:
        return "TREND_DOWN"
    if adx <= 18:
        return "RANGE"
    return "TRANSITION"
