"""V3: 5-minute closed-candle entry strategy with closed higher-timeframe context.

All signal thresholds below are calibrated as an explicit 5m profile. No 1m
trigger, fallback, or intrabar signal is permitted. Open positions are managed
separately by the live position monitor.
"""
from math import isfinite


class _V3Base:
    TIMEFRAME = "5m"
    PROFILE = "V3_5M"
    STOP_ATR = 1.15
    MIN_NET_RR = 1.5
    MIN_NET_REWARD_RATE = 0.00045
    FEE_RATE = {"spot": 0.0010, "futures": 0.0005}
    ADVERSE_RATE = 0.0005
    VOLUME_LOOKBACK = 24
    BREAKOUT_LOOKBACK = 12

    # 5m candle quality gates (ratios are dimensionless, not raw price units).
    PULLBACK_BODY_ATR = 0.10
    BREAKOUT_BODY_ATR = 0.18
    PULLBACK_VOLUME = 0.85
    BREAKOUT_VOLUME = 1.05
    PULLBACK_CLOSE_LOCATION = 0.62
    BREAKOUT_CLOSE_LOCATION = 0.75
    REJECTION_WICK_RANGE = 0.35
    BREAKOUT_MAX_OPPOSING_WICK = 0.18

    @staticmethod
    def value(frame, name, index=-1):
        if frame is None or not hasattr(frame, "columns") or len(frame) < abs(index):
            return None
        cols = {str(c).strip().lower(): c for c in frame.columns}
        col = cols.get(name.lower())
        if col is None:
            return None
        try:
            value = float(frame[col].iloc[index])
            return value if isfinite(value) else None
        except (TypeError, ValueError, IndexError, KeyError):
            return None

    @classmethod
    def frame_is_closed(cls, frame, expected_timeframe):
        if frame is None or len(frame) < 60:
            return False
        attrs = getattr(frame, "attrs", {})
        timeframe = str(attrs.get("timeframe") or attrs.get("interval") or "").lower()
        if timeframe != str(expected_timeframe).lower():
            return False
        if "is_closed" not in frame.columns:
            return False
        try:
            return bool(frame["is_closed"].iloc[-1])
        except (TypeError, ValueError, IndexError):
            return False

    @classmethod
    def frame_is_closed_5m(cls, frame):
        return cls.frame_is_closed(frame, cls.TIMEFRAME)

    @classmethod
    def volume_ratio(cls, frame, lookback=None):
        lookback = int(lookback or cls.VOLUME_LOOKBACK)
        if frame is None or len(frame) < lookback + 1 or "volume" not in frame.columns:
            return None
        try:
            prior = frame["volume"].iloc[-lookback-1:-1].astype(float)
            avg = float(prior.mean())
            current = float(frame["volume"].iloc[-1])
            if not isfinite(avg) or not isfinite(current) or avg <= 0 or current < 0:
                return None
            return current / avg
        except (TypeError, ValueError):
            return None

    @classmethod
    def bias(cls, five, fifteen, direction):
        c5, e21_5 = cls.value(five, "close"), cls.value(five, "EMA_21")
        c15, e21_15, e50_15 = (
            cls.value(fifteen, x) for x in ("close", "EMA_21", "EMA_50")
        )
        if None in (c5, e21_5, c15, e21_15, e50_15):
            return False
        if direction == "LONG":
            return c15 > e50_15 and (c5 >= e21_5 or e21_15 > e50_15)
        return c15 < e50_15 and (c5 <= e21_5 or e21_15 < e50_15)

    @classmethod
    def candle_shape(cls, frame):
        o, h, low, c = (cls.value(frame, x) for x in ("open", "high", "low", "close"))
        if None in (o, h, low, c) or h <= low:
            return None
        span = h - low
        body = abs(c - o)
        upper = max(0.0, h - max(o, c))
        lower = max(0.0, min(o, c) - low)
        return {
            "body_atr": None,
            "body_range": body / span,
            "upper_wick_range": upper / span,
            "lower_wick_range": lower / span,
            "close_location": (c - low) / span,
            "range": span,
        }

    @classmethod
    def plan(cls, market, direction, five, setup, regime, base_target_r, current_price):
        atr = cls.value(five, "ATR")
        entry = current_price
        if entry is None or atr is None or entry <= 0 or atr <= 0:
            return {"success": False, "reason": "V3 cannot build a valid 5m entry/ATR plan"}
        distance = max(cls.STOP_ATR * atr, entry * 0.0010)
        fee = cls.FEE_RATE[market]
        round_trip = 2.0 * (fee + cls.ADVERSE_RATE)
        risk_rate = distance / entry
        required_target_r = cls.MIN_NET_RR + (1.0 + cls.MIN_NET_RR) * round_trip / risk_rate
        target_r = max(float(base_target_r), required_target_r)
        stop = entry - distance if direction == "LONG" else entry + distance
        target = entry + target_r * distance if direction == "LONG" else entry - target_r * distance
        if not isfinite(stop) or not isfinite(target) or stop <= 0 or target <= 0:
            return {"success": False, "reason": "V3 calculated non-positive stop or target"}
        if direction == "LONG" and not stop < entry < target:
            return {"success": False, "reason": "V3 LONG stop/target geometry is invalid"}
        if direction == "SHORT" and not target < entry < stop:
            return {"success": False, "reason": "V3 SHORT stop/target geometry is invalid"}
        net_reward = target_r * risk_rate - round_trip
        net_risk = risk_rate + round_trip
        net_rr = net_reward / net_risk if net_risk > 0 else 0.0
        if net_reward < cls.MIN_NET_REWARD_RATE or net_rr + 1e-9 < cls.MIN_NET_RR:
            return {
                "success": False, "reason": "V3 5m plan fails the net RR requirement",
                "net_reward_risk": round(net_rr, 5),
            }
        return {
            "success": True, "market": market, "direction": direction,
            "entry": entry, "stop_loss": stop, "take_profit": target,
            "atr": atr, "risk_distance": distance, "reward_risk": target_r,
            "net_reward_risk": round(net_rr, 5), "setup": setup,
            "market_regime": regime, "estimated_round_trip_cost_rate": round_trip,
            "estimated_net_reward_rate": net_reward,
            "strategy": market + "_scalper_v3_5m", "strategy_version": "v3",
            "strategy_profile": cls.PROFILE, "stop_atr_multiple": cls.STOP_ATR,
            "timeframe": cls.TIMEFRAME, "signal_timeframe": cls.TIMEFRAME,
            "max_hold_minutes": 0,
        }


class FuturesScalpingStrategyV3(_V3Base):
    def analyze(self, data, costs=None):
        trigger, fifteen, hourly = (data.get(x) for x in ("5m", "15m", "1h"))
        if not self.frame_is_closed_5m(trigger):
            return {"success": False, "reason": "V3 requires at least 60 closed 5m candles"}
        if (not self.frame_is_closed(fifteen, "15m")
                or not self.frame_is_closed(hourly, "1h")):
            return {"success": False, "reason": "V3 Futures requires confirmed closed 15m and 1h context"}
        v = self.value
        c, o, h, low, ema9, ema20, atr, rsi = (
            v(trigger, x) for x in ("close", "open", "high", "low", "EMA_9", "EMA_20", "ATR", "RSI")
        )
        hist, prev, adx5 = v(trigger, "MACD_hist"), v(trigger, "MACD_hist", -2), v(trigger, "ADX")
        if None in (c, o, h, low, ema9, ema20, atr, rsi, hist, prev, adx5) or atr <= 0:
            return {"success": False, "reason": "V3 Futures 5m indicators are incomplete"}
        shape = self.candle_shape(trigger)
        vr = self.volume_ratio(trigger)
        if shape is None or vr is None:
            return {"success": False, "reason": "V3 Futures 5m candle/volume profile unavailable"}
        body = abs(c-o) / atr
        loc = shape["close_location"]
        prior_high = float(trigger["high"].iloc[-self.BREAKOUT_LOOKBACK-1:-1].max())
        prior_low = float(trigger["low"].iloc[-self.BREAKOUT_LOOKBACK-1:-1].min())
        bull, bear = self.bias(fifteen, hourly, "LONG"), self.bias(fifteen, hourly, "SHORT")

        pull_long = (
            bull and low <= ema20 + .12*atr and c > ema9 and c > o
            and 45 <= rsi <= 67 and hist >= prev
            and loc >= self.PULLBACK_CLOSE_LOCATION
            and body >= self.PULLBACK_BODY_ATR and vr >= self.PULLBACK_VOLUME
            and shape["lower_wick_range"] >= .16
        )
        pull_short = (
            bear and h >= ema20 - .12*atr and c < ema9 and c < o
            and 33 <= rsi <= 55 and hist <= prev
            and loc <= 1-self.PULLBACK_CLOSE_LOCATION
            and body >= self.PULLBACK_BODY_ATR and vr >= self.PULLBACK_VOLUME
            and shape["upper_wick_range"] >= .16
        )
        breakout_long = (
            bull and c > prior_high + .05*atr and c > o
            and 52 <= rsi <= 74 and loc >= self.BREAKOUT_CLOSE_LOCATION
            and body >= self.BREAKOUT_BODY_ATR and vr >= self.BREAKOUT_VOLUME
            and shape["upper_wick_range"] <= self.BREAKOUT_MAX_OPPOSING_WICK
        )
        breakout_short = (
            bear and c < prior_low - .05*atr and c < o
            and 26 <= rsi <= 48 and loc <= 1-self.BREAKOUT_CLOSE_LOCATION
            and body >= self.BREAKOUT_BODY_ATR and vr >= self.BREAKOUT_VOLUME
            and shape["lower_wick_range"] <= self.BREAKOUT_MAX_OPPOSING_WICK
        )
        setups = (
            (pull_long, "LONG", "PULLBACK_5M", "BULL_TREND", 1.60),
            (pull_short, "SHORT", "PULLBACK_5M", "BEAR_TREND", 1.60),
            (breakout_long, "LONG", "BREAKOUT_5M", "BULL_TREND", 1.80),
            (breakout_short, "SHORT", "BREAKOUT_5M", "BEAR_TREND", 1.80),
        )
        for valid, direction, setup, regime, target_r in setups:
            if valid:
                return self.plan("futures", direction, trigger, setup, regime, target_r, c)

        bb_low, bb_high = v(trigger, "BB_lower"), v(trigger, "BB_upper")
        range_market = v(fifteen, "ADX")
        if (range_market is not None and range_market <= 18
                and loc >= .72 and c > o and rsi <= 30 and vr >= .80
                and shape["lower_wick_range"] >= self.REJECTION_WICK_RANGE
                and bb_low is not None and c <= bb_low*1.002):
            return self.plan("futures", "LONG", trigger, "RANGE_REJECTION_5M", "RANGE", 1.55, c)
        if (range_market is not None and range_market <= 18
                and loc <= .28 and c < o and rsi >= 70 and vr >= .80
                and shape["upper_wick_range"] >= self.REJECTION_WICK_RANGE
                and bb_high is not None and c >= bb_high*.998):
            return self.plan("futures", "SHORT", trigger, "RANGE_REJECTION_5M", "RANGE", 1.55, c)
        return {"success": False, "reason": "No qualifying V3 Futures 5m setup"}


class SpotScalpingStrategyV3(_V3Base):
    def analyze(self, data, costs=None):
        trigger, fifteen, hourly, four = (data.get(x) for x in ("5m", "15m", "1h", "4h"))
        if not self.frame_is_closed_5m(trigger):
            return {"success": False, "reason": "V3 requires at least 60 closed 5m candles"}
        if (not self.frame_is_closed(fifteen, "15m")
                or not self.frame_is_closed(hourly, "1h")):
            return {"success": False, "reason": "V3 Spot requires confirmed closed 15m and 1h context"}
        v = self.value
        c, o, h, low, ema9, ema20, atr, rsi = (
            v(trigger, x) for x in ("close", "open", "high", "low", "EMA_9", "EMA_20", "ATR", "RSI")
        )
        hist, prev = v(trigger, "MACD_hist"), v(trigger, "MACD_hist", -2)
        c1, e200 = v(hourly, "close"), v(hourly, "EMA_200")
        if None in (c, o, h, low, ema9, ema20, atr, rsi, hist, prev, c1, e200) or atr <= 0:
            return {"success": False, "reason": "V3 Spot indicators are incomplete"}
        if four is not None and len(four) >= 210:
            if not self.frame_is_closed(four, "4h"):
                return {"success": False, "reason": "V3 Spot requires confirmed closed 4h context"}
            c4, e2004 = v(four, "close"), v(four, "EMA_200")
            if c4 is not None and e2004 and c4 < e2004*.985 and c1 < e200:
                return {"success": False, "reason": "Deep 4h and 1h bearish context"}
        shape = self.candle_shape(trigger)
        vr = self.volume_ratio(trigger)
        if shape is None or vr is None:
            return {"success": False, "reason": "V3 Spot 5m candle/volume profile unavailable"}
        body = abs(c-o) / atr
        loc = shape["close_location"]
        prior_high = float(trigger["high"].iloc[-self.BREAKOUT_LOOKBACK-1:-1].max())
        c15, e21_15 = v(fifteen, "close"), v(fifteen, "EMA_21")
        bullish = self.bias(fifteen, hourly, "LONG") or (
            c1 > e200 and c15 is not None and e21_15 is not None and c15 >= e21_15
        )
        pull = (
            bullish and low <= ema20+.14*atr and c > ema9 and c > o
            and 44 <= rsi <= 68 and hist >= prev
            and loc >= .62 and body >= .10 and vr >= .82
            and shape["lower_wick_range"] >= .16
        )
        breakout = (
            bullish and c > prior_high+.05*atr and c > o
            and 52 <= rsi <= 74 and loc >= .75
            and body >= .18 and vr >= 1.05
            and shape["upper_wick_range"] <= .18
        )
        if pull:
            return self.plan("spot", "LONG", trigger, "PULLBACK_5M", "BULL_TREND", 1.65, c)
        if breakout:
            return self.plan("spot", "LONG", trigger, "BREAKOUT_5M", "BULL_TREND", 1.85, c)
        adx15, bb = v(fifteen, "ADX"), v(trigger, "BB_lower")
        if (adx15 is not None and adx15 <= 18 and bb is not None
                and rsi <= 30 and c <= bb*1.002 and c > o
                and loc >= .72 and vr >= .80
                and shape["lower_wick_range"] >= self.REJECTION_WICK_RANGE):
            return self.plan("spot", "LONG", trigger, "RANGE_BOUNCE_5M", "RANGE", 1.55, c)
        return {"success": False, "reason": "No qualifying V3 Spot 5m setup"}
