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
    def prefilter_5m(cls, frame):
        """Fast 5m-only candidate gate; it never approves or opens a trade.

        It mirrors the 5m-local clauses of V3. Higher-timeframe bias, final
        setup selection and the authoritative trade plan remain in V3.
        """
        if not cls.frame_is_closed_5m(frame):
            return {"candidate": False, "setups": [], "reason": "Closed 5m cache unavailable"}

        v = cls.value
        c, o, h, low, ema9, ema20, atr, rsi = (
            v(frame, x) for x in ("close", "open", "high", "low", "EMA_9", "EMA_20", "ATR", "RSI")
        )
        hist, previous_hist = v(frame, "MACD_hist"), v(frame, "MACD_hist", -2)
        if None in (c, o, h, low, ema9, ema20, atr, rsi, hist, previous_hist) or atr <= 0:
            return {"candidate": False, "setups": [], "reason": "5m indicators incomplete"}

        shape = cls.candle_shape(frame)
        volume_ratio = cls.volume_ratio(frame)
        if shape is None or volume_ratio is None:
            return {"candidate": False, "setups": [], "reason": "5m candle/volume profile unavailable"}

        try:
            prior_high = float(frame["high"].iloc[-cls.BREAKOUT_LOOKBACK-1:-1].max())
            prior_low = float(frame["low"].iloc[-cls.BREAKOUT_LOOKBACK-1:-1].min())
        except (TypeError, ValueError, KeyError, IndexError):
            return {"candidate": False, "setups": [], "reason": "5m breakout range unavailable"}

        body = abs(c - o) / atr
        location = shape["close_location"]
        lower_wick = shape["lower_wick_range"]
        upper_wick = shape["upper_wick_range"]
        setups = []

        if cls.MARKET == "futures":
            if (low <= ema20 + .12 * atr and c > ema9 and c > o
                    and 45 <= rsi <= 67 and hist >= previous_hist
                    and location >= cls.PULLBACK_CLOSE_LOCATION
                    and body >= cls.PULLBACK_BODY_ATR
                    and volume_ratio >= cls.PULLBACK_VOLUME
                    and lower_wick >= .16):
                setups.append("PULLBACK_LONG")
            if (h >= ema20 - .12 * atr and c < ema9 and c < o
                    and 33 <= rsi <= 55 and hist <= previous_hist
                    and location <= 1 - cls.PULLBACK_CLOSE_LOCATION
                    and body >= cls.PULLBACK_BODY_ATR
                    and volume_ratio >= cls.PULLBACK_VOLUME
                    and upper_wick >= .16):
                setups.append("PULLBACK_SHORT")
            if (c > prior_high + .05 * atr and c > o
                    and 52 <= rsi <= 74
                    and location >= cls.BREAKOUT_CLOSE_LOCATION
                    and body >= cls.BREAKOUT_BODY_ATR
                    and volume_ratio >= cls.BREAKOUT_VOLUME
                    and upper_wick <= cls.BREAKOUT_MAX_OPPOSING_WICK):
                setups.append("BREAKOUT_LONG")
            if (c < prior_low - .05 * atr and c < o
                    and 26 <= rsi <= 48
                    and location <= 1 - cls.BREAKOUT_CLOSE_LOCATION
                    and body >= cls.BREAKOUT_BODY_ATR
                    and volume_ratio >= cls.BREAKOUT_VOLUME
                    and lower_wick <= cls.BREAKOUT_MAX_OPPOSING_WICK):
                setups.append("BREAKOUT_SHORT")
            bb_low, bb_high = v(frame, "BB_lower"), v(frame, "BB_upper")
            if (rsi <= 30 and bb_low is not None and c <= bb_low * 1.002
                    and c > o and location >= .72 and volume_ratio >= .80
                    and lower_wick >= cls.REJECTION_WICK_RANGE):
                setups.append("RANGE_REJECTION_LONG")
            if (rsi >= 70 and bb_high is not None and c >= bb_high * .998
                    and c < o and location <= .28 and volume_ratio >= .80
                    and upper_wick >= cls.REJECTION_WICK_RANGE):
                setups.append("RANGE_REJECTION_SHORT")
        elif cls.MARKET == "spot":
            if (low <= ema20 + .14 * atr and c > ema9 and c > o
                    and 44 <= rsi <= 68 and hist >= previous_hist
                    and location >= .62 and body >= .10
                    and volume_ratio >= .82 and lower_wick >= .16):
                setups.append("PULLBACK_LONG")
            if (c > prior_high + .05 * atr and c > o
                    and 52 <= rsi <= 74 and location >= .75
                    and body >= .18 and volume_ratio >= 1.05
                    and upper_wick <= .18):
                setups.append("BREAKOUT_LONG")
            bb_low = v(frame, "BB_lower")
            if (rsi <= 30 and bb_low is not None and c <= bb_low * 1.002
                    and c > o and location >= .72 and volume_ratio >= .80
                    and lower_wick >= cls.REJECTION_WICK_RANGE):
                setups.append("RANGE_BOUNCE_LONG")

        return {
            "candidate": bool(setups),
            "setups": setups,
            "reason": "" if setups else "No 5m-local V3 trigger",
        }

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
    def nearest_opposing_structure(cls, frames, direction, entry):
        """Find the nearest confirmed swing level opposing a proposed entry.

        Only confirmed pivots (two closed candles on each side) are considered.
        Higher-timeframe levels are included because a 5m trigger must not buy
        directly into 15m/1h supply or sell directly into 15m/1h demand.
        """
        candidates = []
        windows = (("5m", 72), ("15m", 64), ("1h", 48))
        for timeframe, lookback in windows:
            frame = (frames or {}).get(timeframe)
            if frame is None or len(frame) < 8:
                continue
            if not cls.frame_is_closed(frame, timeframe):
                continue
            column = "high" if direction == "LONG" else "low"
            try:
                values = [float(value) for value in frame[column].tail(lookback)]
            except (KeyError, TypeError, ValueError):
                continue
            start = max(2, len(values) - lookback)
            for index in range(start, len(values) - 2):
                level = values[index]
                left = values[index-2:index]
                right = values[index+1:index+3]
                if direction == "LONG":
                    pivot = (level >= max(left) and level >= max(right)\n                             and (level > max(left) or level > max(right))\n                             and level > entry)
                    distance = level - entry
                else:
                    pivot = (level <= min(left) and level <= min(right)\n                             and (level < min(left) or level < min(right))\n                             and level < entry)
                    distance = entry - level
                if pivot and distance > 0:
                    candidates.append((distance, level, timeframe))
        if not candidates:
            return None
        distance, level, timeframe = min(candidates, key=lambda item: item[0])
        return {"level": level, "distance": distance, "timeframe": timeframe}

    @classmethod
    def plan_with_structure(cls, market, direction, five, frames, setup,
                             regime, base_target_r, current_price):
        """Build a V3 plan and veto entries with inadequate room to S/R."""
        plan = cls.plan(market, direction, five, setup, regime,
                        base_target_r, current_price)
        if not plan.get("success"):
            return plan
        atr = float(plan["atr"])
        entry = float(plan["entry"])
        structure = cls.nearest_opposing_structure(frames, direction, entry)
        if structure is None:
            # No confirmed opposing swing means there is no identified nearby
            # wall in the supplied closed context. The required 5m/15m/1h
            # frames were already validated by analyze(); keep the trade
            # eligible and expose that no overhead/underfoot pivot was found.
            plan["structure_guard"] = {
                "blocked": False, "direction": direction, "level": None,
                "timeframe": None, "distance": None,
                "reason": "No confirmed opposing swing level in closed context",
            }
            return plan
        level = float(structure["level"])
        buffer = max(0.65 * atr, entry * 0.0008)
        target = float(plan["take_profit"])
        if direction == "LONG":
            clearance = level - entry
            target_blocked = target >= level - 0.20 * atr
        else:
            clearance = entry - level
            target_blocked = target <= level + 0.20 * atr
        if clearance <= buffer or target_blocked:
            return {
                "success": False,
                "reason": (
                    "V3 structure veto: %s entry blocked by %s resistance/support "
                    "at %.10g (clearance %.3f ATR; target lacks free space)"
                    % (direction, structure["timeframe"], level, clearance / atr)
                ),
                "structure_guard": {
                    "blocked": True, "direction": direction, "level": level,
                    "timeframe": structure["timeframe"],
                    "distance": clearance, "distance_atr": round(clearance / atr, 4),
                    "buffer": buffer, "planned_target": target,
                },
            }
        plan["structure_guard"] = {
            "blocked": False, "direction": direction, "level": level,
            "timeframe": structure["timeframe"],
            "distance": clearance, "distance_atr": round(clearance / atr, 4),
            "buffer": buffer,
        }
        return plan

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
    MARKET = "futures"

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
                return self.plan_with_structure("futures", direction, trigger, data, setup, regime, target_r, c)

        bb_low, bb_high = v(trigger, "BB_lower"), v(trigger, "BB_upper")
        range_market = v(fifteen, "ADX")
        if (range_market is not None and range_market <= 18
                and loc >= .72 and c > o and rsi <= 30 and vr >= .80
                and shape["lower_wick_range"] >= self.REJECTION_WICK_RANGE
                and bb_low is not None and c <= bb_low*1.002):
            return self.plan_with_structure("futures", "LONG", trigger, data, "RANGE_REJECTION_5M", "RANGE", 1.55, c)
        if (range_market is not None and range_market <= 18
                and loc <= .28 and c < o and rsi >= 70 and vr >= .80
                and shape["upper_wick_range"] >= self.REJECTION_WICK_RANGE
                and bb_high is not None and c >= bb_high*.998):
            return self.plan_with_structure("futures", "SHORT", trigger, data, "RANGE_REJECTION_5M", "RANGE", 1.55, c)
        return {"success": False, "reason": "No qualifying V3 Futures 5m setup"}


class SpotScalpingStrategyV3(_V3Base):
    MARKET = "spot"

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
            return self.plan_with_structure("spot", "LONG", trigger, data, "PULLBACK_5M", "BULL_TREND", 1.65, c)
        if breakout:
            return self.plan_with_structure("spot", "LONG", trigger, data, "BREAKOUT_5M", "BULL_TREND", 1.85, c)
        adx15, bb = v(fifteen, "ADX"), v(trigger, "BB_lower")
        if (adx15 is not None and adx15 <= 18 and bb is not None
                and rsi <= 30 and c <= bb*1.002 and c > o
                and loc >= .72 and vr >= .80
                and shape["lower_wick_range"] >= self.REJECTION_WICK_RANGE):
            return self.plan_with_structure("spot", "LONG", trigger, data, "RANGE_BOUNCE_5M", "RANGE", 1.55, c)
        return {"success": False, "reason": "No qualifying V3 Spot 5m setup"}
