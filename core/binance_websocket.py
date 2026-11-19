"""Binance multi-timeframe candle cache and event clock for V3.

5m closes trigger strategy decisions. Closed 15m/1h/4h candles are cached
alongside 5m candles so the decision path does not make per-symbol REST calls
at the next-candle execution boundary.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
from websocket import WebSocketApp


class BinanceKlineStream:
    BASE_URLS = {
        # Spot market-data-only endpoint is the official public stream mirror;
        # no user-data streams are consumed by this application.
        "spot": "wss://data-stream.binance.vision:443/ws",
        "futures": "wss://fstream.binance.com/market/stream",
    }
    CACHE_LIMITS = {
        "futures": {"5m": 2412, "15m": 800, "1h": 200},
        "spot": {"5m": 3000, "15m": 800, "1h": 200, "4h": 210},
    }
    REQUIRED_ROWS = {
        "futures": {"5m": 60, "15m": 60, "1h": 60},
        # Spot's 4h EMA200 is an optional macro filter for newer listings.
        "spot": {"5m": 60, "15m": 60, "1h": 200},
    }
    MAX_STREAMS_PER_CONNECTION = 1020
    # Binance sends server ping frames; websocket-client auto-replies with pong.
    # Disable client-originated ping timers to avoid false ping/pong timeouts on mobile links.
    CLIENT_PING_INTERVAL_SECONDS = 0
    INTERVAL_MS = {"5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000}

    def __init__(self, market_feed, on_kline, max_symbols=None,
                 refresh_seconds=1800, reconnect_seconds=3, on_universe=None,
                 on_cache_ready=None):
        self.feed = market_feed
        self.on_kline = on_kline
        self.on_universe = on_universe
        self.on_cache_ready = on_cache_ready
        # Production defaults to the 30 most liquid USDT pairs per market.
        # Zero remains an explicit full-market setting for isolated tests only.
        explicit_test_limit = max_symbols is not None
        raw_limit = (
            max_symbols if explicit_test_limit
            else os.environ.get("AI_TRADER_WS_MAX_SYMBOLS", "30")
        )
        try:
            parsed_limit = int(raw_limit)
            # An explicit zero is retained only for isolated unit fixtures.
            # Environment/runtime configuration is always bounded to top 30.
            self.max_symbols = (
                0 if explicit_test_limit and parsed_limit == 0
                else 30 if parsed_limit <= 0
                else min(30, parsed_limit)
            )
        except (TypeError, ValueError):
            self.max_symbols = 30
        self.full_universe = explicit_test_limit and self.max_symbols == 0
        # Every symbol streams only 5m. Higher timeframes are derived locally
        # from the 5m cache; Spot's long 1h seed and optional 4h macro frame
        # are bootstrapped/refreshed separately, never fetched per signal.
        self.fast_scan_mode = True
        try:
            warmup_workers = int(os.environ.get("AI_TRADER_WS_WARMUP_WORKERS", "6"))
        except (TypeError, ValueError):
            warmup_workers = 4
        self.warmup_workers = min(8, max(1, warmup_workers))
        self.refresh_seconds = max(300, int(refresh_seconds))
        self.reconnect_seconds = max(1, int(reconnect_seconds))
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._cache_lock = threading.RLock()
        self._apps = {}
        self._threads = {}
        # Exactly one long-lived cache warmup worker per market. WebSocket
        # reconnects must not spawn duplicate REST warmup pools.
        self._warmup_threads = {}
        self._thread_factory = threading.Thread
        self._connection_threads = []
        self._connection_live = {}
        self._symbol_connection = {}
        self._market_symbols = {"spot": set(), "futures": set()}
        self._cache = {}
        self._cache_ready = set()
        self._last_4h_refresh_at = {}
        self._last_4h_refresh_attempt_at = {}
        self._last_open = {}
        self._closed_seen = set()
        self._latest_candles = {}
        self._warm_errors = {}
        self._status = {
            market: {
                "connected": False, "fully_connected": False, "state": "stopped",
                "symbols": 0, "total_chunks": 0, "connected_chunks": 0,
                "last_event": None, "last_closed_candle": None,
                "reconnects": 0, "watchdog_restarts": 0, "last_error": None,
                "cache_ready_symbols": 0, "cache_warmup_total": 0,
                "cache_warmup_completed": 0, "cache_warmup_errors": 0,
                "cache_warmup_worker_starts": 0,
                "cache_not_ready_reasons": {},
                "cache_not_ready_samples": [],
                "cache_warmup_last_error": None,
                "last_close_code": None, "last_close_reason": None,
                "last_disconnect_at": None, "reconnect_delay_seconds": 3,
                "ranking_day": None,
            }
            for market in ("spot", "futures")
        }

    def status(self):
        with self._lock:
            snapshot = json.loads(json.dumps(self._status))
            snapshot["latest_candle_symbols"] = {
                market: sum(1 for key in self._latest_candles if key[0] == market)
                for market in ("spot", "futures")
            }
            snapshot["symbol_mode"] = "full_universe" if self.full_universe else "top_n"
            snapshot["selection_limit"] = self.max_symbols or None
            snapshot["fast_scan_mode"] = self.fast_scan_mode
            snapshot["active_intervals"] = {
                market: self._active_intervals(market)
                for market in ("spot", "futures")
            }
            for market in ("spot", "futures"):
                warmup_worker = self._warmup_threads.get(market)
                snapshot[market]["cache_warmup_worker_alive"] = bool(
                    warmup_worker and warmup_worker.is_alive()
                )
                last_event = snapshot[market].get("last_event")
                try:
                    stamp = datetime.fromisoformat(last_event)
                    age = max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds())
                except (TypeError, ValueError):
                    age = None
                snapshot[market]["last_event_age_seconds"] = age
                snapshot[market]["watchdog_healthy"] = bool(
                    snapshot[market].get("fully_connected")
                    and age is not None and age <= 360
                )
            return snapshot

    def latest_candle(self, market, symbol):
        key = (str(market).strip().lower(), str(symbol).strip().upper())
        with self._lock:
            value = self._latest_candles.get(key)
            return dict(value) if value else None

    def symbol_connected(self, market, symbol):
        key = (str(market).strip().lower(), str(symbol).strip().upper())
        with self._lock:
            index = self._symbol_connection.get(key)
            return index is not None and bool(
                self._connection_live.get((key[0], index), False))

    def registered_symbols(self, market):
        """Return the current eligible 5m universe for one market."""
        market = str(market).strip().lower()
        if market not in self._market_symbols:
            return []
        with self._lock:
            return sorted(self._market_symbols[market])

    def _prune_market_state(self, market, symbols):
        """Discard candle state for pairs removed from the daily top-N universe."""
        market = str(market).strip().lower()
        keep = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}

        def is_removed(key):
            return (
                isinstance(key, tuple) and len(key) >= 2
                and key[0] == market and key[1] not in keep
            )

        with self._lock:
            self._cache_ready = {
                key for key in self._cache_ready
                if not is_removed(key)
            }
            self._latest_candles = {
                key: value for key, value in self._latest_candles.items()
                if not is_removed(key)
            }
            self._last_open = {
                key: value for key, value in self._last_open.items()
                if not is_removed(key)
            }
            self._closed_seen = {
                key for key in self._closed_seen
                if not is_removed(key)
            }
            self._warm_errors = {
                key: value for key, value in self._warm_errors.items()
                if not is_removed(key)
            }
            state = self._status[market]
            ready_count = sum(1 for key in self._cache_ready if key[0] == market)
            state["cache_ready_symbols"] = ready_count
            state["cache_warmup_completed"] = ready_count
            state["cache_warmup_total"] = len(keep)
            state["cache_warmup_errors"] = sum(
                1 for key in self._warm_errors if key[0] == market
            )

        with self._cache_lock:
            self._cache = {
                key: value for key, value in self._cache.items()
                if key[0] != market or key[1] in keep
            }

    def latest_closed_events(self, market, expected_open_time_ms, symbols=None):
        """Build cache-backed audit events for the exact latest closed 5m candle.

        This is a bounded fallback for symbols whose close notification was
        missed. It reads the already-warmed Binance cache only (no REST calls),
        and marks delayed candles audit-only so an old cache snapshot cannot
        create a late entry.
        """
        market = str(market).strip().lower()
        if market not in self._market_symbols:
            return {"registered": 0, "requested": 0, "events": [],
                    "missing_symbols": []}
        expected_open_time_ms = int(expected_open_time_ms or 0)
        with self._lock:
            registered = set(self._market_symbols[market])
        requested = registered if symbols is None else (
            registered.intersection(
                str(symbol).strip().upper() for symbol in symbols
                if str(symbol).strip()
            )
        )
        events = []
        missing = []
        now_ms = int(time.time() * 1000)
        for symbol in sorted(requested):
            try:
                frames = self._snapshot(market, symbol)
                frame = (frames or {}).get("5m")
                if frame is None or frame.empty:
                    missing.append(symbol)
                    continue
                # Select the requested cycle from the cached history rather
                # than assuming the final row is always that cycle. A newer
                # closed row can arrive while reconciliation is processing a
                # batch; rejecting the whole cache in that case loses a valid
                # exact-cycle candle that is still present in the frame.
                opens = pd.to_datetime(frame["open_time"], utc=True)
                matching = frame.loc[
                    opens.map(lambda value: self._row_key(value))
                    == expected_open_time_ms
                ]
                if matching.empty:
                    missing.append(symbol)
                    continue
                row = matching.iloc[-1]
                if not bool(row.get("is_closed", False)):
                    missing.append(symbol)
                    continue
                open_ms = self._row_key(row["open_time"])
                close_ms = self._row_key(row["close_time"])
                if open_ms != expected_open_time_ms or close_ms > now_ms + 10000:
                    missing.append(symbol)
                    continue
                age_ms = max(0, now_ms - close_ms)
                if age_ms > 2 * self.INTERVAL_MS["5m"] + 30000:
                    missing.append(symbol)
                    continue
                events.append({
                    "symbol": symbol,
                    "interval": "5m",
                    "open_time": open_ms,
                    "close_time": close_ms,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    # A cache reconciliation event is anchored to the candle
                    # close, not to the time this fallback happened to run.
                    "event_time": close_ms + 1,
                    "is_closed": True,
                    "first_update": False,
                    "source": "binance_cache_reconciliation",
                    "market_data": frames,
                    "cache_ready": True,
                    "audit_only": age_ms > 30000,
                })
            except Exception:
                missing.append(symbol)
        return {
            "registered": len(registered),
            "requested": len(requested),
            "available": len(events),
            "events": events,
            "missing_symbols": missing,
        }

    def _active_intervals(self, market):
        # One canonical 5m stream per symbol. Higher timeframes are aggregated
        # from the local cache, so websocket subscriptions do not duplicate data.
        return ["5m"]

    def start(self):
        if any(thread.is_alive() for thread in self._threads.values()):
            return self.status()
        self._stop.clear()
        for market in ("spot", "futures"):
            thread = threading.Thread(
                target=self._run_market, args=(market,),
                name="binance-ws-" + market, daemon=True,
            )
            self._threads[market] = thread
            thread.start()
        refresh = threading.Thread(
            target=self._refresh_loop, name="binance-ws-refresh", daemon=True
        )
        self._threads["refresh"] = refresh
        refresh.start()
        watchdog = threading.Thread(
            target=self._watchdog_loop, name="binance-ws-watchdog", daemon=True
        )
        self._threads["watchdog"] = watchdog
        watchdog.start()
        return self.status()

    def stop(self):
        self._stop.set()
        with self._lock:
            apps = list(self._apps.values())
        for app in apps:
            try:
                app.close()
            except Exception:
                pass
        with self._lock:
            for value in self._status.values():
                value["connected"] = False
                value["fully_connected"] = False
                value["state"] = "stopping"

    def _watchdog_loop(self):
        """Reconnect a silently stalled market stream and revive dead supervisors."""
        while not self._stop.wait(30):
            now = datetime.now(timezone.utc)
            for market in ("spot", "futures"):
                with self._lock:
                    supervisor = self._threads.get(market)
                    if supervisor is None or not supervisor.is_alive():
                        self._threads[market] = threading.Thread(
                            target=self._run_market, args=(market,),
                            name="binance-ws-" + market, daemon=True,
                        )
                        self._threads[market].start()
                        self._status[market]["watchdog_restarts"] += 1
                        self._status[market]["state"] = "watchdog_restarted"
                        continue
                    last_event = self._status[market].get("last_event")
                    apps = [app for (app_market, _), app in self._apps.items()
                            if app_market == market]
                try:
                    stamp = datetime.fromisoformat(last_event) if last_event else None
                    stale = stamp is None or (now - stamp).total_seconds() > 360
                except (TypeError, ValueError):
                    stale = True
                if stale and apps:
                    with self._lock:
                        self._status[market]["watchdog_restarts"] += 1
                        self._status[market]["state"] = "watchdog_stale_stream"
                        self._status[market]["connected"] = False
                        self._status[market]["fully_connected"] = False
                    for app in apps:
                        try:
                            app.close()
                        except Exception:
                            pass

    def _refresh_loop(self):
        while not self._stop.wait(self.refresh_seconds):
            with self._lock:
                apps = list(self._apps.values())
            for app in apps:
                try:
                    app.close()
                except Exception:
                    pass

    def _load_symbols(self, market):
        top_selector = getattr(self.feed, "fetch_top_symbols", None)
        if self.max_symbols > 0 and callable(top_selector):
            ranked = top_selector(market, limit=self.max_symbols)
            symbols = [
                str(item.get("symbol", "")).strip().upper()
                for item in ranked if isinstance(item, dict)
            ]
            symbols = [symbol for symbol in symbols if symbol]
            if len(symbols) != self.max_symbols:
                raise RuntimeError(
                    "Daily %s universe must contain exactly %s symbols; received %s"
                    % (market, self.max_symbols, len(symbols))
                )
            ranking_day = (
                str(ranked[0].get("ranking_day") or "")
                if ranked else ""
            )
            with self._lock:
                self._status[market]["ranking_day"] = ranking_day or (
                    datetime.now(timezone.utc).date().isoformat()
                )
            return symbols

        if market == "futures":
            eligible = set(self.feed.fetch_symbols("futures"))
            tickers = self.feed._request("futures", "/fapi/v1/ticker/24hr", {})
        else:
            eligible = set(self.feed.fetch_symbols("spot"))
            tickers = self.feed._request("spot", "/api/v3/ticker/24hr", {})
        ranked = []
        for item in tickers if isinstance(tickers, list) else []:
            symbol = str(item.get("symbol", "")).upper()
            if symbol not in eligible:
                continue
            try:
                volume = float(item.get("quoteVolume", 0) or 0)
                price = float(item.get("lastPrice", 0) or 0)
            except (TypeError, ValueError):
                continue
            if volume > 0 and price > 0:
                ranked.append((volume, symbol))
        ranked.sort(reverse=True)
        symbols = [
            symbol for _, symbol in (
                ranked[:self.max_symbols] if self.max_symbols > 0 else ranked
            )
        ]
        if not symbols:
            raise RuntimeError("Binance returned no eligible liquid symbols for " + market)
        return symbols

    def _register_universe(self, market, symbols):
        """Notify the strategy runtime of the complete eligible market universe."""
        if callable(self.on_universe):
            return self.on_universe(market, list(symbols))
        return None

    @staticmethod
    def _row_key(value):
        stamp = value
        if not isinstance(stamp, pd.Timestamp):
            stamp = pd.to_datetime(stamp, utc=True)
        return int(stamp.timestamp() * 1000)

    def _merge_frame(self, market, symbol, interval, frame):
        if frame is None or frame.empty:
            return
        key = (market, symbol, interval)
        incoming = frame.copy()
        if "open_time" not in incoming.columns:
            return
        incoming["open_time"] = pd.to_datetime(incoming["open_time"], utc=True)
        incoming["close_time"] = pd.to_datetime(incoming["close_time"], utc=True)
        incoming["is_closed"] = incoming["is_closed"].astype(bool)
        with self._cache_lock:
            current = self._cache.get(key)
            merged = incoming if current is None else pd.concat(
                [current, incoming], ignore_index=True)
            merged = merged.drop_duplicates(subset=["open_time"], keep="last")
            merged = merged.sort_values("open_time").tail(
                self.CACHE_LIMITS[market][interval]).reset_index(drop=True)
            merged.attrs.update({
                "market": market, "symbol": symbol,
                "interval": interval, "timeframe": interval,
                "source": "binance_websocket_cache",
            })
            self._cache[key] = merged
        if interval == "5m":
            self._rebuild_derived_frames(market, symbol)

    def _aggregate_5m_frame(self, market, symbol, timeframe, frame):
        """Aggregate observed closed 5m bars into aligned higher-timeframe bars.

        Binance omits 5m klines for intervals with no trades. Those absent
        intervals are valid market data, not missing candles to fabricate.
        Aggregate the real observations inside each completed 15m/1h bucket
        instead of requiring exactly 3/12 source rows.
        """
        if frame is None or frame.empty or timeframe not in ("15m", "1h"):
            return pd.DataFrame()
        duration = self.INTERVAL_MS[timeframe]
        source = frame.loc[frame["is_closed"].astype(bool)].copy()
        if source.empty:
            return pd.DataFrame()
        source["open_time"] = pd.to_datetime(source["open_time"], utc=True)
        source["close_time"] = pd.to_datetime(source["close_time"], utc=True)
        source = source.sort_values("open_time").drop_duplicates("open_time", keep="last")
        frequency = "15min" if timeframe == "15m" else "1h"
        current_bucket = pd.Timestamp.now(tz="UTC").floor(frequency)
        source = source.loc[source["open_time"] < current_bucket]
        source["bucket_open"] = source["open_time"].dt.floor(frequency)
        rows = []
        for bucket_open, group in source.groupby("bucket_open", sort=True):
            group = group.sort_values("open_time")
            if group.empty or not bool(group["is_closed"].astype(bool).all()):
                continue
            bucket_end = bucket_open + pd.Timedelta(milliseconds=duration)
            row = {
                "open_time": bucket_open,
                "open": float(group["open"].iloc[0]),
                "high": float(group["high"].max()),
                "low": float(group["low"].min()),
                "close": float(group["close"].iloc[-1]),
                "volume": float(group["volume"].sum()),
                "close_time": bucket_end - pd.Timedelta(milliseconds=1),
                "is_closed": True,
            }
            for column in ("quote_volume", "trades", "taker_buy_base", "taker_buy_quote"):
                if column in group.columns:
                    row[column] = float(group[column].sum())
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        result.attrs.update({
            "market": market, "symbol": symbol, "interval": timeframe,
            "timeframe": timeframe, "source": "derived_from_binance_5m_cache",
        })
        return result

    def _rebuild_derived_frames(self, market, symbol):
        """Merge locally aggregated 15m/1h bars into the existing history cache."""
        with self._cache_lock:
            base = self._cache.get((market, symbol, "5m"))
            if base is None or base.empty:
                return
            base = base.copy()
        for timeframe in ("15m", "1h"):
            derived = self._aggregate_5m_frame(market, symbol, timeframe, base)
            if derived.empty:
                continue
            key = (market, symbol, timeframe)
            with self._cache_lock:
                current = self._cache.get(key)
                merged = derived if current is None else pd.concat(
                    [current, derived], ignore_index=True)
                merged = merged.drop_duplicates(
                    subset=["open_time"], keep="last")
                merged = merged.sort_values("open_time").tail(
                    self.CACHE_LIMITS[market][timeframe]).reset_index(drop=True)
                merged.attrs.update({
                    "market": market, "symbol": symbol,
                    "interval": timeframe, "timeframe": timeframe,
                    "source": "derived_from_binance_5m_cache",
                })
                self._cache[key] = merged

    def _refresh_spot_4h_if_due(self, symbol, force=False):
        """Refresh the optional Spot 4h macro context at most once per hour."""
        now = time.monotonic()
        last = self._last_4h_refresh_at.get(symbol)
        last_attempt = self._last_4h_refresh_attempt_at.get(symbol)
        if not force and last is not None and now - last < 3600:
            return False
        if not force and last_attempt is not None and now - last_attempt < 60:
            return False
        self._last_4h_refresh_attempt_at[symbol] = now
        try:
            frame = self.feed.fetch_candles(
                "spot", symbol, "4h", limit=self.CACHE_LIMITS["spot"]["4h"],
                closed_only=True)
            self._merge_frame("spot", symbol, "4h", frame)
            self._last_4h_refresh_at[symbol] = now
            return True
        except Exception as error:
            with self._lock:
                self._status["spot"]["cache_warmup_last_error"] = (
                    symbol + " 4h refresh: " + str(error))
            return False

    def _merge_event_candle(self, market, symbol, interval, kline):
        try:
            open_ms, close_ms = int(kline["t"]), int(kline["T"])
            row = {
                "open_time": pd.to_datetime(open_ms, unit="ms", utc=True),
                "open": float(kline["o"]), "high": float(kline["h"]),
                "low": float(kline["l"]), "close": float(kline["c"]),
                "volume": float(kline["v"]),
                "close_time": pd.to_datetime(close_ms, unit="ms", utc=True),
                "quote_volume": float(kline.get("q", 0) or 0),
                "trades": int(kline.get("n", 0) or 0),
                "taker_buy_base": float(kline.get("V", 0) or 0),
                "taker_buy_quote": float(kline.get("Q", 0) or 0),
                "ignore": 0.0, "is_closed": True,
            }
            self._merge_frame(market, symbol, interval, pd.DataFrame([row]))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            with self._lock:
                self._status[market]["last_error"] = "cache candle: " + str(error)

    def _frame_health_reason(self, market, interval, frame, now_ms=None):
        """Return an explicit cache rejection reason without weakening data guards."""
        minimum = self.REQUIRED_ROWS[market].get(
            interval, self.CACHE_LIMITS[market][interval])
        if frame is None:
            return "missing_frame"
        if len(frame) < minimum:
            return "insufficient_rows"
        if "is_closed" not in frame.columns or not bool(frame["is_closed"].iloc[-1]):
            return "latest_candle_not_closed"
        if "open_time" not in frame.columns or "close_time" not in frame.columns:
            return "missing_time_columns"
        try:
            interval_ms = self.INTERVAL_MS[interval]
            opens = pd.to_datetime(frame["open_time"], utc=True)
            closes = pd.to_datetime(frame["close_time"], utc=True)
            expected_delta = pd.Timedelta(milliseconds=interval_ms)
            # Binance may omit intervals with no trades. Such gaps are valid
            # historical market data, not a broken candle series. Preserve the
            # real observations; never fabricate zero-volume candles.
            if len(opens) > 1 and not bool((opens.diff().dropna() > pd.Timedelta(0)).all()):
                return "non_increasing_candle_times"
            if not bool(((closes - opens) == expected_delta - pd.Timedelta(milliseconds=1)).all()):
                return "invalid_candle_duration"
            last_close = int(pd.Timestamp(closes.iloc[-1]).timestamp() * 1000)
            now_ms = int(now_ms or time.time() * 1000)
            if last_close > now_ms + 30000:
                return "future_candle"
            # Historical cache readiness is independent of whether this market
            # traded in the latest interval. Exact-cycle freshness is enforced
            # by latest_closed_events() and by the incoming closed WS candle.
            return None
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
            return "invalid_candle_data"

    def _frame_healthy(self, market, interval, frame, now_ms=None):
        return self._frame_health_reason(market, interval, frame, now_ms) is None

    def _snapshot(self, market, symbol, max_age_seconds=None):
        self._rebuild_derived_frames(market, symbol)
        required = self.REQUIRED_ROWS[market]
        frames = {}
        with self._cache_lock:
            for interval, minimum in required.items():
                frame = self._cache.get((market, symbol, interval))
                if not self._frame_healthy(market, interval, frame):
                    return None
                copied = frame.copy()
                copied.attrs.update({
                    "market": market, "symbol": symbol,
                    "interval": interval, "timeframe": interval,
                    "source": "binance_websocket_cache",
                })
                frames[interval] = copied
            if market == "spot":
                optional_four_hour = self._cache.get((market, symbol, "4h"))
                if self._frame_healthy(market, "4h", optional_four_hour):
                    copied = optional_four_hour.copy()
                    copied.attrs.update({
                        "market": market, "symbol": symbol,
                        "interval": "4h", "timeframe": "4h",
                        "source": "binance_websocket_cache",
                    })
                    frames["4h"] = copied
        if max_age_seconds is not None:
            latest_close_ms = self._row_key(
                frames["5m"].iloc[-1]["close_time"])
            age_ms = int(time.time() * 1000) - latest_close_ms
            if age_ms < -5000 or age_ms > int(max_age_seconds * 1000):
                return None
        return frames

    def _warm_symbol(self, market, symbol):
        # One REST bootstrap: 5m only, plus Spot 4h macro context.
        # 15m/1h are derived from the canonical 5m history.
        # Only canonical 5m history participates in cache readiness.
        # Optional Spot 4h context is refreshed independently by _warm_market.
        bootstrap_intervals = ["5m"]
        for interval in bootstrap_intervals:
            limit = self.CACHE_LIMITS[market][interval]
            if self._stop.is_set():
                return False
            with self._cache_lock:
                existing = self._cache.get((market, symbol, interval))
                already_ready = self._frame_healthy(market, interval, existing)
            if already_ready:
                continue
            last_error = None
            for attempt in range(2):
                if self._stop.is_set():
                    return False
                try:
                    if interval == "5m" and limit > 1000:
                        pages = []
                        remaining = int(limit)
                        end_time = None
                        while remaining > 0 and not self._stop.is_set():
                            page_limit = min(1000, remaining)
                            page = self.feed.fetch_candles(
                                market, symbol, "5m", limit=page_limit,
                                closed_only=True, end_time=end_time)
                            if page is None or page.empty:
                                raise RuntimeError("5m bootstrap pagination returned an empty page")
                            pages.append(page)
                            remaining -= len(page)
                            oldest_ms = self._row_key(page["open_time"].iloc[0])
                            end_time = oldest_ms - 1
                            if len(page) < page_limit:
                                break
                        if not pages:
                            raise RuntimeError("5m bootstrap produced no pages")
                        frame = pd.concat(pages, ignore_index=True)
                        frame = frame.drop_duplicates(subset=["open_time"], keep="last")
                        frame = frame.sort_values("open_time").tail(limit).reset_index(drop=True)
                    else:
                        frame = self.feed.fetch_candles(
                            market, symbol, interval, limit=min(1000, limit),
                            closed_only=True)
                    self._merge_frame(market, symbol, interval, frame)
                    last_error = None
                    break
                except Exception as error:
                    last_error = str(error)
                    if attempt == 0 and not self._stop.wait(1):
                        continue
            if last_error:
                with self._lock:
                    self._warm_errors[(market, symbol, interval)] = last_error
                    self._status[market]["cache_warmup_last_error"] = (
                        symbol + " " + interval + ": " + last_error
                    )
            else:
                with self._lock:
                    self._warm_errors.pop((market, symbol, interval), None)
        self._rebuild_derived_frames(market, symbol)
        frames = self._snapshot(market, symbol)
        ready = frames is not None
        with self._lock:
            if ready:
                self._cache_ready.add((market, symbol))
            else:
                self._cache_ready.discard((market, symbol))
            state = self._status[market]
            state["cache_warmup_completed"] = len(
                [key for key in self._cache_ready if key[0] == market])
            state["cache_ready_symbols"] = state["cache_warmup_completed"]
            state["cache_warmup_errors"] = sum(
                1 for key in self._warm_errors if key[0] == market)
        if ready and callable(self.on_cache_ready):
            try:
                self.on_cache_ready(market, symbol, frames)
            except Exception as error:
                with self._lock:
                    self._status[market]["last_error"] = "cache-ready callback: " + str(error)
        return ready

    def _ensure_warmup_worker(self, market):
        """Start at most one cache warmup worker for a market."""
        market = str(market).strip().lower()
        if market not in ("spot", "futures"):
            return False
        with self._lock:
            current = self._warmup_threads.get(market)
            if current is not None and current.is_alive():
                return False
            worker = self._thread_factory(
                target=self._warm_market,
                args=(market,),
                name="binance-cache-warm-" + market,
                daemon=True,
            )
            self._warmup_threads[market] = worker
            self._status[market]["cache_warmup_worker_starts"] += 1
            worker.start()
            return True

    def _warm_market(self, market, symbols=None):
        """Keep one cache warmup loop aligned with the live registered universe."""
        fixed_symbols = None if symbols is None else set(symbols)
        while not self._stop.is_set():
            if fixed_symbols is None:
                with self._lock:
                    current_symbols = sorted(self._market_symbols.get(market, set()))
            else:
                current_symbols = sorted(fixed_symbols)
            self._prune_market_state(market, current_symbols)
            with self._lock:
                self._status[market]["cache_warmup_total"] = len(current_symbols)
            if not current_symbols:
                if self._stop.wait(5):
                    return
                continue

            # The canonical 5m history is required for analysis. Never let
            # optional Spot 4h macro refreshes delay the primary 5m bootstrap.
            ready = {
                symbol for symbol in current_symbols
                if self._snapshot(market, symbol) is not None
            }
            # A cache can become healthy after the initial REST warmup because
            # live closed candles fill a short history gap. Detect that
            # transition and replay any close event retained by Autopilot.
            with self._lock:
                previously_ready = {
                    key[1] for key in self._cache_ready if key[0] == market
                }
                self._cache_ready = {
                    key for key in self._cache_ready
                    if key[0] != market or key[1] in ready
                }
                self._cache_ready.update((market, symbol) for symbol in ready)
                newly_ready = ready - previously_ready
                state = self._status[market]
                state["cache_ready_symbols"] = len(ready)
                state["cache_warmup_completed"] = len(ready)
            if callable(self.on_cache_ready):
                for symbol in newly_ready:
                    frames = self._snapshot(market, symbol)
                    if frames is not None:
                        try:
                            self.on_cache_ready(market, symbol, frames)
                        except Exception as error:
                            with self._lock:
                                self._status[market]["last_error"] = (
                                    "cache-ready replay callback: " + str(error)
                                )
            pending = [symbol for symbol in current_symbols if symbol not in ready]
            reason_counts = {}
            reason_samples = []
            for symbol in pending:
                for interval in self.REQUIRED_ROWS[market]:
                    frame = self._cache.get((market, symbol, interval))
                    reason = self._frame_health_reason(market, interval, frame)
                    if reason:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                        if len(reason_samples) < 12:
                            reason_samples.append({
                                "symbol": symbol, "interval": interval,
                                "reason": reason,
                                "rows": 0 if frame is None else int(len(frame)),
                                "required_rows": int(self.REQUIRED_ROWS[market].get(
                                    interval, self.CACHE_LIMITS[market][interval])),
                            })
                        break
            with self._lock:
                self._status[market]["cache_not_ready_reasons"] = reason_counts
                self._status[market]["cache_not_ready_samples"] = reason_samples
            if pending:
                with ThreadPoolExecutor(
                    max_workers=self.warmup_workers,
                    thread_name_prefix="candle-warm-" + market,
                ) as pool:
                    futures = [
                        pool.submit(self._warm_symbol, market, symbol)
                        for symbol in pending
                    ]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as error:
                            with self._lock:
                                self._status[market]["last_error"] = (
                                    "cache warmup: " + str(error)
                                )
                        if self._stop.is_set():
                            return
                # Recompute health after warmup so published cache counters
                # describe the current cache, not the pre-warmup snapshot.
                ready_after = {
                    symbol for symbol in current_symbols
                    if self._snapshot(market, symbol) is not None
                }
                reasons_after = {}
                samples_after = []
                for symbol in current_symbols:
                    if symbol in ready_after:
                        continue
                    for interval in self.REQUIRED_ROWS[market]:
                        with self._cache_lock:
                            frame = self._cache.get((market, symbol, interval))
                        reason = self._frame_health_reason(market, interval, frame)
                        if reason:
                            reasons_after[reason] = reasons_after.get(reason, 0) + 1
                            if len(samples_after) < 12:
                                samples_after.append({
                                    "symbol": symbol, "interval": interval,
                                    "reason": reason,
                                    "rows": 0 if frame is None else int(len(frame)),
                                    "required_rows": int(self.REQUIRED_ROWS[market].get(
                                        interval, self.CACHE_LIMITS[market][interval])),
                                })
                            break
                with self._lock:
                    self._cache_ready = {
                        key for key in self._cache_ready
                        if key[0] != market or key[1] not in current_symbols
                    } | {(market, symbol) for symbol in ready_after}
                    self._status[market]["cache_ready_symbols"] = len(ready_after)
                    self._status[market]["cache_warmup_completed"] = len(ready_after)
                    self._status[market]["cache_not_ready_reasons"] = reasons_after
                    self._status[market]["cache_not_ready_samples"] = samples_after
            # Refresh optional Spot 4h context only after the required 5m
            # cache has had its warmup pass. A slow 4h REST request must never
            # block the initial analysis cache from becoming ready.
            if market == "spot" and not self._stop.is_set():
                for symbol in current_symbols:
                    if self._stop.is_set():
                        return
                    self._refresh_spot_4h_if_due(symbol)

            # Re-read the registered universe after each pass. This worker is
            # intentionally retained across WebSocket reconnects.
            if self._stop.wait(10):
                return

    def _recompute_connection_status(self, market):
        live = sum(1 for key, value in self._connection_live.items()
                   if key[0] == market and value)
        total = int(self._status[market]["total_chunks"])
        self._status[market]["connected_chunks"] = live
        self._status[market]["connected"] = live > 0
        self._status[market]["fully_connected"] = total > 0 and live == total
        if live == total and total:
            self._status[market]["state"] = "live"
        elif live:
            self._status[market]["state"] = "partial"
        elif not self._stop.is_set():
            self._status[market]["state"] = "connecting"

    def _run_market(self, market):
        while not self._stop.is_set():
            try:
                symbols = self._load_symbols(market)
                # Provision one logical auditor record for every eligible pair
                # before websocket events start arriving; status can distinguish
                # registered coverage from symbols that actually emitted candles.
                self._register_universe(market, symbols)
                intervals = self._active_intervals(market)
                stream_names = [
                    symbol.lower() + "@kline_" + interval
                    for symbol in symbols for interval in intervals
                ]
                chunks = [
                    stream_names[i:i + self.MAX_STREAMS_PER_CONNECTION]
                    for i in range(0, len(stream_names), self.MAX_STREAMS_PER_CONNECTION)
                ]
                symbol_connections = {}
                for chunk_index, chunk in enumerate(chunks):
                    for stream_name in chunk:
                        if stream_name.endswith("@kline_5m"):
                            symbol_connections[(
                                market, stream_name.split("@", 1)[0].upper()
                            )] = chunk_index
                with self._lock:
                    self._market_symbols[market] = set(symbols)
                    self._symbol_connection = {
                        key: value for key, value in self._symbol_connection.items()
                        if key[0] != market
                    }
                    self._symbol_connection.update(symbol_connections)
                    self._status[market].update({
                        "symbols": len(symbols), "universe_registered": len(symbols),
                        "total_chunks": len(chunks),
                        "connected_chunks": 0, "connected": False,
                        "fully_connected": False,
                        "state": "warming_and_connecting", "last_error": None,
                    })
                    for key in list(self._connection_live):
                        if key[0] == market:
                            self._connection_live.pop(key, None)
                self._prune_market_state(market, symbols)
                self._ensure_warmup_worker(market)
                with self._lock:
                    universe_day = self._status[market].get("ranking_day") or (
                        datetime.now(timezone.utc).date().isoformat()
                    )
                threads = []
                for index, streams in enumerate(chunks):
                    thread = threading.Thread(
                        target=self._run_connection,
                        args=(market, index, streams, universe_day),
                        name="binance-ws-%s-%s" % (market, index),
                        daemon=True,
                    )
                    threads.append(thread)
                    thread.start()
                self._connection_threads.extend(threads)
                while not self._stop.wait(1):
                    if datetime.now(timezone.utc).date().isoformat() != universe_day:
                        # Rebuild subscriptions from the new daily ranking at
                        # the UTC day boundary; never keep yesterday's list.
                        with self._lock:
                            apps = [
                                app for (app_market, _), app in self._apps.items()
                                if app_market == market
                            ]
                        for app in apps:
                            try:
                                app.close()
                            except Exception:
                                pass
                        # Do not subscribe to today's ranking until every
                        # connection using yesterday's symbol list has exited.
                        for thread in threads:
                            thread.join(timeout=10)
                        while (
                            any(thread.is_alive() for thread in threads)
                            and not self._stop.wait(1)
                        ):
                            pass
                        break
                    if not any(thread.is_alive() for thread in threads):
                        break
                if self._stop.is_set():
                    return
            except Exception as error:
                with self._lock:
                    self._status[market]["state"] = "error"
                    self._status[market]["connected"] = False
                    self._status[market]["last_error"] = str(error)
            if not self._stop.wait(self.reconnect_seconds):
                continue

    def _run_connection(self, market, index, streams, universe_day=None):
        conn_key = (market, index)
        ack_id = index + 1
        retry_delay = float(self.reconnect_seconds)
        while not self._stop.is_set():
            attempt_started = time.monotonic()
            if (
                universe_day
                and datetime.now(timezone.utc).date().isoformat() != universe_day
            ):
                return
            try:
                def on_open(app):
                    app.send(json.dumps({
                        "method": "SUBSCRIBE", "params": streams, "id": ack_id,
                    }))
                    with self._lock:
                        self._connection_live[conn_key] = False
                        self._status[market]["state"] = "subscribing"
                        self._recompute_connection_status(market)

                def on_message(app, message):
                    self._handle_message(market, index, ack_id, app, message)

                def on_error(_app, error):
                    with self._lock:
                        self._connection_live[conn_key] = False
                        self._status[market]["last_error"] = str(error)
                        self._status[market]["state"] = "error"
                        self._recompute_connection_status(market)

                def on_close(_app, code, reason):
                    with self._lock:
                        self._connection_live[conn_key] = False
                        self._status[market]["last_close_code"] = code
                        self._status[market]["last_close_reason"] = str(reason or "")[:500]
                        self._status[market]["last_disconnect_at"] = (
                            datetime.now(timezone.utc).isoformat()
                        )
                        self._recompute_connection_status(market)

                app = WebSocketApp(
                    self.BASE_URLS[market], on_open=on_open,
                    on_message=on_message, on_error=on_error,
                    on_close=on_close,
                )
                with self._lock:
                    self._apps[conn_key] = app
                app.run_forever(
                    ping_interval=self.CLIENT_PING_INTERVAL_SECONDS
                )
            except Exception as error:
                with self._lock:
                    self._status[market]["last_error"] = str(error)
                    self._status[market]["state"] = "error"
            finally:
                with self._lock:
                    self._connection_live[conn_key] = False
                    self._recompute_connection_status(market)
                    self._status[market]["reconnects"] += 1
                    self._apps.pop(conn_key, None)
                    attempt_duration = time.monotonic() - attempt_started
                    if attempt_duration >= 120:
                        retry_delay = float(self.reconnect_seconds)
                    else:
                        retry_delay = min(60.0, max(
                            float(self.reconnect_seconds), retry_delay * 2.0
                        ))
                    self._status[market]["reconnect_delay_seconds"] = retry_delay
            if (
                universe_day
                and datetime.now(timezone.utc).date().isoformat() != universe_day
            ):
                return
            if not self._stop.wait(retry_delay):
                continue

    def _handle_message(self, market, index, ack_id, app, message):
        try:
            packet = json.loads(message)
            if packet.get("id") == ack_id:
                if packet.get("result", "__missing__") is None and "code" not in packet:
                    with self._lock:
                        self._connection_live[(market, index)] = True
                        self._status[market]["last_error"] = None
                        self._recompute_connection_status(market)
                else:
                    with self._lock:
                        self._connection_live[(market, index)] = False
                        self._status[market]["last_error"] = str(
                            packet.get("msg") or packet.get("error") or packet)
                        self._status[market]["state"] = "subscription_error"
                        self._recompute_connection_status(market)
                    app.close()
                return

            data = packet.get("data", packet)
            if data.get("e") != "kline":
                return
            kline = data.get("k") or {}
            interval = str(kline.get("i", ""))
            if interval not in self.CACHE_LIMITS[market]:
                return
            symbol = str(data.get("s") or kline.get("s") or "").upper()
            open_time = int(kline.get("t", 0))
            if not symbol or open_time <= 0:
                return
            event_time = int(data.get("E", 0))
            is_closed = bool(kline.get("x"))
            if is_closed:
                self._merge_event_candle(market, symbol, interval, kline)

            if interval != "5m":
                with self._lock:
                    self._status[market]["last_event"] = datetime.now(
                        timezone.utc).isoformat()
                return

            key = (market, symbol)
            with self._lock:
                previous = self._last_open.get(key)
                first_for_candle = previous != open_time
                if first_for_candle:
                    self._last_open[key] = open_time
                opening_event = (
                    first_for_candle and not is_closed
                    and event_time >= open_time
                    and event_time - open_time <= 10000
                )
                self._latest_candles[key] = {
                    "symbol": symbol, "interval": "5m",
                    "time": open_time // 1000, "open_time": open_time,
                    "open": float(kline.get("o", 0)),
                    "high": float(kline.get("h", 0)),
                    "low": float(kline.get("l", 0)),
                    "close": float(kline.get("c", 0)),
                    "volume": float(kline.get("v", 0)),
                    "is_closed": is_closed, "event_time": event_time,
                    "received_at_ms": int(time.time() * 1000),
                    "source": "binance_websocket",
                }
                self._status[market]["last_event"] = datetime.now(
                    timezone.utc).isoformat()
                if is_closed:
                    self._status[market]["last_closed_candle"] = {
                        "symbol": symbol, "open_time": open_time,
                        "close": kline.get("c"),
                    }

            identity = (market, symbol, open_time)
            if is_closed:
                with self._lock:
                    if identity in self._closed_seen:
                        return
                    self._closed_seen.add(identity)
                    if len(self._closed_seen) > 20000:
                        self._closed_seen = set(list(self._closed_seen)[-10000:])
                frames = self._snapshot(market, symbol)
                self.on_kline(market, {
                    "symbol": symbol, "interval": "5m",
                    "open_time": open_time, "close_time": int(kline.get("T", 0)),
                    "open": float(kline.get("o", 0)),
                    "high": float(kline.get("h", 0)),
                    "low": float(kline.get("l", 0)),
                    "close": float(kline.get("c", 0)),
                    "volume": float(kline.get("v", 0)),
                    "event_time": event_time, "is_closed": True,
                    "first_update": False, "source": "binance_websocket",
                    "market_data": frames,
                    "cache_ready": frames is not None,
                })
            elif opening_event:
                self.on_kline(market, {
                    "symbol": symbol, "interval": "5m",
                    "open_time": open_time, "close_time": int(kline.get("T", 0)),
                    "open": float(kline.get("o", 0)),
                    "high": float(kline.get("h", 0)),
                    "low": float(kline.get("l", 0)),
                    "close": float(kline.get("c", 0)),
                    "volume": float(kline.get("v", 0)),
                    "event_time": event_time, "is_closed": False,
                    "first_update": True, "source": "binance_websocket",
                })
        except Exception as error:
            with self._lock:
                self._status[market]["last_error"] = "message: " + str(error)