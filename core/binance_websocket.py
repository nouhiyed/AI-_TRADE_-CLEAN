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
        "spot": "wss://stream.binance.com:9443/ws",
        "futures": "wss://fstream.binance.com/market/stream",
    }
    CACHE_LIMITS = {
        "futures": {"5m": 100, "15m": 60, "1h": 60},
        "spot": {"5m": 100, "15m": 60, "1h": 210, "4h": 210},
    }
    REQUIRED_ROWS = {
        "futures": {"5m": 60, "15m": 60, "1h": 60},
        # Spot's 4h EMA200 is an optional macro filter for newer listings.
        "spot": {"5m": 60, "15m": 60, "1h": 210},
    }
    MAX_STREAMS_PER_CONNECTION = 1020
    INTERVAL_MS = {"5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000}

    def __init__(self, market_feed, on_kline, max_symbols=None,
                 refresh_seconds=1800, reconnect_seconds=3):
        self.feed = market_feed
        self.on_kline = on_kline
        # Initial mobile rollout deliberately watches only the most liquid 5-8 symbols.
        self.max_symbols = int(max_symbols or os.environ.get("AI_TRADER_WS_MAX_SYMBOLS", "8"))
        self.max_symbols = min(8, max(5, self.max_symbols))
        self.refresh_seconds = max(300, int(refresh_seconds))
        self.reconnect_seconds = max(1, int(reconnect_seconds))
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._cache_lock = threading.RLock()
        self._apps = {}
        self._threads = {}
        self._connection_threads = []
        self._connection_live = {}
        self._symbol_connection = {}
        self._cache = {}
        self._cache_ready = set()
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
            for market in ("spot", "futures"):
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
        symbols = [symbol for _, symbol in ranked[:self.max_symbols]]
        if not symbols:
            raise RuntimeError("Binance returned no eligible liquid symbols for " + market)
        return symbols

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

    def _frame_healthy(self, market, interval, frame, now_ms=None):
        minimum = self.REQUIRED_ROWS[market].get(
            interval, self.CACHE_LIMITS[market][interval])
        if frame is None or len(frame) < minimum:
            return False
        if "is_closed" not in frame.columns or not bool(frame["is_closed"].iloc[-1]):
            return False
        if "open_time" not in frame.columns or "close_time" not in frame.columns:
            return False
        try:
            interval_ms = self.INTERVAL_MS[interval]
            opens = pd.to_datetime(frame["open_time"], utc=True)
            closes = pd.to_datetime(frame["close_time"], utc=True)
            expected_delta = pd.Timedelta(milliseconds=interval_ms)
            if len(opens) > 1 and not bool((opens.diff().dropna() == expected_delta).all()):
                return False
            if not bool(((closes - opens) == expected_delta - pd.Timedelta(milliseconds=1)).all()):
                return False
            last_close = int(pd.Timestamp(closes.iloc[-1]).timestamp() * 1000)
            now_ms = int(now_ms or time.time() * 1000)
            if last_close > now_ms + 30000:
                return False
            if now_ms - last_close > (2 * interval_ms + 30000):
                return False
            return True
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
            return False

    def _snapshot(self, market, symbol):
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
        return frames

    def _warm_symbol(self, market, symbol):
        for interval, limit in self.CACHE_LIMITS[market].items():
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
                    frame = self.feed.fetch_candles(
                        market, symbol, interval, limit=limit, closed_only=True)
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
            else:
                with self._lock:
                    self._warm_errors.pop((market, symbol, interval), None)
        ready = self._snapshot(market, symbol) is not None
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
        return ready

    def _warm_market(self, market, symbols):
        with self._lock:
            self._status[market]["cache_warmup_total"] = len(symbols)
            self._status[market]["cache_warmup_completed"] = 0
            self._status[market]["cache_ready_symbols"] = 0
            self._status[market]["cache_warmup_errors"] = 0
        while not self._stop.is_set():
            ready = {
                symbol for symbol in symbols
                if self._snapshot(market, symbol) is not None
            }
            with self._lock:
                self._cache_ready = {
                    key for key in self._cache_ready
                    if key[0] != market or key[1] in ready
                }
                self._cache_ready.update((market, symbol) for symbol in ready)
                state = self._status[market]
                state["cache_ready_symbols"] = len(ready)
                state["cache_warmup_completed"] = len(ready)
            pending = [symbol for symbol in symbols if symbol not in ready]
            if pending:
                with ThreadPoolExecutor(max_workers=6, thread_name_prefix="candle-warm-" + market) as pool:
                    futures = [pool.submit(self._warm_symbol, market, symbol) for symbol in pending]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as error:
                            with self._lock:
                                self._status[market]["last_error"] = "cache warmup: " + str(error)
                        if self._stop.is_set():
                            return
            # Keep checking cache continuity and age while the stream runs.
            if self._stop.wait(60):
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
                intervals = list(self.CACHE_LIMITS[market])
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
                    self._symbol_connection.update(symbol_connections)
                    self._status[market].update({
                        "symbols": len(symbols), "total_chunks": len(chunks),
                        "connected_chunks": 0, "connected": False,
                        "fully_connected": False,
                        "state": "warming_and_connecting", "last_error": None,
                    })
                    for key in list(self._connection_live):
                        if key[0] == market:
                            self._connection_live.pop(key, None)
                threading.Thread(
                    target=self._warm_market, args=(market, symbols),
                    name="binance-cache-warm-" + market, daemon=True,
                ).start()
                threads = []
                for index, streams in enumerate(chunks):
                    thread = threading.Thread(
                        target=self._run_connection,
                        args=(market, index, streams),
                        name="binance-ws-%s-%s" % (market, index),
                        daemon=True,
                    )
                    threads.append(thread)
                    thread.start()
                self._connection_threads.extend(threads)
                while not self._stop.wait(1):
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

    def _run_connection(self, market, index, streams):
        conn_key = (market, index)
        ack_id = index + 1
        while not self._stop.is_set():
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
                        self._recompute_connection_status(market)

                app = WebSocketApp(
                    self.BASE_URLS[market], on_open=on_open,
                    on_message=on_message, on_error=on_error,
                    on_close=on_close,
                )
                with self._lock:
                    self._apps[conn_key] = app
                app.run_forever(ping_interval=20, ping_timeout=10)
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
            if not self._stop.wait(self.reconnect_seconds):
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