"""Low-latency Binance 5m kline streams for Spot and USD-M Futures.

The stream is the clock for strategy evaluation. REST remains the authoritative
backfill/context source; WebSocket events are deduplicated before reaching agents.
"""
import json
import os
import threading
import time
from datetime import datetime, timezone
from urllib.parse import quote

from websocket import WebSocketApp


class BinanceKlineStream:
    BASE_URLS = {
        "spot": "wss://stream.binance.com:9443/stream?streams=",
        "futures": "wss://fstream.binance.com/stream?streams=",
    }

    def __init__(self, market_feed, on_kline, max_symbols=None,
                 refresh_seconds=1800, reconnect_seconds=3):
        self.feed = market_feed
        self.on_kline = on_kline
        self.max_symbols = int(max_symbols or os.environ.get("AI_TRADER_WS_MAX_SYMBOLS", "100"))
        self.max_symbols = min(200, max(10, self.max_symbols))
        self.refresh_seconds = max(300, int(refresh_seconds))
        self.reconnect_seconds = max(1, int(reconnect_seconds))
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._apps = {}
        self._threads = {}
        self._last_open = {}
        self._latest_candles = {}
        self._status = {
            market: {
                "connected": False, "symbols": 0, "last_event": None,
                "last_closed_candle": None, "reconnects": 0,
                "last_error": None, "state": "stopped",
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
            return snapshot

    def latest_candle(self, market, symbol):
        key = (str(market).strip().lower(), str(symbol).strip().upper())
        with self._lock:
            value = self._latest_candles.get(key)
            return dict(value) if value else None

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
        threading.Thread(
            target=self._refresh_loop, name="binance-ws-refresh", daemon=True
        ).start()
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
                value["state"] = "stopping"

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
            rows = self.feed.fetch_symbols("futures")
            eligible = set(rows)
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

    def _run_market(self, market):
        while not self._stop.is_set():
            try:
                symbols = self._load_symbols(market)
                streams = "/".join(
                    quote(symbol.lower(), safe="") + "@kline_5m" for symbol in symbols
                )
                url = self.BASE_URLS[market] + streams
                with self._lock:
                    self._status[market]["state"] = "connecting"
                    self._status[market]["symbols"] = len(symbols)
                    self._status[market]["last_error"] = None

                def on_open(_app):
                    with self._lock:
                        self._status[market]["connected"] = True
                        self._status[market]["state"] = "live"
                        self._status[market]["last_error"] = None
                    print("[WS] %s connected: %s symbols, 5m klines" %
                          (market.upper(), len(symbols)))

                def on_message(_app, message):
                    try:
                        packet = json.loads(message)
                        data = packet.get("data", packet)
                        if data.get("e") != "kline":
                            return
                        kline = data.get("k") or {}
                        if kline.get("i") != "5m":
                            return
                        symbol = str(data.get("s") or kline.get("s") or "").upper()
                        open_time = int(kline.get("t", 0))
                        if not symbol or open_time <= 0:
                            return
                        key = (market, symbol)
                        with self._lock:
                            previous = self._last_open.get(key)
                            first_update = previous != open_time
                            if first_update:
                                self._last_open[key] = open_time
                            self._latest_candles[key] = {
                                "symbol": symbol, "interval": "5m",
                                "time": open_time // 1000,
                                "open_time": open_time,
                                "open": float(kline.get("o", 0)),
                                "high": float(kline.get("h", 0)),
                                "low": float(kline.get("l", 0)),
                                "close": float(kline.get("c", 0)),
                                "volume": float(kline.get("v", 0)),
                                "is_closed": bool(kline.get("x")),
                                "event_time": int(data.get("E", 0)),
                                "source": "binance_websocket",
                            }
                            self._status[market]["last_event"] = datetime.now(
                                timezone.utc).isoformat()
                            if bool(kline.get("x")):
                                self._status[market]["last_closed_candle"] = {
                                    "symbol": symbol, "open_time": open_time,
                                    "close": kline.get("c"),
                                }
                        # First update of the next candle is the entry clock.
                        # Closed events are the only events that trigger analysis.
                        if first_update or bool(kline.get("x")):
                            self.on_kline(market, {
                                "symbol": symbol,
                                "interval": "5m",
                                "open_time": open_time,
                                "close_time": int(kline.get("T", 0)),
                                "open": float(kline.get("o", 0)),
                                "high": float(kline.get("h", 0)),
                                "low": float(kline.get("l", 0)),
                                "close": float(kline.get("c", 0)),
                                "volume": float(kline.get("v", 0)),
                                "event_time": int(data.get("E", 0)),
                                "is_closed": bool(kline.get("x")),
                                "first_update": first_update,
                                "source": "binance_websocket",
                            })
                    except Exception as error:
                        with self._lock:
                            self._status[market]["last_error"] = (
                                "message: " + str(error)
                            )

                def on_error(_app, error):
                    with self._lock:
                        self._status[market]["last_error"] = str(error)
                        self._status[market]["state"] = "error"

                def on_close(_app, code, reason):
                    with self._lock:
                        self._status[market]["connected"] = False
                        if not self._stop.is_set():
                            self._status[market]["state"] = "reconnecting"

                app = WebSocketApp(
                    url, on_open=on_open, on_message=on_message,
                    on_error=on_error, on_close=on_close,
                )
                with self._lock:
                    self._apps[market] = app
                app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as error:
                with self._lock:
                    self._status[market]["last_error"] = str(error)
                    self._status[market]["state"] = "error"
            finally:
                with self._lock:
                    self._status[market]["connected"] = False
                    self._status[market]["reconnects"] += 1
                    self._apps.pop(market, None)
            if not self._stop.wait(self.reconnect_seconds):
                print("[WS] %s reconnecting" % market.upper())
