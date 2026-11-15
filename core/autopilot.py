"""Autopilot: scans the market on a timer and opens DEMO trades by itself.

Safety: off by default; every trade still goes through the signal match, trade
plan checks, risk reservation and the learning engine's setup filter.
"""
import json
import math
import os
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from core.learning import setup_key
from agents.futures.agent import FuturesAgent
from agents.spot.agent import SpotAgent

DEFAULTS = {
    "enabled": False,
    "interval_sec": 120,
    "min_confidence": 60.0,
    "max_concurrent": 3,
    "risk_pct": 0.5,
    "cooldown_min": 30,
    "use_context": True,
    "trade_futures": True,
    "trade_spot": True,

}
LIMITS = {
    "interval_sec": (30, 3600),
    "min_confidence": (25.0, 100.0), "max_concurrent": (1, 5),
    "risk_pct": (0.05, 1.0), "cooldown_min": (0, 1440),
}


class Autopilot:

    def __init__(self, coordinator, context=None, config_path="config/autopilot.json",
                 clock=time.time):
        self.c = coordinator
        self.context = context
        self.path = Path(config_path)
        self.clock = clock
        self.config = dict(DEFAULTS)
        self.last_run = None
        self.last_result = {}
        self._cooldown = {}          # symbol -> unix time when it may trade again
        self._lock = threading.Lock()
        self._event_lock = threading.RLock()
        self._event_seen = set()
        self._pending_entries = {}
        self._latest_open = {}
        # All DEMO position openings are serialized through one risk/execution writer.
        self._event_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="demo-risk-execution-writer")
        # Independent parallel analysis pools: Futures work never queues behind
        # Spot work, and neither pool runs on the WebSocket callback thread.
        # Start with one analyzer per market. Increase only after on-device profiling.
        workers = min(2, max(1, int(os.environ.get(
            "AI_TRADER_ANALYSIS_WORKERS_PER_MARKET", "1"))))
        queue_limit = min(8, max(5, int(os.environ.get(
            "AI_TRADER_ANALYSIS_QUEUE_LIMIT", "8"))))
        self._analysis_pools = {
            domain: ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="v3-analysis-" + domain,
            )
            for domain in ("futures", "spot")
        }
        self._analysis_slots = {
            domain: threading.BoundedSemaphore(queue_limit)
            for domain in ("futures", "spot")
        }
        self._analysis_stats_lock = threading.Lock()
        self._analysis_local = threading.local()
        self._analysis_agents_created = {"futures": 0, "spot": 0}
        self._analysis_pending = {}
        self._analysis_scheduled = set()
        self._analysis_stats = {
            domain: {"workers": workers, "queue_limit": queue_limit,
                     "submitted": 0, "completed": 0, "failed": 0,
                     "rejected": 0, "inflight": 0, "queue_depth": 0,
                     "superseded": 0, "dropped_stale": 0,
                     "last_received_at_ms": None, "last_started_at_ms": None,
                     "last_completed_at_ms": None, "last_queue_wait_ms": None,
                     "last_analysis_ms": None, "last_candle_close_time_ms": None}
            for domain in ("futures", "spot")
        }
        self._load()

    # ---- config ------------------------------------------------------------

    def _load(self):
        try:
            saved = json.loads(self.path.read_text())
            if isinstance(saved, dict):
                self.update(saved, persist=False)
        except Exception:
            pass

    def update(self, changes, persist=True):
        for key, value in (changes or {}).items():
            if key not in DEFAULTS:
                continue
            default = DEFAULTS[key]
            if isinstance(default, bool):
                if isinstance(value, bool):
                    self.config[key] = value
                elif isinstance(value, int) and value in (0, 1):
                    self.config[key] = bool(value)
                # Do not treat strings such as "false" as truthy.
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number != number:
                continue
            lo, hi = LIMITS[key]
            number = min(max(number, lo), hi)
            self.config[key] = int(number) if isinstance(default, int) else number
        if persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.config, indent=2))
        return self.status()

    def status(self):
        with self._analysis_stats_lock:
            analysis = {domain: dict(stats)
                        for domain, stats in self._analysis_stats.items()}
        for domain in ("futures", "spot"):
            analysis[domain]["agent_instances"] = self._analysis_agents_created[domain]
        analysis["execution_writer"] = {
            "workers": 1,
            "queue_depth": self._event_executor._work_queue.qsize(),
            "alive": any(thread.is_alive()
                         for thread in getattr(self._event_executor, "_threads", set())),
        }
        return {"config": dict(self.config), "last_run": self.last_run,
                "last_result": self.last_result,
                "analysis_pools": analysis,
                "cooldowns": {k: int(v - self.clock())
                              for k, v in self._cooldown.items() if v > self.clock()}}

    def worker_agent(self, domain):
        """Return this analysis worker's own persistent market-specific agent."""
        domain = str(domain).strip().lower()
        if domain not in ("futures", "spot"):
            raise ValueError("Unknown analysis market")
        current = getattr(self._analysis_local, "agent", None)
        if current is None or getattr(self._analysis_local, "domain", None) != domain:
            current = FuturesAgent() if domain == "futures" else SpotAgent()
            self._analysis_local.agent = current
            self._analysis_local.domain = domain
            with self._analysis_stats_lock:
                self._analysis_agents_created[domain] += 1
        return current

    def _queue_latest_analysis(self, domain, symbol, candle):
        """Coalesce pending closes per market/symbol; only the newest may wait."""
        key = (domain, symbol)
        received_at_ms = int(time.time() * 1000)
        event = dict(candle)
        event["received_at_ms"] = received_at_ms
        with self._event_lock:
            previous = self._analysis_pending.get(key)
            if previous:
                if int(previous.get("close_time") or 0) >= int(event.get("close_time") or 0):
                    with self._analysis_stats_lock:
                        self._analysis_stats[domain]["dropped_stale"] += 1
                    return False
                with self._analysis_stats_lock:
                    self._analysis_stats[domain]["superseded"] += 1
            self._analysis_pending[key] = event
            with self._analysis_stats_lock:
                stats = self._analysis_stats[domain]
                stats["last_received_at_ms"] = received_at_ms
                stats["queue_depth"] = len(self._analysis_pending)
            if key in self._analysis_scheduled:
                return True
            if len(self._analysis_scheduled) >= self._analysis_stats[domain]["queue_limit"]:
                self._analysis_pending.pop(key, None)
                with self._analysis_stats_lock:
                    stats = self._analysis_stats[domain]
                    stats["rejected"] += 1
                    stats["queue_depth"] = len(self._analysis_pending)
                return False
            self._analysis_scheduled.add(key)
        if not self.submit_analysis(domain, self._drain_symbol_analysis, domain, symbol):
            with self._event_lock:
                self._analysis_scheduled.discard(key)
                self._analysis_pending.pop(key, None)
                with self._analysis_stats_lock:
                    stats = self._analysis_stats[domain]
                    stats["queue_depth"] = len(self._analysis_pending)
            return False
        return True

    def _drain_symbol_analysis(self, domain, symbol):
        """One scheduled worker drains the latest pending event for this symbol."""
        key = (domain, symbol)
        while True:
            with self._event_lock:
                candle = self._analysis_pending.pop(key, None)
                if candle is None:
                    self._analysis_scheduled.discard(key)
                    with self._analysis_stats_lock:
                        self._analysis_stats[domain]["queue_depth"] = len(self._analysis_pending)
                    return
                with self._analysis_stats_lock:
                    self._analysis_stats[domain]["queue_depth"] = len(self._analysis_pending)
            received_at_ms = int(candle.get("received_at_ms") or 0)
            if received_at_ms <= 0 or int(time.time() * 1000) - received_at_ms > 30000:
                with self._analysis_stats_lock:
                    self._analysis_stats[domain]["dropped_stale"] += 1
                continue
            self._analyze_ws_close(domain, symbol, candle, received_at_ms)

    def submit_analysis(self, domain, fn, *args):
        """Dispatch analysis to a bounded, market-isolated worker pool."""
        domain = str(domain).strip().lower()
        if domain not in self._analysis_pools:
            return False
        if not self._analysis_slots[domain].acquire(blocking=False):
            with self._analysis_stats_lock:
                self._analysis_stats[domain]["rejected"] += 1
            return False
        with self._analysis_stats_lock:
            self._analysis_stats[domain]["submitted"] += 1
            self._analysis_stats[domain]["inflight"] += 1
        try:
            future = self._analysis_pools[domain].submit(fn, *args)
        except Exception:
            self._analysis_slots[domain].release()
            with self._analysis_stats_lock:
                self._analysis_stats[domain]["inflight"] -= 1
                self._analysis_stats[domain]["failed"] += 1
            return False

        def finished(done):
            failed = done.cancelled() or done.exception() is not None
            with self._analysis_stats_lock:
                self._analysis_stats[domain]["inflight"] -= 1
                self._analysis_stats[domain]["completed"] += 1
                if failed:
                    self._analysis_stats[domain]["failed"] += 1
            self._analysis_slots[domain].release()

        future.add_done_callback(finished)
        return True

    # ---- one scan cycle --------------------------------------------------------

    def _open_symbols(self):
        return {str(p.get("symbol")).upper()
                for p in self.c.demo_positions.values()
                if p.get("status") in ("DEMO_OPEN", "DEMO_PARTIALLY_CLOSED",
                                       "DEMO_UNPROTECTED",
                                       "DEMO_RECONCILIATION_REQUIRED")}

    def tick(self, force=False):
        """Legacy polling is intentionally non-executing; WS close events own entries."""
        with self._lock:
            self.last_run = int(self.clock())
            if self.c.mode != "DEMO":
                result = {
                    "status": "SKIPPED", "phase": "websocket_5m",
                    "skipped": [{"reason": "Only internal DEMO execution is enabled"}],
                    "opened": [], "scanned": 0, "actionable": 0,
                }
            elif not self.config.get("enabled") and not force:
                result = {
                    "status": "IDLE", "phase": "websocket_5m",
                    "skipped": [{"reason": "Autopilot is off"}],
                    "opened": [], "scanned": 0, "actionable": 0,
                }
            else:
                result = {
                    "status": "WAITING_FOR_5M_EVENT",
                    "phase": "websocket_5m",
                    "message": "Entries are evaluated only from confirmed closed 5m WebSocket candles.",
                    "opened": [], "skipped": [], "scanned": 0, "actionable": 0,
                }
            self.last_result = dict(result)
            return result


    def _process(self, domain, scan, cfg, open_symbols, slots, result):
        # Kept for API compatibility only. Polling/scanner opportunities are
        # never allowed to bypass the confirmed 5m WebSocket opening event.
        for opportunity in scan.get("opportunities", []):
            result["skipped"].append({
                "domain": domain,
                "symbol": str(opportunity.get("symbol", "")).upper(),
                "reason": "Waiting for the confirmed 5m close and next-candle WebSocket open",
            })
        return slots


    # ---- WebSocket-driven 5m strategy clock ---------------------------------

    def on_market_kline(self, market, event):
        """Accept only deduplicated 5m close and next-candle-open events."""
        domain = str(market).strip().lower()
        if domain not in ("spot", "futures") or not isinstance(event, dict):
            return
        symbol = str(event.get("symbol", "")).upper()
        open_time = int(event.get("open_time") or 0)
        if not symbol or open_time <= 0 or event.get("interval") != "5m":
            return
        key = (domain, symbol)
        if event.get("first_update") and not event.get("is_closed"):
            pending = None
            with self._event_lock:
                self._latest_open[key] = dict(event)
                candidate = self._pending_entries.get(key)
                if candidate and int(candidate.get("next_open_time", 0)) == open_time:
                    pending = self._pending_entries.pop(key, None)
                elif candidate and open_time > int(candidate.get("next_open_time", 0)):
                    self._pending_entries.pop(key, None)
            if pending:
                self._event_executor.submit(self._execute_ws_entry, domain, symbol,
                                            pending, dict(event))
        if event.get("is_closed"):
            identity = (domain, symbol, open_time)
            with self._event_lock:
                if identity in self._event_seen:
                    return
                self._event_seen.add(identity)
                if len(self._event_seen) > 20000:
                    # Bounded memory: retain only recent candle identities.
                    self._event_seen = set(list(self._event_seen)[-10000:])
            if not self._queue_latest_analysis(domain, symbol, event):
                self.last_result = {
                    "status": "ANALYSIS_QUEUE_FULL", "phase": "websocket_analysis",
                    "domain": domain, "symbol": symbol,
                    "reason": "Bounded latest-event analysis queue is full; stale work was not replayed",
                }


    def queue_manual_signal(self, domain, symbol, signal):
        """Queue an agent-approved chart submission for the next 5m open."""
        domain = str(domain).strip().lower()
        symbol = str(symbol).strip().upper()
        if domain not in ("spot", "futures") or not symbol:
            return {"success": False, "reason": "Invalid market or symbol"}
        if self.c.mode != "DEMO" or not self.c._agent_enabled(domain):
            return {"success": False, "reason": "DEMO mode or agent control blocks this entry"}
        if not isinstance(signal, dict):
            return {"success": False, "reason": "Missing approved agent signal"}
        if str(signal.get("symbol", "")).strip().upper() != symbol:
            return {"success": False, "reason": "Signal symbol does not match requested symbol"}
        plan = signal.get("trade_plan") or {}
        direction = str(signal.get("direction", "WAIT")).upper()
        expected_plan_direction = "LONG" if direction == "BUY" else direction
        if (signal.get("strategy_version") != "v3"
                or plan.get("strategy_version") != "v3"
                or str(plan.get("timeframe", "")).lower() != "5m"
                or str(plan.get("direction", "")).upper() != expected_plan_direction
                or plan.get("success") is not True
                or direction not in (("BUY",) if domain == "spot" else ("LONG", "SHORT"))):
            return {"success": False, "reason": "The agent signal is not an approved V3 setup"}
        now_ms = int(self.clock() * 1000)
        generated_at = int(signal.get("signal_generated_at_ms") or 0)
        if generated_at <= 0 or now_ms - generated_at > 30000 or generated_at > now_ms + 2000:
            return {"success": False, "reason": "Agent signal is stale; request a fresh 5m analysis"}
        next_open = ((now_ms // 300000) + 1) * 300000
        key = (domain, symbol)
        with self._event_lock:
            latest = self._latest_open.get(key)
            if latest and int(latest.get("open_time", 0)) >= next_open:
                next_open = int(latest["open_time"]) + 300000
            pending = {
                "signal": dict(signal), "next_open_time": next_open,
                "signal_open_time": int(signal.get("signal_candle_open_time") or 0),
                "created_at": self.clock(), "manual": True,
            }
            self._pending_entries[key] = pending
            current = self._latest_open.get(key)
            ready = bool(current and int(current.get("open_time", 0)) == next_open)
            if ready:
                self._pending_entries.pop(key, None)
        if ready:
            self._event_executor.submit(self._execute_ws_entry, domain, symbol,
                                        pending, current)
        return {
            "success": True, "queued": True, "opened": False,
            "decision_status": "QUEUED",
            "decision_reason": "Agent approved; waiting for the next 5m candle open",
            "next_entry_candle_open_ms": next_open,
        }

    def _analyze_ws_close(self, domain, symbol, candle, received_at_ms=None):
        """Analyze a fresh confirmed 5m close and record end-to-end timing."""
        analysis_started_at_ms = int(time.time() * 1000)
        received_at_ms = int(received_at_ms or candle.get("received_at_ms") or analysis_started_at_ms)
        try:
            with self._analysis_stats_lock:
                stats = self._analysis_stats[domain]
                stats["last_started_at_ms"] = analysis_started_at_ms
                stats["last_queue_wait_ms"] = max(0, analysis_started_at_ms - received_at_ms)
                stats["last_candle_close_time_ms"] = int(candle.get("close_time") or 0)
            if not self.c._agent_enabled(domain):
                return
            if not candle.get("is_closed"):
                return
            close_time = int(candle.get("close_time") or 0)
            close_event_time = int(candle.get("event_time") or 0)
            local_now_ms = int(time.time() * 1000)
            if (close_time <= 0 or close_event_time < close_time
                    or close_event_time - close_time > 10000
                    or abs(local_now_ms - close_event_time) > 30000):
                return
            frames = candle.get("market_data")
            if not isinstance(frames, dict):
                self.last_result = {
                    "status": "WAITING_FOR_CACHE", "phase": "websocket_analysis",
                    "domain": domain, "symbol": symbol,
                    "reason": "Closed multi-timeframe candle cache is not ready",
                }
                return
            analysis = self.c.analyze_live_from_frames(
                symbol, domain, frames, agent=self.worker_agent(domain)
            )
            signal = (analysis.get("signals") or {}).get(domain)
            if not isinstance(signal, dict):
                self.last_result = {
                    "status": "SKIPPED", "phase": "websocket_analysis",
                    "domain": domain, "symbol": symbol,
                    "reason": str((analysis.get("data_errors") or {}).get(domain)
                                  or "Agent returned no signal from the closed 5m cache"),
                }
                return
            signal = dict(signal)
            signal["symbol"] = symbol
            signal["signal_timeframe"] = "5m"
            signal["signal_candle_open_time"] = int(candle["open_time"])
            signal["signal_candle_close_time"] = int(candle["close_time"])
            analysis_completed_at_ms = int(time.time() * 1000)
            signal["signal_generated_at_ms"] = analysis_completed_at_ms
            signal["signal_received_at_ms"] = received_at_ms
            signal["analysis_started_at_ms"] = analysis_started_at_ms
            signal["analysis_completed_at_ms"] = analysis_completed_at_ms
            signal["analysis_queue_wait_ms"] = max(0, analysis_started_at_ms - received_at_ms)
            signal["analysis_duration_ms"] = max(0, analysis_completed_at_ms - analysis_started_at_ms)
            signal["signal_source"] = "BINANCE_WEBSOCKET_CACHE"
            plan = signal.get("trade_plan")
            direction = str(signal.get("direction", "WAIT")).upper()
            valid_direction = direction in (("BUY",) if domain == "spot"
                                             else ("LONG", "SHORT"))
            if (not valid_direction or signal.get("strategy_version") != "v3"
                    or not isinstance(plan, dict) or plan.get("success") is not True
                    or plan.get("strategy_version") != "v3"
                    or str(plan.get("timeframe", "")).lower() != "5m"):
                return
            with self._event_lock:
                self.c.last_signals[domain] = signal
                cfg = dict(self.config)
                if not cfg.get("enabled") or not cfg.get("trade_" + domain):
                    return
                key = (domain, symbol)
                expected_open = int(candle["close_time"]) + 1
                pending = {
                    "signal": signal,
                    "signal_open_time": int(candle["open_time"]),
                    "next_open_time": expected_open,
                    "created_at": time.time(),
                }
                self._pending_entries[key] = pending
                already_open = self._latest_open.get(key)
            # If the next-candle event arrived during local cached analysis,
            # consume it now rather than losing the entry window.
            if already_open and int(already_open.get("open_time", 0)) == expected_open:
                with self._event_lock:
                    current = self._pending_entries.pop(key, None)
                if current:
                    self._event_executor.submit(
                        self._execute_ws_entry, domain, symbol, current, already_open)
        except Exception as error:
            self.last_result = {
                "status": "ERROR", "phase": "websocket_analysis",
                "domain": domain, "symbol": symbol, "error": str(error),
            }
        finally:
            completed_at_ms = int(time.time() * 1000)
            with self._analysis_stats_lock:
                stats = self._analysis_stats[domain]
                stats["last_completed_at_ms"] = completed_at_ms
                stats["last_analysis_ms"] = max(0, completed_at_ms - analysis_started_at_ms)

    def _execute_ws_entry(self, domain, symbol, pending, opening_candle):
        """Execute DEMO at the observed next 5m candle open, never on 1m."""
        result = {
            "status": "RUNNING", "phase": "next_5m_open",
            "domain": domain, "symbol": symbol, "opened": [], "skipped": [],
        }
        try:
            cfg = dict(self.config)
            manual = bool(pending.get("manual"))
            if not manual and (not cfg.get("enabled") or not cfg.get("trade_" + domain)):
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Autopilot or market is disabled"})
                return
            if self.c.mode != "DEMO":
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Only internal DEMO execution is enabled"})
                return
            if not self.c._agent_enabled(domain):
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Agent is stopped"})
                return
            event_time = int(opening_candle.get("event_time") or 0)
            open_time = int(opening_candle.get("open_time") or 0)
            entry_price = float(opening_candle.get("open") or 0)
            local_now_ms = int(time.time() * 1000)
            if (int(pending.get("next_open_time") or 0) != open_time
                    or opening_candle.get("first_update") is not True
                    or opening_candle.get("source") != "binance_websocket"
                    or entry_price <= 0 or event_time < open_time
                    or event_time - open_time > 10000
                    or abs(local_now_ms - event_time) > 30000):
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Next-candle opening event arrived stale or invalid"})
                return
            signal = dict(pending["signal"])
            signal_close_time = int(signal.get("signal_candle_close_time") or 0)
            if signal_close_time and signal_close_time + 1 != open_time:
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Entry is not the immediate next 5m candle"})
                return
            if str(signal.get("symbol", "")).upper() != symbol:
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Signal symbol mismatch"})
                return
            direction = str(signal.get("direction", "WAIT")).upper()
            normalized = "LONG" if direction == "BUY" else direction
            if normalized not in ("LONG", "SHORT"):
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Signal direction is not actionable"})
                return
            if (signal.get("strategy_version") != "v3"
                    or str((signal.get("trade_plan") or {}).get("timeframe", "")).lower() != "5m"):
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Only V3 5m signals may execute"})
                return
            plan = signal.get("trade_plan") or {}
            if plan.get("success") is not True or plan.get("strategy_version") != "v3":
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "V3 trade plan is not approved"})
                return
            try:
                planned_entry = float(plan.get("entry"))
                max_deviation_pct = float(os.environ.get(
                    "AI_TRADER_MAX_ENTRY_DEVIATION_PCT_FUTURES" if domain == "futures"
                    else "AI_TRADER_MAX_ENTRY_DEVIATION_PCT_SPOT",
                    "0.15" if domain == "futures" else "0.20"))
            except (TypeError, ValueError):
                planned_entry, max_deviation_pct = 0.0, 0.0
            if (not math.isfinite(planned_entry) or planned_entry <= 0
                    or not math.isfinite(max_deviation_pct) or max_deviation_pct <= 0):
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Invalid planned entry or deviation guard"})
                return
            deviation_pct = abs(entry_price - planned_entry) / planned_entry * 100.0
            if deviation_pct > max_deviation_pct:
                result["status"] = "SKIPPED"
                result["skipped"].append({
                    "reason": "Entry price moved %.4f%% from the approved signal price (limit %.4f%%)"
                              % (deviation_pct, max_deviation_pct),
                    "entry_deviation_pct": deviation_pct,
                    "max_entry_deviation_pct": max_deviation_pct,
                })
                return
            confidence = float(signal.get("confidence") or 0)
            if confidence < float(cfg["min_confidence"]):
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Confidence below minimum"})
                return
            if self._cooldown.get(symbol, 0) > self.clock():
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Symbol is in cooldown"})
                return
            open_symbols = self._open_symbols()
            if symbol in open_symbols:
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Symbol already has an open position"})
                return
            if len(open_symbols) >= int(cfg["max_concurrent"]):
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Max concurrent positions reached"})
                return
            strategy_tag = signal.get("strategy_tag") or "v3"
            key = setup_key(normalized, confidence, plan.get("reward_risk"),
                            domain, strategy_tag)
            if not self.c.learning.allows(key):
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Setup disabled by learning: " + key})
                return
            assessment = signal.get("market_context")
            if assessment and assessment.get("veto"):
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": "Context veto: " + str(assessment["veto"])})
                return
            signal["entry_origin"] = "MANUAL_AGENT_WS_5M" if manual else "AUTOPILOT_WS_5M"
            signal["entry_open_time"] = open_time
            signal["entry_reference_price"] = entry_price
            signal["execution_sent_at_ms"] = int(time.time() * 1000)
            signal["entry_deviation_pct"] = deviation_pct
            signal["setup_key"] = key
            signal["confidence_adjusted"] = confidence
            self.c.last_signals[domain] = signal
            if domain == "futures":
                leverage = self.c.select_futures_leverage(signal, plan)
                trade = self.c.request_demo_futures_trade(
                    symbol, normalized, risk_pct=float(cfg["risk_pct"]),
                    leverage=leverage, signal_override=signal,
                    entry_reference_price=entry_price)
            else:
                trade = self.c.request_demo_spot_trade(
                    symbol, domain="spot", risk_pct=float(cfg["risk_pct"]),
                    signal_override=signal, entry_reference_price=entry_price)
            if trade.get("approved"):
                self._cooldown[symbol] = self.clock() + float(cfg["cooldown_min"]) * 60
                result["opened"].append({
                    "domain": domain, "symbol": symbol,
                    "direction": normalized, "trade_id": trade.get("trade_id"),
                    "entry_price": trade.get("entry_price"),
                })
                result["status"] = "COMPLETED"
                result["phase"] = "opened"
            else:
                result["status"] = "SKIPPED"
                result["skipped"].append({"reason": trade.get("reason", "Execution rejected")})
        except Exception as error:
            result["status"] = "ERROR"
            result["error"] = str(error)
        finally:
            self.last_run = int(self.clock())
            self.last_result = result

    def run_forever(self, stop_event=None):
        print("[AUTOPILOT] worker started")
        while not (stop_event and stop_event.is_set()):
            try:
                if self.config["enabled"]:
                    summary = self.tick()
                    if summary.get("opened"):
                        print("[AUTOPILOT] opened:", summary["opened"])
                    pause = max(5, float(self.config["interval_sec"]))
                else:
                    pause = 5
            except Exception as error:
                print("[AUTOPILOT] cycle error:", error)
                pause = 30
            time.sleep(pause)
