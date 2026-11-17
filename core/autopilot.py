"""Autopilot: scans the market on a timer and opens DEMO trades by itself.

Safety: off by default; every trade still goes through the signal match, trade
plan checks, risk reservation and the learning engine's setup filter.
"""
import json
import math
import os
import threading
import time
from collections import OrderedDict
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
        self._event_seen = OrderedDict()
        self._pending_entries = {}
        # One retained closed candle per symbol while its cache warms.
        self._waiting_cache_events = {}
        self._latest_open = {}
        self._last_execution_result = {}
        # One logical audit record per symbol; work itself is dispatched to a
        # bounded pool so hundreds of symbols never create hundreds of threads.
        self._symbol_audits = {"futures": {}, "spot": {}}
        self._symbol_universes = {"futures": set(), "spot": set()}
        self._universe_initialized = {"futures": False, "spot": False}
        self._symbol_audit_totals = {
            domain: {
                "events_received": 0, "cache_waiting": 0,
                "cache_replayed": 0, "audit_only": 0,
                "prefilter_passed": 0, "prefilter_rejected": 0, "deep_analysis": 0,
                "actionable": 0, "opened": 0, "errors": 0,
                "stale": 0, "queue_rejected": 0,
                "entry_queued": 0, "entry_submitted": 0,
                "entry_skipped": 0, "entry_expired": 0,
            }
            for domain in ("futures", "spot")
        }
        # All DEMO position openings are serialized through one risk/execution writer.
        self._event_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="demo-risk-execution-writer")
        # Each market owns a separate group of persistent V3 worker agents.
        # A symbol is an independent queued job; workers are shared safely instead
        # of spawning one operating-system thread per coin.
        workers = min(8, max(1, int(os.environ.get(
            "AI_TRADER_ANALYSIS_WORKERS_PER_MARKET", "4"))))
        # Reserve at least one scheduled slot for every symbol in a
        # Binance full-universe market (up to the current 1020-stream chunk).
        queue_limit = min(2048, max(1024, int(os.environ.get(
            "AI_TRADER_ANALYSIS_QUEUE_LIMIT", "1024"))))
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
                     "prefilter_passed": 0, "prefilter_rejected": 0,
                     "deep_analysis": 0, "actionable": 0,
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
        writer_threads = getattr(self._event_executor, "_threads", set())
        analysis["execution_writer"] = {
            "workers": 1,
            "queue_depth": self._event_executor._work_queue.qsize(),
            "alive": not getattr(self._event_executor, "_shutdown", False)
                     and (not writer_threads or any(thread.is_alive()
                                                    for thread in writer_threads)),
        }
        with self._event_lock:
            pending_entries = [
                {"domain": domain, "symbol": symbol,
                 "next_open_time_ms": int(item.get("next_open_time") or 0),
                 "age_seconds": max(0, round(self.clock() - float(item.get("created_at") or self.clock()), 1)),
                 "manual": bool(item.get("manual")),
                 "direction": (item.get("signal") or {}).get("direction")}
                for (domain, symbol), item in self._pending_entries.items()
            ]
            recent_entry_audits = []
            for domain in ("futures", "spot"):
                for record in self._symbol_audits[domain].values():
                    if record.get("state") in ("entry_queued", "entry_submitted",
                                               "entry_skipped", "entry_expired",
                                               "execution_error", "opened"):
                        recent_entry_audits.append({"domain": domain, **dict(record)})
            recent_entry_audits.sort(
                key=lambda item: int(item.get("updated_at_ms") or 0), reverse=True)
        return {"config": dict(self.config), "last_run": self.last_run,
                "last_result": self.last_result,
                "last_execution_result": dict(self._last_execution_result),
                "pending_entries": pending_entries,
                "recent_entry_audits": recent_entry_audits[:30],
                "analysis_pools": analysis,
                "symbol_auditors": self.symbol_auditor_status(),
                "cooldowns": {k: int(v - self.clock())
                              for k, v in self._cooldown.items() if v > self.clock()}}

    def register_symbol_universe(self, domain, symbols):
        """Register every eligible market pair before its first 5m candle event."""
        domain = str(domain).strip().lower()
        if domain not in self._symbol_audits:
            return {"success": False, "reason": "Unknown market domain"}
        normalized = {
            str(symbol).strip().upper()
            for symbol in (symbols or [])
            if str(symbol).strip()
        }
        now_ms = int(self.clock() * 1000)
        with self._event_lock:
            previous = self._symbol_universes[domain]
            self._symbol_universes[domain] = normalized
            self._universe_initialized[domain] = True
            records = self._symbol_audits[domain]
            for symbol in normalized:
                record = records.setdefault(symbol, {
                    "symbol": symbol, "state": "waiting_for_5m_close",
                    "events": 0, "prefilter_events": 0,
                })
                record["registered"] = True
                record.setdefault("registered_at_ms", now_ms)
                if int(record.get("events") or 0) == 0:
                    record["state"] = "waiting_for_5m_close"
                record["universe_updated_at_ms"] = now_ms
            for symbol in previous - normalized:
                if symbol in records:
                    records[symbol]["registered"] = False
                    records[symbol]["state"] = "not_in_current_universe"
                    records[symbol]["universe_updated_at_ms"] = now_ms
        return {
            "success": True, "domain": domain,
            "registered": len(normalized),
            "added": len(normalized - previous),
            "removed": len(previous - normalized),
        }

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

    def _record_symbol_audit(self, domain, symbol, state, increment=None,
                             also_increment=None, **details):
        domain = str(domain).strip().lower()
        symbol = str(symbol).strip().upper()
        if domain not in self._symbol_audits or not symbol:
            return
        with self._event_lock:
            # Standalone/test callers without a universe callback still get
            # an auditor record for every observed symbol. In the production
            # WebSocket path the complete universe is registered first.
            if not self._universe_initialized[domain]:
                self._symbol_universes[domain].add(symbol)
            record = self._symbol_audits[domain].setdefault(symbol, {
                "symbol": symbol, "state": "waiting", "events": 0,
                "prefilter_events": 0,
            })
            record.update(details)
            record["state"] = str(state)
            record["updated_at_ms"] = int(self.clock() * 1000)
            if increment == "events_received":
                record["events"] = int(record.get("events", 0)) + 1
                record["last_event_at_ms"] = record["updated_at_ms"]
                record["last_close_time_ms"] = int(details.get("close_time_ms") or 0)
            if increment in ("prefilter_passed", "prefilter_rejected"):
                record["prefilter_events"] = int(record.get("prefilter_events", 0)) + 1
                record["last_prefilter_at_ms"] = record["updated_at_ms"]
            for counter in (increment, also_increment):
                if counter and counter in self._symbol_audit_totals[domain]:
                    self._symbol_audit_totals[domain][counter] += 1

    def symbol_auditor_status(self):
        """Compact per-market counts; detailed per-symbol state stays internal."""
        result = {}
        with self._event_lock:
            for domain in ("futures", "spot"):
                records = self._symbol_audits[domain]
                active_symbols = self._symbol_universes[domain]
                active_records = [
                    records[symbol] for symbol in active_symbols
                    if symbol in records
                ]
                states = {}
                for record in active_records:
                    state = str(record.get("state", "unknown"))
                    states[state] = states.get(state, 0) + 1
                latest = max(
                    active_records,
                    key=lambda item: int(item.get("updated_at_ms") or 0),
                    default={},
                )
                now_ms = int(self.clock() * 1000)
                fresh_window_ms = 6 * 60 * 1000
                symbols_seen_recently = sum(
                    1 for record in active_records
                    if 0 <= now_ms - int(record.get("last_event_at_ms") or 0)
                    <= fresh_window_ms
                )
                symbols_audited_recently = sum(
                    1 for record in active_records
                    if 0 <= now_ms - int(record.get("last_prefilter_at_ms") or 0)
                    <= fresh_window_ms
                )
                symbols_seen = sum(
                    1 for record in active_records
                    if int(record.get("events") or 0) > 0
                )
                symbols_audited = sum(
                    1 for record in active_records
                    if int(record.get("prefilter_events") or 0) > 0
                )
                result[domain] = {
                    "symbols_tracked": symbols_seen,
                    "symbols_registered": len(active_symbols),
                    "symbols_seen": symbols_seen,
                    "symbols_audited": symbols_audited,
                    "symbols_seen_recently": symbols_seen_recently,
                    "symbols_audited_recently": symbols_audited_recently,
                    "symbols_waiting_for_first_event": max(0, len(active_symbols) - symbols_seen),
                    "symbols_waiting_for_cache": states.get("waiting_cache", 0),
                    "last_symbol": latest.get("symbol"),
                    "last_state": latest.get("state"),
                    "last_updated_at_ms": latest.get("updated_at_ms"),
                    "states": states,
                    **dict(self._symbol_audit_totals[domain]),
                }
        return result

    def _queue_latest_analysis(self, domain, symbol, candle, count_event=True):
        """Coalesce pending closes per market/symbol; only the newest may wait."""
        key = (domain, symbol)
        received_at_ms = int(time.time() * 1000)
        event = dict(candle)
        event["received_at_ms"] = received_at_ms
        close_time_ms = int(event.get("close_time") or 0)
        if count_event:
            with self._event_lock:
                previous_record = self._symbol_audits[domain].get(symbol) or {}
                if int(previous_record.get("last_close_time_ms") or 0) == close_time_ms:
                    count_event = False
        self._record_symbol_audit(
            domain, symbol, "queued",
            increment="events_received" if count_event else None,
            close_time_ms=close_time_ms,
        )
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
            domain_pending = sum(1 for pending_domain, _ in self._analysis_pending
                                 if pending_domain == domain)
            domain_scheduled = sum(1 for scheduled_domain, _ in self._analysis_scheduled
                                   if scheduled_domain == domain)
            with self._analysis_stats_lock:
                stats = self._analysis_stats[domain]
                stats["last_received_at_ms"] = received_at_ms
                stats["queue_depth"] = domain_pending
            if key in self._analysis_scheduled:
                return True
            if domain_scheduled >= self._analysis_stats[domain]["queue_limit"]:
                self._analysis_pending.pop(key, None)
                with self._analysis_stats_lock:
                    stats = self._analysis_stats[domain]
                    stats["rejected"] += 1
                    stats["queue_depth"] = sum(
                        1 for pending_domain, _ in self._analysis_pending
                        if pending_domain == domain
                    )
                return False
            self._analysis_scheduled.add(key)
        if not self.submit_analysis(domain, self._drain_symbol_analysis, domain, symbol):
            with self._event_lock:
                self._analysis_scheduled.discard(key)
                self._analysis_pending.pop(key, None)
                self._record_symbol_audit(
                    domain, symbol, "queue_rejected",
                    increment="queue_rejected",
                    reason="Could not submit analysis worker",
                )
                with self._analysis_stats_lock:
                    stats = self._analysis_stats[domain]
                    stats["queue_depth"] = sum(
                        1 for pending_domain, _ in self._analysis_pending
                        if pending_domain == domain
                    )
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
                        self._analysis_stats[domain]["queue_depth"] = sum(
                            1 for pending_domain, _ in self._analysis_pending
                            if pending_domain == domain
                        )
                    return
                with self._analysis_stats_lock:
                    self._analysis_stats[domain]["queue_depth"] = sum(
                        1 for pending_domain, _ in self._analysis_pending
                        if pending_domain == domain
                    )
            received_at_ms = int(candle.get("received_at_ms") or 0)
            age_ms = int(time.time() * 1000) - received_at_ms if received_at_ms > 0 else 10**12
            if age_ms > 10 * 60 * 1000:
                with self._analysis_stats_lock:
                    self._analysis_stats[domain]["dropped_stale"] += 1
                self._record_symbol_audit(
                    domain, symbol, "stale", increment="stale",
                    reason="Closed candle exceeded the 10-minute audit retention window",
                )
                continue
            if age_ms > 30000:
                candle["audit_only"] = True
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
        """Accept deduplicated 5m closes and next-candle-open events."""
        domain = str(market).strip().lower()
        if domain not in ("spot", "futures") or not isinstance(event, dict):
            return False
        symbol = str(event.get("symbol", "")).upper()
        open_time = int(event.get("open_time") or 0)
        if not symbol or open_time <= 0 or event.get("interval") != "5m":
            return False
        key = (domain, symbol)
        if event.get("first_update") and not event.get("is_closed"):
            pending = None
            expired = None
            with self._event_lock:
                self._latest_open[key] = dict(event)
                candidate = self._pending_entries.get(key)
                if candidate and int(candidate.get("next_open_time", 0)) == open_time:
                    pending = self._pending_entries.pop(key, None)
                elif candidate and open_time > int(candidate.get("next_open_time", 0)):
                    expired = self._pending_entries.pop(key, None)
            if expired:
                expected = int(expired.get("next_open_time") or 0)
                self._record_symbol_audit(
                    domain, symbol, "entry_expired", increment="entry_expired",
                    reason="The exact next 5m opening event was missed; stale entry was discarded safely",
                    expected_open_time_ms=expected, observed_open_time_ms=open_time,
                )
            if pending:
                try:
                    self._event_executor.submit(self._execute_ws_entry, domain, symbol,
                                                pending, dict(event))
                    self._record_symbol_audit(
                        domain, symbol, "entry_submitted", increment="entry_submitted",
                        next_open_time_ms=open_time,
                    )
                except Exception as error:
                    self._record_symbol_audit(
                        domain, symbol, "execution_error", increment="errors",
                        reason="Execution writer submission failed: " + str(error),
                    )
            return True
        if not event.get("is_closed"):
            return False

        identity = (domain, symbol, open_time)
        with self._event_lock:
            if identity in self._event_seen:
                return False
            self._event_seen[identity] = None
            if len(self._event_seen) > 20000:
                self._event_seen.popitem(last=False)

        frames = event.get("market_data")
        if isinstance(frames, dict) and frames:
            with self._event_lock:
                self._waiting_cache_events.pop(key, None)
        if not isinstance(frames, dict) or not frames:
            with self._event_lock:
                previous = self._waiting_cache_events.get(key)
                if previous is None or int(previous.get("close_time") or 0) < int(event.get("close_time") or 0):
                    self._waiting_cache_events[key] = dict(event)
            self._record_symbol_audit(
                domain, symbol, "waiting_cache", increment="events_received", also_increment="cache_waiting",
                close_time_ms=int(event.get("close_time") or 0),
                reason="Closed 5m candle retained until its historical cache is ready",
            )
            return True

        if self._queue_latest_analysis(domain, symbol, event):
            return True

        self.last_result = {
            "status": "ANALYSIS_QUEUE_FULL", "phase": "websocket_analysis",
            "domain": domain, "symbol": symbol,
            "reason": "Analysis queue full; latest event retained for retry",
        }
        with self._event_lock:
            # A rejected submission must be eligible for the next reconciliation
            # pass; otherwise the dedupe marker would turn a temporary full queue
            # into a permanent lost candle.
            self._event_seen.pop(identity, None)
            self._waiting_cache_events[key] = dict(event)
        return False


    def on_symbol_cache_ready(self, domain, symbol, frames):
        """Replay the newest retained close when the 5m history becomes healthy."""
        domain = str(domain).strip().lower()
        symbol = str(symbol).strip().upper()
        if domain not in ("spot", "futures") or not symbol or not isinstance(frames, dict) or not frames:
            return False
        key = (domain, symbol)
        with self._event_lock:
            event = self._waiting_cache_events.pop(key, None)
        if not event:
            return False
        replay = dict(event)
        replay["market_data"] = frames
        replay["cache_ready"] = True
        age_ms = int(time.time() * 1000) - int(replay.get("event_time") or 0)
        if age_ms > 10 * 60 * 1000:
            self._record_symbol_audit(
                domain, symbol, "stale", increment="stale",
                reason="Cache became ready after the 10-minute audit retention window",
            )
            return False
        replay["audit_only"] = bool(replay.get("audit_only")) or age_ms > 30000
        accepted = self._queue_latest_analysis(domain, symbol, replay, count_event=False)
        if accepted:
            self._record_symbol_audit(
                domain, symbol, "cache_replayed", increment="cache_replayed",
                reason="Retained closed candle replayed after cache warmup",
            )
        else:
            with self._event_lock:
                self._event_seen.pop((
                    domain, symbol, int(replay.get("open_time") or 0)
                ), None)
                self._waiting_cache_events[key] = replay
        return accepted


    def has_seen_candle(self, domain, symbol, open_time_ms):
        """Whether this exact market/symbol/5m close has been accepted or retained."""
        identity = (
            str(domain).strip().lower(),
            str(symbol).strip().upper(),
            int(open_time_ms or 0),
        )
        with self._event_lock:
            return identity in self._event_seen


    def completed_symbols_for_cycle(self, domain, open_time_ms, symbols):
        """Return symbols whose analysis worker finished this exact 5m candle."""
        domain = str(domain).strip().lower()
        expected_open = int(open_time_ms or 0)
        completed = []
        errors = []
        with self._event_lock:
            records = self._symbol_audits.get(domain, {})
            for raw_symbol in symbols or []:
                symbol = str(raw_symbol).strip().upper()
                record = records.get(symbol) or {}
                if int(record.get("analysis_completed_open_time_ms") or 0) != expected_open:
                    continue
                completed.append(symbol)
                if record.get("analysis_completed_error"):
                    errors.append(symbol)
        return {"completed": completed, "errors": errors}


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
        """Run the 5m symbol gate, then route candidates to the full V3 worker."""
        analysis_started_at_ms = int(time.time() * 1000)
        received_at_ms = int(received_at_ms or candle.get("received_at_ms") or analysis_started_at_ms)
        prefilter_info = None
        try:
            with self._analysis_stats_lock:
                stats = self._analysis_stats[domain]
                stats["last_started_at_ms"] = analysis_started_at_ms
                stats["last_queue_wait_ms"] = max(0, analysis_started_at_ms - received_at_ms)
                stats["last_candle_close_time_ms"] = int(candle.get("close_time") or 0)
            self._record_symbol_audit(
                domain, symbol, "analyzing",
                queue_wait_ms=max(0, analysis_started_at_ms - received_at_ms),
                open_time_ms=int(candle.get("open_time") or 0),
                close_time_ms=int(candle.get("close_time") or 0),
            )
            if not self.c._agent_enabled(domain):
                self._record_symbol_audit(
                    domain, symbol, "agent_stopped",
                    reason="The market agent is stopped",
                )
                return
            if not candle.get("is_closed"):
                return
            close_time = int(candle.get("close_time") or 0)
            close_event_time = int(candle.get("event_time") or 0)
            local_now_ms = int(time.time() * 1000)
            audit_only = bool(candle.get("audit_only"))

            # A reconnect can deliver a valid closed kline long after its
            # Binance event timestamp. It is still useful for cycle coverage,
            # but must never be allowed to create a late entry. Keep recent
            # delayed events as audit-only instead of incorrectly marking the
            # symbol as an analysis error. Truly expired (>10m), malformed, or
            # future-dated events remain stale and fail the coverage check.
            event_age_ms = local_now_ms - close_event_time
            if (close_time <= 0 or close_event_time < close_time
                    or close_event_time - close_time > 10000
                    or event_age_ms < -30000):
                self._record_symbol_audit(
                    domain, symbol, "stale",
                    increment="stale",
                    reason="Closed-candle event has invalid close/event timestamps",
                )
                return
            if event_age_ms > 10 * 60 * 1000:
                self._record_symbol_audit(
                    domain, symbol, "stale",
                    increment="stale",
                    reason="Closed-candle event exceeded the 10-minute audit retention window",
                )
                return
            if event_age_ms > 30000:
                audit_only = True

            frames = candle.get("market_data")
            if not isinstance(frames, dict) or not frames:
                self.last_result = {
                    "status": "WAITING_FOR_CACHE", "phase": "websocket_analysis",
                    "domain": domain, "symbol": symbol,
                    "reason": "Closed 5m candle cache is not ready",
                }
                self._record_symbol_audit(
                    domain, symbol, "waiting_cache",
                    reason="5m candle history is still warming",
                )
                return

            agent = self.worker_agent(domain)
            if set(frames) == {"5m"}:
                # Full-universe stream: cheap V3-local 5m gate first. Only
                # candidates pay for REST retrieval of closed 15m/1h(/4h) context.
                raw_five = frames.get("5m")
                try:
                    prepared_five = self.c.market_engine._prepare_frame(
                        raw_five, symbol, domain, "5m")
                    prefilter_info = agent.strategy.prefilter_5m(prepared_five)
                except Exception as error:
                    self._record_symbol_audit(
                        domain, symbol, "prefilter_error",
                        increment="errors", reason=str(error),
                    )
                    return
                if not prefilter_info.get("candidate"):
                    with self._analysis_stats_lock:
                        self._analysis_stats[domain]["prefilter_rejected"] += 1
                    self._record_symbol_audit(
                        domain, symbol, "prefilter_rejected",
                        increment="prefilter_rejected",
                        reason=prefilter_info.get("reason") or "No 5m-local setup",
                    )
                    return
                with self._analysis_stats_lock:
                    self._analysis_stats[domain]["prefilter_passed"] += 1
                    self._analysis_stats[domain]["deep_analysis"] += 1
                self._record_symbol_audit(
                    domain, symbol, "deep_analysis",
                    increment="prefilter_passed", also_increment="deep_analysis",
                    candidate_setups=prefilter_info.get("setups", []),
                )
                analysis = self.c.analyze_live(
                    symbol, domains=[domain], include_orderbook=False,
                    agent_overrides={domain: agent},
                )
                signal_source = "BINANCE_WEBSOCKET_5M_PREFILTER_REST_CONTEXT"
            else:
                # Explicit top-N mode retains the original multi-timeframe cache path.
                analysis = self.c.analyze_live_from_frames(
                    symbol, domain, frames, agent=agent
                )
                signal_source = "BINANCE_WEBSOCKET_CACHE"

            signal = (analysis.get("signals") or {}).get(domain)
            if not isinstance(signal, dict):
                reason = str(
                    (analysis.get("data_errors") or {}).get(domain)
                    or (analysis.get("data_errors") or {}).get("market_data")
                    or "Agent returned no V3 signal"
                )
                self._record_symbol_audit(
                    domain, symbol, "strategy_rejected", reason=reason,
                )
                self.last_result = {
                    "status": "SKIPPED", "phase": "websocket_analysis",
                    "domain": domain, "symbol": symbol, "reason": reason,
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
            signal["signal_source"] = signal_source

            plan = signal.get("trade_plan")
            direction = str(signal.get("direction", "WAIT")).upper()
            valid_direction = direction in (("BUY",) if domain == "spot"
                                             else ("LONG", "SHORT"))
            if (not valid_direction or signal.get("strategy_version") != "v3"
                    or not isinstance(plan, dict) or plan.get("success") is not True
                    or plan.get("strategy_version") != "v3"
                    or str(plan.get("timeframe", "")).lower() != "5m"):
                reason = str(
                    (plan or {}).get("reason")
                    or signal.get("reason")
                    or "V3 did not approve a valid 5m entry plan"
                )
                self._record_symbol_audit(
                    domain, symbol, "strategy_rejected", reason=reason,
                )
                return

            with self._analysis_stats_lock:
                self._analysis_stats[domain]["actionable"] += 1
            self._record_symbol_audit(
                domain, symbol, "signal_ready", increment="actionable",
                direction=direction, setup=plan.get("setup"),
                confidence=float(signal.get("confidence") or 0),
                analysis_duration_ms=signal["analysis_duration_ms"],
            )
            with self._event_lock:
                self.c.last_signals[domain] = signal
                cfg = dict(self.config)
                if audit_only:
                    self._record_symbol_audit(
                        domain, symbol, "audit_only_candidate", increment="audit_only",
                        reason="Delayed cache/worker replay is audited but cannot open a late trade",
                        direction=direction, setup=plan.get("setup"),
                    )
                    return
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
            self._record_symbol_audit(
                domain, symbol, "entry_queued", increment="entry_queued",
                next_open_time_ms=expected_open,
            )
            # If the next-candle event arrived during REST context analysis,
            # consume it immediately, but execution still validates its age,
            # exact candle, price deviation, risk and agent state.
            if already_open and int(already_open.get("open_time", 0)) == expected_open:
                with self._event_lock:
                    current = self._pending_entries.pop(key, None)
                if current:
                    try:
                        self._event_executor.submit(
                            self._execute_ws_entry, domain, symbol, current, already_open)
                        self._record_symbol_audit(
                            domain, symbol, "entry_submitted", increment="entry_submitted",
                            next_open_time_ms=expected_open,
                        )
                    except Exception as error:
                        self._record_symbol_audit(
                            domain, symbol, "execution_error", increment="errors",
                            reason="Execution writer submission failed: " + str(error),
                        )
        except Exception as error:
            with self._analysis_stats_lock:
                self._analysis_stats[domain]["failed"] += 1
            self._record_symbol_audit(
                domain, symbol, "analysis_error",
                increment="errors", reason=str(error),
            )
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
            with self._event_lock:
                record = self._symbol_audits[domain].get(symbol)
                if record is not None:
                    record["analysis_completed_open_time_ms"] = int(candle.get("open_time") or 0)
                    record["analysis_completed_close_time_ms"] = int(candle.get("close_time") or 0)
                    record["analysis_completed_at_ms"] = completed_at_ms
                    record["analysis_completed_error"] = record.get("state") in {
                        "analysis_error", "prefilter_error", "stale",
                        "waiting_cache", "agent_stopped"
                    }

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
            if result.get("opened"):
                self._record_symbol_audit(
                    domain, symbol, "opened", increment="opened",
                    trade_ids=[item.get("trade_id") for item in result["opened"]],
                )
            elif result.get("error"):
                self._record_symbol_audit(
                    domain, symbol, "execution_error", increment="errors",
                    reason=result.get("error"),
                )
            elif result.get("skipped"):
                reasons = [str(item.get("reason", "")) for item in result["skipped"]]
                self._record_symbol_audit(
                    domain, symbol, "entry_skipped", increment="entry_skipped",
                    reason="; ".join(reason for reason in reasons if reason)[:500],
                )
            self.last_run = int(self.clock())
            self.last_result = result
            self._last_execution_result = {
                "timestamp": self.last_run, "domain": domain, "symbol": symbol,
                "status": result.get("status"), "phase": result.get("phase"),
                "opened": list(result.get("opened") or []),
                "skipped": list(result.get("skipped") or []),
                "error": result.get("error"),
            }

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
