"""Live end-to-end Binance 5m coverage smoke test for the top-250 markets.

This runs the same WebSocket cache, exact-cycle reconciliation, and Autopilot
analysis path used by the Android backend. It never enables DEMO entries.
"""
import json
import os
import tempfile
import time
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import select_reconcile_batch
from core.autopilot import Autopilot
from core.binance_websocket import BinanceKlineStream
from core.coordinator import TradingCoordinator


def expected_closed_open(now_ms=None):
    now_ms = int(now_ms or time.time() * 1000)
    return ((now_ms // 300000) - 1) * 300000


def wait_for_stream(stream, timeout_seconds=300):
    deadline = time.monotonic() + timeout_seconds
    last = {}
    while time.monotonic() < deadline:
        last = stream.status()
        ready = all(
            int(last.get(market, {}).get("cache_ready_symbols", 0)) == 250
            and int(last.get(market, {}).get("symbols", 0)) == 250
            and bool(last.get(market, {}).get("fully_connected"))
            for market in ("futures", "spot")
        )
        if ready:
            return last
        time.sleep(2)
    raise RuntimeError("Live stream/cache warmup timeout: " + json.dumps({
        market: {
            "symbols": last.get(market, {}).get("symbols"),
            "connected": last.get(market, {}).get("fully_connected"),
            "cache_ready": last.get(market, {}).get("cache_ready_symbols"),
            "last_error": last.get(market, {}).get("last_error"),
            "warmup_error": last.get(market, {}).get("cache_warmup_last_error"),
        }
        for market in ("futures", "spot")
    }, sort_keys=True))


def collect_exact_cycle(stream, autopilot, domain, timeout_seconds=240):
    symbols = stream.registered_symbols(domain)
    if len(symbols) != 250 or len(set(symbols)) != 250:
        raise RuntimeError("%s universe is not exactly 250 unique pairs" % domain)

    cycle_open = expected_closed_open()
    pending = set(symbols)
    cursor = 0
    deadline = time.monotonic() + timeout_seconds
    last_missing = set(pending)

    while pending and time.monotonic() < deadline:
        batch_symbols, cursor = select_reconcile_batch(
            symbols, pending, cursor, 96
        )
        if not batch_symbols:
            time.sleep(1)
            continue
        batch = stream.latest_closed_events(
            domain, cycle_open, symbols=batch_symbols
        )
        for event in batch.get("events", []):
            symbol = str(event.get("symbol", "")).upper()
            if not symbol:
                continue
            if autopilot.has_seen_candle(domain, symbol, cycle_open):
                pending.discard(symbol)
                continue
            accepted = autopilot.on_market_kline(domain, event)
            if accepted or autopilot.has_seen_candle(domain, symbol, cycle_open):
                pending.discard(symbol)
        last_missing = set(pending)
        if pending:
            time.sleep(2)

    if pending:
        raise RuntimeError(
            "%s exact-cycle cache/reconcile incomplete: %s/250 missing; sample=%s"
            % (domain, len(pending), sorted(pending)[:30])
        )
    return cycle_open, symbols


def wait_for_analysis(autopilot, cycles, timeout_seconds=360):
    deadline = time.monotonic() + timeout_seconds
    snapshots = {}
    while time.monotonic() < deadline:
        complete = True
        for domain, (cycle_open, symbols) in cycles.items():
            audit = autopilot.completed_symbols_for_cycle(
                domain, cycle_open, symbols
            )
            snapshots[domain] = audit
            if len(audit.get("completed", [])) != 250:
                complete = False
        if complete:
            failures = {
                domain: audit.get("errors", [])
                for domain, audit in snapshots.items()
                if audit.get("errors")
            }
            if failures:
                raise RuntimeError("Analysis completed with errors: " + json.dumps(
                    {key: value[:30] for key, value in failures.items()},
                    sort_keys=True,
                ))
            return snapshots
        time.sleep(2)
    raise RuntimeError("Analysis completion timeout: " + json.dumps({
        domain: {
            "completed": len(audit.get("completed", [])),
            "errors": len(audit.get("errors", [])),
        }
        for domain, audit in snapshots.items()
    }, sort_keys=True))


def main():
    coordinator = None
    autopilot = None
    stream = None
    with tempfile.TemporaryDirectory(prefix="ai-trader-live-smoke-") as temp:
        try:
            coordinator = TradingCoordinator(initial_balance=1000)
            autopilot = Autopilot(
                coordinator, config_path=os.path.join(temp, "autopilot.json")
            )
            # Analyze every eligible candle but never submit a trade.
            autopilot.config["enabled"] = False
            autopilot.config["trade_futures"] = False
            autopilot.config["trade_spot"] = False
            stream = BinanceKlineStream(
                coordinator.market_feed,
                autopilot.on_market_kline,
                max_symbols=250,
                on_universe=autopilot.register_symbol_universe,
                on_cache_ready=autopilot.on_symbol_cache_ready,
            )
            stream.start()
            status = wait_for_stream(stream)
            cycles = {}
            for domain in ("futures", "spot"):
                cycle_open, symbols = collect_exact_cycle(
                    stream, autopilot, domain
                )
                cycles[domain] = (cycle_open, symbols)
            completed = wait_for_analysis(autopilot, cycles)
            result = {
                "success": True,
                "markets": {
                    domain: {
                        "expected_symbols": 250,
                        "accepted_exact_cycle": 250,
                        "analysis_completed": len(completed[domain]["completed"]),
                        "analysis_errors": len(completed[domain]["errors"]),
                        "cycle_open_time_ms": cycles[domain][0],
                    }
                    for domain in ("futures", "spot")
                },
                "websocket": {
                    domain: {
                        "fully_connected": status[domain]["fully_connected"],
                        "cache_ready_symbols": status[domain]["cache_ready_symbols"],
                    }
                    for domain in ("futures", "spot")
                },
                "trading_enabled": False,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
        finally:
            if stream is not None:
                stream.stop()
            if autopilot is not None:
                autopilot._event_executor.shutdown(wait=True)
                for pool in autopilot._analysis_pools.values():
                    pool.shutdown(wait=True)
            if coordinator is not None:
                try:
                    coordinator.demo_database.connection.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
