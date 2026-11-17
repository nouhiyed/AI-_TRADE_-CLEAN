"""Live end-to-end Binance 5m coverage smoke test.

Runs the production API reconciliation worker against Binance public market
data. It never enables DEMO entries. A market is reported as verified only
when its WebSocket, 250-symbol cache, exact 5m cycle, and all 250 analyses pass.
"""
import json
import os
import sys
import tempfile
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import api.main as api_runtime
from core.autopilot import Autopilot
from core.binance_websocket import BinanceKlineStream
from core.coordinator import TradingCoordinator


def wait_for_stream(stream, timeout_seconds=180):
    deadline = time.monotonic() + timeout_seconds
    last = {}
    ready = set()
    while time.monotonic() < deadline:
        last = stream.status()
        ready = {
            market for market in ("futures", "spot")
            if int(last.get(market, {}).get("symbols", 0)) == 250
            and int(last.get(market, {}).get("cache_ready_symbols", 0)) == 250
            and bool(last.get(market, {}).get("fully_connected"))
        }
        if len(ready) == 2:
            break
        time.sleep(2)
    return last, ready


def wait_for_coverage(api_module, markets, timeout_seconds=240):
    deadline = time.monotonic() + timeout_seconds
    latest = {}
    while time.monotonic() < deadline:
        done = set()
        for market in markets:
            coverage = dict(
                api_module.scan_status.get(market, {}).get("coverage") or {}
            )
            latest[market] = coverage
            if (
                coverage.get("complete") is True
                and int(coverage.get("expected_symbols", 0)) == 250
                and int(coverage.get("accepted_symbols", 0)) == 250
                and int(coverage.get("completed_symbols", 0)) == 250
                and int(coverage.get("pending_cache_symbols", -1)) == 0
                and int(coverage.get("pending_analysis_symbols", -1)) == 0
                and int(coverage.get("pending_reconcile_symbols", -1)) == 0
                and int(coverage.get("queue_rejected_symbols", -1)) == 0
                and int(coverage.get("analysis_errors", -1)) == 0
            ):
                done.add(market)
        if done == set(markets):
            return latest
        time.sleep(2)
    raise RuntimeError("Production API coverage did not reach 250/250: " + json.dumps(
        latest, sort_keys=True
    ))


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

            # Exercise the exact production startup ordering: reconciliation
            # worker must be alive before WebSocket subscriptions can deliver.
            api_runtime.coordinator = coordinator
            api_runtime.autopilot = autopilot
            api_runtime.market_stream = stream
            api_runtime.reconciliation_ready.clear()
            reconcile_thread = threading.Thread(
                target=api_runtime.full_universe_reconcile_worker,
                name="live-smoke-production-reconciler",
                daemon=True,
            )
            reconcile_thread.start()
            if not api_runtime.reconciliation_ready.wait(timeout=5):
                raise RuntimeError("Production reconciliation worker did not start")

            stream.start()
            status, ready_markets = wait_for_stream(stream)
            coverage = {}
            if ready_markets:
                coverage = wait_for_coverage(api_runtime, ready_markets)

            report = {
                "success": ready_markets == {"futures", "spot"}
                    and set(coverage) == {"futures", "spot"},
                "markets": {
                    market: {
                        "websocket_connected": bool(
                            status.get(market, {}).get("fully_connected")
                        ),
                        "symbols": int(status.get(market, {}).get("symbols", 0)),
                        "cache_ready_symbols": int(
                            status.get(market, {}).get("cache_ready_symbols", 0)
                        ),
                        "last_error": status.get(market, {}).get("last_error"),
                        "coverage": coverage.get(market),
                        "verified": market in coverage,
                    }
                    for market in ("futures", "spot")
                },
                "trading_enabled": False,
            }
            print(json.dumps(report, indent=2, sort_keys=True))
            if not report["success"]:
                raise RuntimeError(
                    "Live Binance verification incomplete; no APK may be released"
                )
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
