"""Persistent round-robin batching for complete Binance market scans."""
import json
import threading
from pathlib import Path

_LOCK = threading.Lock()
_STATE_FILE = Path(__file__).resolve().parents[1] / "config" / "scan_rotation.json"


def next_scan_batch(domain, universe, batch_size=20):
    """Return the next stable alphabetical batch; every eligible symbol is covered once per cycle."""
    # The incoming universe may be ranked by changing 24h volume. Never use that
    # volatile order for a persistent cursor: sort by symbol for stable coverage.
    items = sorted(
        list(universe),
        key=lambda item: str(item.get("symbol", "")).upper(),
    )
    if not items:
        return [], {
            "universe_total": 0,
            "batch_size": 0,
            "cycle": 0,
            "cycle_complete": True,
            "last_symbol": None,
        }

    symbols = [str(item.get("symbol", "")).upper() for item in items]
    if any(not symbol for symbol in symbols):
        raise ValueError("Scan universe contains an empty symbol")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Scan universe contains duplicate symbols")

    with _LOCK:
        try:
            state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                state = {}
        except (OSError, ValueError, TypeError):
            state = {}

        saved = state.get(domain, {})
        if isinstance(saved, dict):
            last_symbol = str(saved.get("last_symbol", "")).upper()
            cycle = max(0, int(saved.get("cycle", 0) or 0))
        else:
            last_symbol = str(saved or "").upper()
            cycle = 0

        if last_symbol in symbols:
            start = (symbols.index(last_symbol) + 1) % len(items)
        else:
            # If the previous symbol disappeared from Binance, resume at the
            # next lexical symbol rather than restarting or losing coverage.
            start = next(
                (index for index, symbol in enumerate(symbols)
                 if symbol > last_symbol),
                0,
            )

        # Do not wrap inside a batch. The final batch may be smaller.
        # This guarantees exactly-once coverage for a stable universe.
        count = min(max(1, int(batch_size)), len(items) - start)
        batch = items[start:start + count]
        wrapped = start + count == len(items)
        if wrapped:
            cycle += 1

        last = str(batch[-1].get("symbol", "")).upper()
        state[domain] = {"last_symbol": last, "cycle": cycle}

        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = _STATE_FILE.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, indent=2),
            encoding="utf-8",
        )
        temporary.replace(_STATE_FILE)

    return batch, {
        "universe_total": len(items),
        "batch_size": len(batch),
        "cycle": cycle,
        "cycle_complete": wrapped,
        "last_symbol": last,
    }
