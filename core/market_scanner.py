from core.binance_market_data import BinanceMarketData


class FuturesMarketScanner:
    TOP_SYMBOL_LIMIT = 250

    def __init__(self, timeout=6, market=None):
        # Share the coordinator's market feed so the daily top-250 selection
        # is fetched once and reused by WebSocket and scanner workers.
        self.market = market or BinanceMarketData(timeout=timeout, total_timeout=timeout)

    def scan_universe(self, limit=None, min_quote_volume=None):
        """Return only the current daily top-250 liquid USDT perpetuals."""
        if limit is not None:
            try:
                limit = min(self.TOP_SYMBOL_LIMIT, max(1, int(limit)))
            except (TypeError, ValueError):
                raise ValueError("Invalid scan limit")
        minimum = float(min_quote_volume) if min_quote_volume is not None else None

        ranked = self.market.fetch_top_symbols(
            "futures", limit=self.TOP_SYMBOL_LIMIT
        )
        if minimum is not None:
            ranked = [
                item for item in ranked
                if float(item.get("quote_volume", 0) or 0) >= minimum
            ]
        return ranked if limit is None else ranked[:limit]
