import unittest

import pandas as pd

from core.binance_websocket import BinanceKlineStream


class SparseCandleAggregationTests(unittest.TestCase):
    def setUp(self):
        self.stream = BinanceKlineStream(
            market_feed=object(),
            on_kline=lambda *_args, **_kwargs: None,
        )

    @staticmethod
    def _five_minute_rows(open_times):
        rows = []
        for index, opened in enumerate(open_times, start=1):
            rows.append({
                "open_time": opened,
                "open": float(index * 10),
                "high": float(index * 10 + 3),
                "low": float(index * 10 - 2),
                "close": float(index * 10 + 1),
                "volume": float(index),
                "close_time": opened + pd.Timedelta(minutes=5)
                - pd.Timedelta(milliseconds=1),
                "is_closed": True,
            })
        return pd.DataFrame(rows)

    def test_15m_aggregation_accepts_no_trade_5m_gap(self):
        bucket = pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta(minutes=30)
        frame = self._five_minute_rows([
            bucket,
            bucket + pd.Timedelta(minutes=10),
        ])

        result = self.stream._aggregate_5m_frame(
            "spot", "BTCUSDT", "15m", frame
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["open"], 10.0)
        self.assertEqual(result.iloc[0]["high"], 23.0)
        self.assertEqual(result.iloc[0]["low"], 8.0)
        self.assertEqual(result.iloc[0]["close"], 21.0)
        self.assertEqual(result.iloc[0]["volume"], 3.0)
        self.assertEqual(
            result.iloc[0]["close_time"],
            bucket + pd.Timedelta(minutes=15) - pd.Timedelta(milliseconds=1),
        )

    def test_1h_aggregation_accepts_missing_no_trade_intervals(self):
        bucket = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=2)
        open_times = [
            bucket + pd.Timedelta(minutes=5 * index)
            for index in range(12) if index != 5
        ]
        frame = self._five_minute_rows(open_times)

        result = self.stream._aggregate_5m_frame(
            "spot", "BTCUSDT", "1h", frame
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["open"], 10.0)
        self.assertEqual(result.iloc[0]["close"], 111.0)
        self.assertEqual(result.iloc[0]["volume"], sum(range(1, 12)))
        self.assertEqual(
            result.iloc[0]["close_time"],
            bucket + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        )

    def test_spot_5m_bootstrap_has_history_buffer_for_200_hour_rows(self):
        self.assertEqual(
            BinanceKlineStream.CACHE_LIMITS["spot"]["5m"],
            3000,
        )


if __name__ == "__main__":
    unittest.main()
