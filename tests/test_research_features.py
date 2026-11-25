import unittest

from core.research_features import extract_research_features, classify_market_regime


class ResearchFeatureTests(unittest.TestCase):
    def candles(self, count=40, slope=0.2):
        rows = []
        price = 100.0
        for index in range(count):
            open_price = price
            price += slope
            rows.append({
                "open": open_price, "high": price + 0.5,
                "low": open_price - 0.2, "close": price,
                "volume": 100 + index, "is_closed": True,
            })
        return rows

    def test_extracts_causal_numeric_features(self):
        features = extract_research_features({
            "5m": self.candles(), "15m": self.candles(slope=0.4),
            "1h": self.candles(slope=0.8),
        })
        self.assertIn("5m__atr_pct_14", features)
        self.assertIn("5m__rsi14", features)
        self.assertIn("1h__ema20_slope_pct", features)
        self.assertTrue(all(isinstance(v, float) for v in features.values()))

    def test_ignores_unclosed_latest_candle(self):
        rows = self.candles()
        rows[-1]["close"] = 100000
        rows[-1]["is_closed"] = False
        features = extract_research_features({"5m": rows})
        self.assertLess(features["5m__close_location"], 1.0)

    def test_regime_classifier_supports_normalized_feature_names(self):
        self.assertEqual(classify_market_regime({
            "15m__adx14_proxy": 30, "1h__ema20_slope_pct": 0.2,
        }), "TREND_UP")
        self.assertEqual(classify_market_regime({
            "5m__atr_pct_14": 2.5,
        }), "HIGH_VOLATILITY")


if __name__ == "__main__":
    unittest.main()
