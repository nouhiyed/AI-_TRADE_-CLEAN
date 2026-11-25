import unittest
from core.edge_backtest import run_chronological_backtest, stress_test


def synthetic_bars(n=520):
    rows = []
    price = 100.0
    for i in range(n):
        # Deterministic alternating drift and volatility to exercise chronology.
        drift = 0.16 if (i // 65) % 2 == 0 else -0.13
        op = price
        close = op + drift
        spread = 0.35 + (0.15 if i % 9 == 0 else 0)
        rows.append({
            "open_time": i * 300000, "open": op,
            "high": max(op, close) + spread,
            "low": min(op, close) - spread,
            "close": close, "volume": 100 + (i % 17) * 3,
        })
        price = close
    return rows


class EdgeBacktestTests(unittest.TestCase):
    def test_insufficient_history_is_explicit(self):
        result = run_chronological_backtest(synthetic_bars(80))
        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertEqual(result["overall"]["closed"], 0)

    def test_chronological_oos_and_cost_stress_report(self):
        rows = synthetic_bars()
        result = run_chronological_backtest(rows, market="futures")
        self.assertEqual(result["execution"], "NEXT_BAR_OPEN")
        self.assertGreater(result["oos_bars"], 0)
        self.assertEqual(set(result["by_strategy"]),
                         {"PULLBACK", "BREAKOUT", "RECLAIM", "RANGE"})
        self.assertTrue(all(t["entry_time"] > t["signal_time"]
                            for t in result["trades"]))
        stressed = stress_test(rows)
        self.assertEqual(set(stressed),
                         {"base", "double_costs", "severe_costs"})
        self.assertEqual(stressed["severe_costs"]["fee_rate"], 0.003)


if __name__ == "__main__":
    unittest.main()
