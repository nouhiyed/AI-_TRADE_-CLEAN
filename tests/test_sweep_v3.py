import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sweep_v3 as sw  # noqa: E402
from core.scalping_v3 import FuturesScalpingStrategyV3  # noqa: E402

DAY = 86_400_000
T0 = 1_700_000_000_000
T1 = T0 + 400 * DAY


def trades(symbol, start_day, count, avg_r, jitter=0.4, seed=1):
    """Deterministic trades spread over [start_day, start_day+count) days."""
    rng = np.random.default_rng(seed)
    values = avg_r + rng.uniform(-jitter, jitter, count)
    return [{"symbol": symbol, "entry_ms": int(T0 + (start_day + i) * DAY + 3600_000),
             "direction": "LONG", "setup": "X", "exit_reason": "STOP_LOSS",
             "net_R": float(v)} for i, v in enumerate(values)]


class SweepProtocolTests(unittest.TestCase):
    def test_periods_are_chronological_and_complete(self):
        bounds = sw.split_bounds(T0, T1)
        self.assertEqual(sw.assign_period(T0 + 10 * DAY, bounds), "design")
        self.assertEqual(sw.assign_period(T0 + 250 * DAY, bounds), "validation")
        self.assertEqual(sw.assign_period(T0 + 350 * DAY, bounds), "holdout")

    def test_critical_z_grows_with_number_of_trials(self):
        self.assertAlmostEqual(sw.critical_z(1), 1.645, places=2)
        self.assertGreater(sw.critical_z(20), sw.critical_z(5))

    def test_patched_restores_class_attributes(self):
        original = FuturesScalpingStrategyV3.PULLBACK_VOLUME
        with sw.patched(FuturesScalpingStrategyV3, {"PULLBACK_VOLUME": 0.5}):
            self.assertEqual(FuturesScalpingStrategyV3.PULLBACK_VOLUME, 0.5)
        self.assertEqual(FuturesScalpingStrategyV3.PULLBACK_VOLUME, original)

    def _base(self):
        return (trades("A", 0, 60, -0.05, seed=1) + trades("A", 200, 40, -0.05, seed=2))

    def test_harmless_relaxation_is_accepted(self):
        base = self._base()
        extra = trades("B", 0, 40, -0.05, seed=3) + trades("B", 200, 30, -0.05, seed=4)
        rows, _ = sw.evaluate({"baseline": base, "relax": base + extra}, T0, T1, 1)
        verdict = {r["id"]: r for r in rows}["relax"]
        self.assertEqual(verdict["verdict"], "ACCEPT", verdict.get("reasons"))

    def test_harmful_marginal_trades_are_rejected(self):
        base = self._base()
        bad = trades("B", 0, 60, -0.6, seed=5) + trades("B", 200, 40, -0.6, seed=6)
        rows, _ = sw.evaluate({"baseline": base, "relax": base + bad}, T0, T1, 1)
        verdict = {r["id"]: r for r in rows}["relax"]
        self.assertEqual(verdict["verdict"], "REJECT")
        self.assertTrue(any("marginal" in r or "avgR" in r for r in verdict["reasons"]))

    def test_no_extra_trades_is_rejected(self):
        base = self._base()
        rows, _ = sw.evaluate({"baseline": base, "relax": list(base)}, T0, T1, 1)
        verdict = {r["id"]: r for r in rows}["relax"]
        self.assertEqual(verdict["verdict"], "REJECT")

    def test_edge_requires_positive_significant_validation(self):
        winning = trades("A", 0, 80, 0.4, jitter=0.2, seed=7) + trades("A", 200, 80, 0.4, jitter=0.2, seed=8)
        rows, _ = sw.evaluate({"baseline": winning}, T0, T1, 1)
        self.assertTrue(rows[0]["edge_proven"])
        losing = self._base()
        rows, _ = sw.evaluate({"baseline": losing}, T0, T1, 1)
        self.assertFalse(rows[0]["edge_proven"])

    def test_holdout_is_sealed_from_the_report_rows(self):
        base = self._base() + trades("A", 320, 30, 0.9, seed=9)
        rows, extra = sw.evaluate({"baseline": base}, T0, T1, 1)
        self.assertNotIn("holdout", rows[0])
        self.assertEqual(extra["sealed"]["baseline"]["n"], 30)

    def test_random_control_uses_the_real_plan_construction(self):
        random_cls = sw.make_random_class(FuturesScalpingStrategyV3, 1.0, 7)
        self.assertTrue(random_cls.prefilter_5m(None)["candidate"])
        import pandas as pd
        frame = pd.DataFrame({"ATR": [0.5], "close": [100.0]})
        plan = random_cls().analyze({"5m": frame})
        self.assertTrue(plan["success"])
        self.assertIn(plan["direction"], ("LONG", "SHORT"))
        self.assertGreater(abs(plan["entry"] - plan["stop_loss"]), 0)


if __name__ == "__main__":
    unittest.main()
