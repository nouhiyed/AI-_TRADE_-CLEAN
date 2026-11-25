import unittest
from core.research_validation import (
    chronological_split, fit_evidence_model, score_evidence,
    evaluate_walk_forward,
)


class ResearchValidationTests(unittest.TestCase):
    def rows(self):
        # Earlier training data has a real monotonic relation; later OOS rows
        # are kept separate so model fitting can be checked for leakage.
        return [{"timestamp": i, "features": {"momentum": float(i % 10)},
                 "result_r": (1.0 if i % 10 >= 7 else -0.4)}
                for i in range(100)]

    def test_chronological_split_keeps_time_order(self):
        train, test = chronological_split(self.rows(), 0.7)
        self.assertLess(max(x["timestamp"] for x in train),
                        min(x["timestamp"] for x in test))

    def test_oos_labels_do_not_change_fitted_evidence(self):
        rows = self.rows()
        train, test = chronological_split(rows, 0.7)
        first = fit_evidence_model(train, min_bin_samples=5)
        for row in test:
            row["result_r"] = 100000.0
        train2, test2 = chronological_split(rows, 0.7)
        second = fit_evidence_model(train2, min_bin_samples=5)
        self.assertEqual(first, second)

    def test_score_and_walk_forward_metrics_are_finite(self):
        rows = self.rows()
        model = fit_evidence_model(rows[:70], min_bin_samples=5)
        score = score_evidence(model, {"momentum": 9})
        self.assertIn("score_r", score)
        report = evaluate_walk_forward(rows, min_bin_samples=5)
        self.assertEqual(report["oos_count"], 30)
        self.assertEqual(report["baseline"]["count"], 30)
        self.assertEqual(report["selected"]["count"], 7)


if __name__ == "__main__":
    unittest.main()
