"""Run a chronological OOS evidence report from the research candidate ledger."""
import argparse
import json
from pathlib import Path

from core.research_candidate_store import ResearchCandidateStore
from core.research_validation import evaluate_walk_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/research_candidates.db")
    parser.add_argument("--market", choices=("futures", "spot", "alpha"))
    parser.add_argument("--setup")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--min-bin-samples", type=int, default=8)
    parser.add_argument("--selection-fraction", type=float, default=0.25)
    parser.add_argument("--output")
    args = parser.parse_args()

    store = ResearchCandidateStore(Path(args.db))
    rows = store.labelled_dataset(args.market, args.setup)
    if len(rows) < 4:
        raise SystemExit(
            "Not enough completed forward labels (%s). Keep collecting closed candles; "
            "no score or performance claim is produced." % len(rows)
        )
    report = evaluate_walk_forward(
        rows, train_fraction=args.train_fraction,
        min_bin_samples=args.min_bin_samples,
        selection_fraction=args.selection_fraction,
    )
    report["scope"] = {"market": args.market, "setup": args.setup,
                       "labelled_rows": len(rows), "database": args.db}
    # Keep model bin details for reproducibility, but never call this a live signal.
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        Path(args.output).write_text(rendered + "\\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
