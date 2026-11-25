"""Leakage-resistant walk-forward evidence scoring for offline research.

The scorer learns feature-bin evidence from the training window only. It is
not connected to order execution and must not be treated as a trading signal.
"""
from collections import defaultdict
from math import isfinite


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if isfinite(value) else None


def chronological_split(rows, train_fraction=0.7):
    """Sort by timestamp and split once; equal timestamps stay on one side."""
    if not 0.5 <= float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1)")
    ordered = sorted(rows, key=lambda row: row["timestamp"])
    if len(ordered) < 4:
        raise ValueError("at least four labelled rows are required")
    cut = int(len(ordered) * float(train_fraction))
    cut = max(2, min(len(ordered) - 2, cut))
    # Do not split a timestamp group across train and test.
    boundary = ordered[cut - 1]["timestamp"]
    while cut < len(ordered) and ordered[cut]["timestamp"] == boundary:
        cut += 1
    if cut >= len(ordered):
        raise ValueError("not enough distinct timestamps for an OOS window")
    return ordered[:cut], ordered[cut:]


def fit_evidence_model(train_rows, min_bin_samples=8, smoothing=8.0):
    """Fit univariate quantile-bin mean-R evidence using training labels only."""
    if min_bin_samples < 2 or smoothing < 0:
        raise ValueError("invalid evidence model settings")
    clean = []
    for row in train_rows:
        result = _number(row.get("result_r"))
        if result is None:
            continue
        features = {str(k): _number(v) for k, v in
                    (row.get("features") or {}).items()}
        clean.append((features, result))
    if len(clean) < min_bin_samples:
        raise ValueError("insufficient labelled training rows")
    prior = sum(result for _, result in clean) / len(clean)
    model = {"prior": prior, "features": {}, "train_count": len(clean),
             "min_bin_samples": int(min_bin_samples), "smoothing": float(smoothing)}
    keys = sorted({key for features, _ in clean for key, value in features.items()
                   if value is not None})
    for key in keys:
        values = sorted(value for features, _ in clean
                        if (value := features.get(key)) is not None)
        if len(values) < min_bin_samples * 2:
            continue
        # Three train-only quantile groups. Boundaries never inspect OOS data.
        q1 = values[len(values) // 3]
        q2 = values[(2 * len(values)) // 3]
        groups = [[], [], []]
        for features, result in clean:
            value = features.get(key)
            if value is None:
                continue
            bucket = 0 if value <= q1 else (1 if value <= q2 else 2)
            groups[bucket].append(result)
        bins = []
        for group in groups:
            if len(group) < min_bin_samples:
                bins.append(None)
            else:
                mean = sum(group) / len(group)
                estimate = (len(group) * mean + smoothing * prior) / (len(group) + smoothing)
                bins.append({"count": len(group), "mean_r": mean,
                             "evidence_r": estimate})
        if any(item is not None for item in bins):
            model["features"][key] = {"q1": q1, "q2": q2, "bins": bins}
    return model


def score_evidence(model, features):
    """Return an interpretable score; missing or under-sampled bins add no weight."""
    contributions = {}
    for key, spec in model.get("features", {}).items():
        value = _number((features or {}).get(key))
        if value is None:
            continue
        bucket = 0 if value <= spec["q1"] else (1 if value <= spec["q2"] else 2)
        item = spec["bins"][bucket]
        if item is not None:
            contributions[key] = item["evidence_r"] - model["prior"]
    return {"score_r": sum(contributions.values()),
            "prior_r": model["prior"], "contributions": contributions}

def evaluate_portfolio_walk_forward(rows, train_fraction=0.7,
                                    min_bin_samples=8, smoothing=8.0,
                                    max_positions_per_candle=3,
                                    max_per_setup_per_candle=1):
    """Cross-sectional OOS selection, capped per candle and setup.

    All candidates at a timestamp are ranked together; the model is fitted only
    on earlier training rows. This is a research simulation, not execution logic.
    """
    if max_positions_per_candle < 1 or max_per_setup_per_candle < 1:
        raise ValueError("portfolio caps must be positive")
    train, test = chronological_split(rows, train_fraction)
    model = fit_evidence_model(train, min_bin_samples, smoothing)
    groups = defaultdict(list)
    for row in test:
        result = _number(row.get("result_r"))
        stamp = row.get("timestamp")
        if result is None or stamp is None:
            continue
        scored = score_evidence(model, row.get("features"))["score_r"]
        groups[stamp].append((scored, result, str(row.get("symbol", "")),
                              str(row.get("setup", ""))))
    selected, baseline = [], []
    for stamp in sorted(groups):
        group = sorted(groups[stamp], key=lambda item: item[0], reverse=True)
        baseline.extend(item[1] for item in group)
        seen_symbols, setup_counts = set(), defaultdict(int)
        for _, result, symbol, setup in group:
            if symbol in seen_symbols:
                continue
            if setup_counts[setup] >= max_per_setup_per_candle:
                continue
            selected.append(result)
            seen_symbols.add(symbol)
            setup_counts[setup] += 1
            if len(seen_symbols) >= max_positions_per_candle:
                break
    if not selected:
        raise ValueError("portfolio selector produced no OOS selections")
    def stats(values):
        wins = sum(value > 0 for value in values)
        gross_win = sum(value for value in values if value > 0)
        gross_loss = -sum(value for value in values if value < 0)
        return {"count": len(values), "mean_r": sum(values) / len(values),
                "win_rate": wins / len(values),
                "profit_factor": gross_win / gross_loss if gross_loss else None}
    return {"train_count": len(train), "oos_candidate_count": len(baseline),
            "selected": stats(selected), "baseline": stats(baseline),
            "max_positions_per_candle": max_positions_per_candle,
            "max_per_setup_per_candle": max_per_setup_per_candle,
            "model_feature_count": len(model["features"])}


def evaluate_walk_forward(rows, train_fraction=0.7, min_bin_samples=8,
                          smoothing=8.0, selection_fraction=0.25):
    """Fit on the chronological train window and report untouched OOS results."""
    if not 0 < selection_fraction <= 1:
        raise ValueError("selection_fraction must be in (0, 1]")
    train, test = chronological_split(rows, train_fraction)
    model = fit_evidence_model(train, min_bin_samples, smoothing)
    scored = []
    for row in test:
        result = _number(row.get("result_r"))
        if result is None:
            continue
        scored.append((score_evidence(model, row.get("features"))["score_r"], result))
    if not scored:
        raise ValueError("OOS window has no valid labels")
    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected_n = max(1, int(len(scored) * selection_fraction))
    selected = [result for _, result in scored[:selected_n]]
    baseline = [result for _, result in scored]
    def stats(values):
        wins = sum(value > 0 for value in values)
        gross_win = sum(value for value in values if value > 0)
        gross_loss = -sum(value for value in values if value < 0)
        return {"count": len(values), "mean_r": sum(values) / len(values),
                "win_rate": wins / len(values),
                "profit_factor": (gross_win / gross_loss if gross_loss else None)}
    return {"train_count": len(train), "oos_count": len(test),
            "model_feature_count": len(model["features"]),
            "selected": stats(selected), "baseline": stats(baseline),
            "selection_fraction": selection_fraction,
            "model": model}
