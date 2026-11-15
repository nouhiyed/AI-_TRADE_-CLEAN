import pandas as pd
import numpy as np


def analyze_market_structure(df, swing_window=3):
    """Calculate causal SMC features with vectorized rolling operations.

    The previous implementation repeatedly sliced pandas DataFrames and
    assigned individual cells inside a per-candle loop. That made scanning
    thousands of live symbols CPU-bound. This version preserves the same
    confirmed-swing timing and BOS/CHoCH state transitions while moving
    sweeps, FVGs, order blocks and trend calculations to NumPy/pandas vectors.
    """
    df = df.copy().reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n = len(df)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    # A swing at pivot p is confirmed at p + swing_window. The centered
    # rolling window matches the original causal [p-w, p+w] comparison.
    swing_window = max(0, int(swing_window))
    swing_width = 2 * swing_window + 1
    rolling_high = (
        df["high"]
        .rolling(swing_width, center=True, min_periods=swing_width)
        .max()
        .to_numpy()
    )
    rolling_low = (
        df["low"]
        .rolling(swing_width, center=True, min_periods=swing_width)
        .min()
        .to_numpy()
    )
    high_pivots = np.isfinite(rolling_high) & (high == rolling_high)
    low_pivots = np.isfinite(rolling_low) & (low == rolling_low)

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    if swing_window == 0:
        swing_high[:] = high_pivots
        swing_low[:] = low_pivots
    elif n > swing_window:
        swing_high[swing_window:] = high_pivots[:-swing_window]
        swing_low[swing_window:] = low_pivots[:-swing_window]

    # Preserve the original sequential structure-break state machine, but
    # operate on NumPy arrays rather than pandas iloc/loc for every candle.
    bos = np.full(n, "", dtype=object)
    choch = np.full(n, "", dtype=object)
    last_high = None
    last_low = None
    trend = 0

    for i in range(n):
        if last_high is not None and close[i] > last_high:
            if trend == -1:
                choch[i] = "BULLISH"
            else:
                bos[i] = "BULLISH"
            trend = 1
            last_high = None
        elif last_low is not None and close[i] < last_low:
            if trend == 1:
                choch[i] = "BEARISH"
            else:
                bos[i] = "BEARISH"
            trend = -1
            last_low = None

        if swing_high[i]:
            last_high = high[i - swing_window]
        if swing_low[i]:
            last_low = low[i - swing_window]

    # Liquidity sweeps compare each candle only with the preceding 20 candles.
    previous_min = (
        df["low"].rolling(20, min_periods=20).min().shift(1).to_numpy()
    )
    previous_max = (
        df["high"].rolling(20, min_periods=20).max().shift(1).to_numpy()
    )
    bullish_sweep = (low < previous_min) & (close > previous_min)
    bearish_sweep = (high > previous_max) & (close < previous_max)
    bullish_sweep[:20] = False
    bearish_sweep[:20] = False

    # Fair Value Gaps; the bearish branch remains mutually exclusive.
    bullish_fvg = np.zeros(n, dtype=bool)
    bearish_fvg = np.zeros(n, dtype=bool)
    fvg_lower = np.full(n, np.nan, dtype=float)
    fvg_upper = np.full(n, np.nan, dtype=float)
    if n > 2:
        bullish_fvg[2:] = low[2:] > high[:-2]
        bearish_fvg[2:] = (
            (~bullish_fvg[2:]) & (high[2:] < low[:-2])
        )
        fvg_lower[2:] = np.where(
            bullish_fvg[2:],
            high[:-2],
            np.where(bearish_fvg[2:], high[2:], np.nan),
        )
        fvg_upper[2:] = np.where(
            bullish_fvg[2:],
            low[2:],
            np.where(bearish_fvg[2:], low[:-2], np.nan),
        )

    # Order blocks use the previous candle and preserve bullish-first priority.
    bullish_ob = np.zeros(n, dtype=bool)
    bearish_ob = np.zeros(n, dtype=bool)
    ob_lower = np.full(n, np.nan, dtype=float)
    ob_upper = np.full(n, np.nan, dtype=float)
    if n > 2:
        bullish_ob[2:] = (
            (close[2:] > high[1:-1])
            & (close[1:-1] < open_[1:-1])
        )
        bearish_ob[2:] = (
            (~bullish_ob[2:])
            & (close[2:] < low[1:-1])
            & (close[1:-1] > open_[1:-1])
        )
        ob_mask = bullish_ob | bearish_ob
        ob_lower[ob_mask] = low[np.maximum(np.flatnonzero(ob_mask) - 1, 0)]
        ob_upper[ob_mask] = high[np.maximum(np.flatnonzero(ob_mask) - 1, 0)]

    average = df["close"].rolling(20).mean().to_numpy()
    market_trend = np.where(close > average, "BULLISH", "BEARISH")

    df["Swing_High"] = swing_high
    df["Swing_Low"] = swing_low
    df["BOS"] = bos
    df["CHoCH"] = choch
    df["Bullish_Sweep"] = bullish_sweep
    df["Bearish_Sweep"] = bearish_sweep
    df["Bullish_FVG"] = bullish_fvg
    df["Bearish_FVG"] = bearish_fvg
    df["FVG_Lower"] = fvg_lower
    df["FVG_Upper"] = fvg_upper
    df["Bullish_OB"] = bullish_ob
    df["Bearish_OB"] = bearish_ob
    df["OB_Lower"] = ob_lower
    df["OB_Upper"] = ob_upper
    df["Market_Trend"] = market_trend

    return df


if __name__ == "__main__":
    import glob
    import os

    os.makedirs("data/analysis", exist_ok=True)

    for path in glob.glob("data/raw/*.csv"):
        df = pd.read_csv(path)
        result = analyze_market_structure(df)

        name = os.path.basename(path)
        result.to_csv(
            f"data/analysis/SMC_{name}",
            index=False
        )

        print(
            f"{name} | "
            f"BOS: {(result['BOS'] != '').sum()} | "
            f"CHoCH: {(result['CHoCH'] != '').sum()} | "
            f"Sweeps: {result['Bullish_Sweep'].sum() + result['Bearish_Sweep'].sum()} | "
            f"FVG: {(result['Bullish_FVG'] | result['Bearish_FVG']).sum()}"
        )

    print("\nCausal SMC analysis completed.")

