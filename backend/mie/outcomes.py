"""
Outcome resolution — attaching what actually happened AFTER an observation, once
enough time has passed to know.

This is the mechanical embodiment of the core idea: record state, then wait, then
label with the truth. Nothing here decides whether a moment was a good trade;
it only measures the forward return, the maximum favorable excursion (MFE) and
the maximum adverse excursion (MAE) at each horizon, from raw OHLCV. The edge
model consumes these as regression/probability targets — never a hand-picked
BUY/SELL/HOLD class — so it can be asked "what's the expected value of a LONG
here" instead of only "which bucket does this fall in".
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .state import HORIZONS_SEC, horizon_label, outcome_columns


def bars_for_horizon(horizon_sec: int, bar_seconds: float) -> int:
    """How many bars of this timeframe make up a horizon. Minimum 1."""
    return max(1, round(horizon_sec / bar_seconds))


def resolve_outcomes(df: pd.DataFrame, bar_seconds: float,
                     horizons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    For every row i, compute forward return / MFE / MAE at each horizon using
    only rows AFTER i (i+1 .. i+k inclusive). Rows too close to the end of the
    frame to resolve a given horizon get NaN for that horizon — they are not
    droppable errors, they are simply "not yet known", and the store keeps them
    unresolved until fresher data arrives.

    Returns a frame indexed like df with just the outcome columns, so callers
    join it onto their feature frame explicitly.
    """
    horizons = horizons or HORIZONS_SEC
    close = df['close'].to_numpy(dtype=float)
    high = df['high'].to_numpy(dtype=float)
    low = df['low'].to_numpy(dtype=float)
    n = len(df)

    out = {}
    for h in horizons:
        k = bars_for_horizon(h, bar_seconds)
        lab = horizon_label(h)
        ret = np.full(n, np.nan)
        mfe = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        for i in range(n - k):
            c0 = close[i]
            if c0 <= 0:
                continue
            window_hi = high[i + 1:i + k + 1]
            window_lo = low[i + 1:i + k + 1]
            if len(window_hi) == 0:
                continue
            ret[i] = close[i + k] / c0 - 1.0
            mfe[i] = window_hi.max() / c0 - 1.0
            mae[i] = window_lo.min() / c0 - 1.0
        out[f'ret_{lab}'] = ret
        out[f'mfe_{lab}'] = mfe
        out[f'mae_{lab}'] = mae

    return pd.DataFrame(out, index=df.index)


def is_resolved(row: pd.Series, horizon_sec: int) -> bool:
    lab = horizon_label(horizon_sec)
    return pd.notna(row.get(f'ret_{lab}'))


def label_maturity_deadline(as_of, horizon_sec: int, bar_seconds: float):
    """
    Wall-clock time by which an observation's outcome at `horizon_sec` SHOULD be
    resolvable, given one extra bar of slack for the exchange to finalize the
    candle. Used by the store to decide when to stop waiting and try resolving.
    """
    from datetime import timedelta
    return as_of + timedelta(seconds=horizon_sec + bar_seconds)
