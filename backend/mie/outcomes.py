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


def resolve_distinct_horizons(candidate_horizons: List[int], bar_seconds: float) -> List[int]:
    """
    Collapse a candidate horizon list to the ones that are actually
    distinguishable at this bar interval.

    `bars_for_horizon` floors every horizon to a whole number of bars, so on a
    5-minute timeframe, 30s/1m/3m/5m all round to "1 bar ahead" — training a
    separate EdgeModel for each produces four models fit on byte-identical
    labels (same feature rows, same forward-return column), which look like a
    bug because they report byte-identical validation stats. They're not
    wrong, they're just the same model wearing four different name tags.

    Returns one horizon per distinct bar count, keeping whichever candidate in
    that group is closest to `bar_count * bar_seconds` — the label that most
    honestly describes what's actually being predicted — while guaranteeing
    the returned value is always one of the ORIGINAL candidates, so its
    ret_/mfe_/mae_ outcome columns are guaranteed to already exist (outcomes
    are resolved for the full original candidate list; nothing here invents a
    horizon whose columns were never computed).
    """
    groups: Dict[int, List[int]] = {}
    for h in candidate_horizons:
        k = bars_for_horizon(h, bar_seconds)
        groups.setdefault(k, []).append(h)

    survivors = []
    for k, group in groups.items():
        target = k * bar_seconds
        survivors.append(min(group, key=lambda h: abs(h - target)))
    return sorted(survivors)


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
