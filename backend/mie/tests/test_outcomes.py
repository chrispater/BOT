import numpy as np
import pandas as pd

from backend.mie.outcomes import resolve_outcomes, bars_for_horizon, resolve_distinct_horizons
from backend.mie.state import HORIZONS_SEC


def test_bars_for_horizon_rounds_and_floors_at_one():
    assert bars_for_horizon(300, 300) == 1
    assert bars_for_horizon(900, 300) == 3
    assert bars_for_horizon(30, 300) == 1   # sub-bar horizon still resolves to 1 bar, not 0


def test_resolve_outcomes_matches_hand_computation():
    # Deterministic small series so ret/mfe/mae can be checked by hand.
    idx = pd.date_range('2026-01-01', periods=6, freq='5min', tz='UTC')
    df = pd.DataFrame({
        'open':  [100, 101, 102, 99, 98, 97],
        'high':  [101, 103, 104, 100, 99, 98],
        'low':   [99, 100, 101, 97, 96, 95],
        'close': [100, 102, 103, 98, 97, 96],
        'volume': [1] * 6,
    }, index=idx)

    out = resolve_outcomes(df, bar_seconds=300, horizons=[300, 600])
    # horizon=300s -> k=1 bar ahead
    assert np.isclose(out['ret_5m'].iloc[0], 102 / 100 - 1)
    assert np.isclose(out['mfe_5m'].iloc[0], 103 / 100 - 1)   # high of bar 1
    assert np.isclose(out['mae_5m'].iloc[0], 100 / 100 - 1)   # low of bar 1
    # horizon=600s -> k=2 bars ahead
    assert np.isclose(out['ret_10m'].iloc[0], 103 / 100 - 1)
    assert np.isclose(out['mfe_10m'].iloc[0], max(103, 104) / 100 - 1)
    assert np.isclose(out['mae_10m'].iloc[0], min(100, 101) / 100 - 1)


def test_tail_rows_unresolved_not_dropped():
    idx = pd.date_range('2026-01-01', periods=5, freq='5min', tz='UTC')
    df = pd.DataFrame({
        'open': [100] * 5, 'high': [101] * 5, 'low': [99] * 5,
        'close': [100] * 5, 'volume': [1] * 5,
    }, index=idx)
    out = resolve_outcomes(df, bar_seconds=300, horizons=[900])   # k=3 bars
    assert len(out) == 5
    assert out['ret_15m'].iloc[-1] != out['ret_15m'].iloc[-1]   # NaN — not yet resolvable
    assert out['ret_15m'].iloc[0] == out['ret_15m'].iloc[0]     # resolvable — not NaN


def test_resolve_distinct_horizons_collapses_aliases_on_coarse_timeframe():
    """
    The bug this guards against: on a 5-minute bot, HORIZONS_SEC's 30s/1m/3m/5m
    all round to "1 bar ahead" via bars_for_horizon, so fitting one EdgeModel
    per nominal horizon silently fits the same model four times over — same
    features, same forward-return column, byte-identical validation reports —
    which reads as a bug (and was reported as one) rather than working as
    intended. Distinct bar counts must collapse to one survivor each.
    """
    distinct = resolve_distinct_horizons(HORIZONS_SEC, bar_seconds=300)
    assert distinct == [300, 900]   # 30s/1m/3m/5m alias to 1 bar; 900s is 3 bars


def test_resolve_distinct_horizons_survivor_is_a_real_candidate():
    """
    The surviving horizon for an aliased group must always be one of the
    ORIGINAL candidates (never an invented value like bar_count*bar_seconds)
    — its ret_/mfe_/mae_ outcome columns are only guaranteed to exist for
    horizons that were actually in the resolved-outcomes candidate list.
    """
    for bar_seconds in (60, 180, 300, 900):
        for h in resolve_distinct_horizons(HORIZONS_SEC, bar_seconds):
            assert h in HORIZONS_SEC


def test_resolve_distinct_horizons_prefers_closest_label():
    """Among an aliased group, the survivor should be whichever candidate's
    wall-clock value most accurately describes the bar count actually being
    predicted — not just the first or last in the group."""
    # bar_seconds=60: 30s and 60s both round to 1 bar; 60 is the more honest
    # label for "1 bar ahead of a 1-minute candle" than 30.
    distinct = resolve_distinct_horizons([30, 60], bar_seconds=60)
    assert distinct == [60]


def test_resolve_distinct_horizons_no_aliasing_on_fine_timeframe():
    """On a timeframe finer than every configured horizon, nothing should
    collapse — this is the common case and must stay a no-op."""
    assert resolve_distinct_horizons(HORIZONS_SEC, bar_seconds=30) == sorted(HORIZONS_SEC)
