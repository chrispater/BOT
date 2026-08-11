import numpy as np
import pandas as pd

from backend.mie.outcomes import resolve_outcomes, bars_for_horizon


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
