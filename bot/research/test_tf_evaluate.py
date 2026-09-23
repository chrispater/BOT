"""
Tests for the timeframe study's adaptive selector (2026-09-23).

    python -m bot.research.test_tf_evaluate
"""

import numpy as np
import pandas as pd

from .tf_evaluate import adaptive, score


def _rows(tf, bar_hours, times, gross, sym='AAA'):
    idx = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    n = len(idx)
    return pd.DataFrame({'sym': sym, 'tf': tf, 'bar_hours': bar_hours, 'ml_trained': True,
                         'val_folds': 4, 'val_n': 100, 'gross_both': gross,
                         'date': idx.tz_convert('America/New_York').date}, index=idx)


def test_score_is_per_hour_and_net_of_cost():
    f = _rows('1d', 24.0, ['2026-01-02'], [0.024])
    s = score(f, 0.004).iloc[0]
    assert abs(s - (0.024 - 0.004) / (5 * 24)) < 1e-12, s
    g = _rows('1h', 1.0, ['2026-01-02'], [0.003])
    assert pd.isna(score(g, 0.004).iloc[0]), 'not validated net of cost -> no score'
    print('ok  score = validated net expectancy per holding hour; unvalidated scores are empty')


def test_picks_best_known_timeframe_without_lookahead():
    # 4h is better from the start; 1d becomes better only at its 2026-01-03 retrain.
    h4 = _rows('4h', 4.0, ['2026-01-02 04:00', '2026-01-02 08:00', '2026-01-03 04:00', '2026-01-03 08:00'],
               [0.010, 0.010, 0.010, 0.010])
    d1 = _rows('1d', 24.0, ['2026-01-02 00:00', '2026-01-03 00:00'], [0.010, 0.060])
    out = adaptive({'4h': h4, '1d': d1}, 0.004)
    chosen = dict(zip(out.index, out['tf']))
    # 4h score 0.006/20=3e-4 beats 1d 0.006/120=5e-5 until 1d's 0.056/120=4.7e-4 appears.
    assert chosen[pd.Timestamp('2026-01-02 04:00', tz='UTC')] == '4h'
    assert chosen[pd.Timestamp('2026-01-02 08:00', tz='UTC')] == '4h'
    # At 00:00 only 1d's validation exists yet (4h's first row is 04:00): 1d is the choice.
    assert chosen[pd.Timestamp('2026-01-02 00:00', tz='UTC')] == '1d'
    assert chosen[pd.Timestamp('2026-01-03 00:00', tz='UTC')] == '1d'
    assert pd.Timestamp('2026-01-03 04:00', tz='UTC') not in chosen, '4h dropped once 1d leads'
    print('ok  the selector switches only when a better validation becomes known, never earlier')


def test_nothing_validated_trades_nothing():
    h4 = _rows('4h', 4.0, ['2026-01-02 04:00'], [0.001])
    d1 = _rows('1d', 24.0, ['2026-01-02 00:00'], [0.002])
    out = adaptive({'4h': h4, '1d': d1}, 0.004)
    assert out.empty, out
    print('ok  a symbol with no validated timeframe is not traded')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('\nall timeframe-evaluation tests passed')
