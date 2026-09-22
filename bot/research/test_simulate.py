"""
Sanity tests for the backtest simulator — the things that, if wrong, would
make every number in the report confidently false.

    python -m bot.research.test_simulate
"""

import datetime as dt
import random

import pandas as pd

from bot.research import simulate as S
from bot.strategy import engine, setups


def _row(sym, ts, close, nxt, ml=(0, 0.5), se=(0, 0.0), fired=(), adx=30.0, vr=1.0,
         gross=0.01, lg=0.01, sg=0.01, day=None, fill=None):
    return {'sym': sym, 'close': close, 'next_open': nxt, 'ml_trained': True,
            'ml_cls': ml[0], 'ml_prob': ml[1], 'se_sig': se[0], 'se_conf': se[1],
            'fired': fired, 'adx': adx, 'vr': vr, 'val_folds': 4, 'val_n': 180,
            'gross_both': gross, 'long_gross': lg, 'n_long': 90, 'short_gross': sg,
            'n_short': 90, 'date': day, 'fill_date': fill}


def _steps(series, sym='AAA', **kw):
    """series: list of (day, close, next_open, ml) -> prepared steps."""
    out = []
    t0 = pd.Timestamp('2026-03-02 15:00', tz='UTC')
    for i, (day, fill, c, n, ml) in enumerate(series):
        ts = t0 + pd.Timedelta(hours=i)
        out.append((ts, fill, day, [_row(sym, ts, c, n, ml=ml, day=day, fill=fill, **kw)]))
    return out


D1, D2, D3 = dt.date(2026, 3, 2), dt.date(2026, 3, 3), dt.date(2026, 3, 4)


def test_resolve_setups_matches_engine():
    """The precomputed-setup resolver must equal setups.best_setup exactly."""
    rng = random.Random(1)
    names = ['rsi_bounce', 'breakout_long', 'rsi_fade', 'breakout_down']
    real = setups.detect_setups
    try:
        for _ in range(500):
            fired = [{'signal': rng.choice([1, -1]), 'confidence': rng.uniform(0.7, 0.85),
                      'name': rng.choice(names)} for _ in range(rng.randint(0, 3))]
            ml = (rng.choice([-1, 0, 1]), rng.uniform(0.4, 0.99))
            setups.detect_setups = lambda df, f=fired: f
            want = setups.best_setup(None, *ml)
            got = S.resolve_setups(tuple((f['signal'], f['confidence'], f['name']) for f in fired), *ml)
            assert want[0] == got[0] and abs(want[1] - got[1]) < 1e-12 and want[2] == got[2], (want, got)
    finally:
        setups.detect_setups = real
    print('ok  resolve_setups == best_setup on 500 random cases')


def test_short_mirror_maps_pnl_and_stops():
    """A short that loses 5.1% must stop out through the mirrored price."""
    cfg = dict(S.BASE_CFG, time_stop_hours=None)
    avg = 100.0
    meta = {'high_water_mark': avg, 'be_armed': False, 'cycles_held': 5, 'reversal_streak': 0}
    ok, reason, _ = engine.exit_decision(meta, avg, 2 * avg - 105.1, 0, 0.5, cfg, False)
    assert ok and reason == 'stop_loss', reason
    ok, reason, _ = engine.exit_decision(meta, avg, 2 * avg - 89.0, 0, 0.5, cfg, False)
    assert ok and reason == 'take_profit', reason
    print('ok  mirrored price gives short stop_loss at -5% and take_profit at +10%')


def test_no_signal_means_flat_equity():
    steps = _steps([(D1, D1, 100, 101, (0, 0.5)), (D1, D2, 101, 102, (0, 0.5)),
                    (D2, D2, 102, 99, (0, 0.5))])
    r = S.run(steps, S.Params(universe=('AAA',)))
    assert r['trades'] == 0 and abs(r['end'] - 1441.11) < 1e-9, r['end']
    print('ok  no qualifying signal -> no trades, equity unchanged')


def test_entry_fills_at_next_open_with_slippage():
    # Zero signal-to-fill gap, so the entry gap haircut does not scale the size.
    steps = _steps([(D1, D1, 102, 102, (1, 0.9)), (D1, D1, 104, 105, (0, 0.5))])
    r = S.run(steps, S.Params(universe=('AAA',), expectancy_block=False))
    # 25% of 1441.11 = 360.27 bought at 102 * 1.001; marked at the last close 104
    qty = 360.27 / (102 * 1.001)
    want = 1441.11 - 360.27 + qty * 104
    assert abs(r['end'] - want) < 0.01, (r['end'], want)
    print('ok  entry fills at next open + slippage, marked to close')


def test_gap_haircut_shrinks_entry():
    """A 2% gap between signal close and fill must shrink the entry, as live."""
    steps = _steps([(D1, D1, 100, 102, (1, 0.9)), (D1, D1, 102, 102, (0, 0.5))])
    r = S.run(steps, S.Params(universe=('AAA',), expectancy_block=False))
    no_gap = S.run(_steps([(D1, D1, 102, 102, (1, 0.9)), (D1, D1, 102, 102, (0, 0.5))]),
                   S.Params(universe=('AAA',), expectancy_block=False))
    # Slippage cost scales with size: the haircut entry must lose less to it.
    assert (1441.11 - r['end']) < (1441.11 - no_gap['end']), (r['end'], no_gap['end'])
    print('ok  signal-to-fill gap haircut shrinks the entry, as in live')


def test_cash_proceeds_do_not_recycle_same_day():
    """Cash account: a sale's proceeds must not fund a same-day buy."""
    s = []
    t0 = pd.Timestamp('2026-03-02 15:00', tz='UTC')
    # Day 1: buy AAA. Day 2 bar 1: AAA stop-loss fires; BBB signals a buy.
    s.append((t0, D1, D1, [_row('AAA', t0, 100, 100, ml=(1, 0.9), day=D1, fill=D1)]))
    t1 = t0 + pd.Timedelta(days=1)
    s.append((t1, D2, D2, [_row('AAA', t1, 90, 90, ml=(0, 0.5), day=D2, fill=D2),
                           _row('BBB', t1, 50, 50, ml=(1, 0.9), day=D2, fill=D2)]))
    p = S.Params(universe=('AAA', 'BBB'), expectancy_block=False, max_pos_pct=100)
    cash = S.run(s, p)
    marg = S.run(s, S.Params(universe=('AAA', 'BBB'), expectancy_block=False,
                             max_pos_pct=100, account='margin'))
    buys_cash = cash['trades_df']
    assert cash['open_at_end'] == 0, 'cash account must not buy BBB with unsettled proceeds'
    assert marg['open_at_end'] == 1, 'margin account should redeploy proceeds immediately'
    print('ok  T+1: cash proceeds idle until next day; margin redeploys immediately')


def test_pdt_caps_same_day_round_trips():
    """Margin under $25k: a 4th same-day discretionary exit in 5 days is refused."""
    s = []
    t = pd.Timestamp('2026-03-02 15:00', tz='UTC')
    days = [dt.date(2026, 3, 2 + i) for i in range(4)]
    for d in days:
        # buy at bar 1, then a take-profit-sized jump at bar 2 the same day
        s.append((t, d, d, [_row('AAA', t, 100, 100, ml=(1, 0.9), day=d, fill=d)]))
        t += pd.Timedelta(hours=1)
        s.append((t, d, d, [_row('AAA', t, 111, 111, ml=(0, 0.5), day=d, fill=d)]))
        t += pd.Timedelta(hours=23)
    r = S.run(s, S.Params(universe=('AAA',), expectancy_block=False, account='margin'))
    same_day = r['trades_df'].query('entry_date == exit_date')
    assert len(same_day) == 3, f'expected 3 day trades, got {len(same_day)}'
    print('ok  PDT: only 3 same-day round trips allowed in 5 days')


if __name__ == '__main__':
    test_resolve_setups_matches_engine()
    test_short_mirror_maps_pnl_and_stops()
    test_no_signal_means_flat_equity()
    test_entry_fills_at_next_open_with_slippage()
    test_gap_haircut_shrinks_entry()
    test_cash_proceeds_do_not_recycle_same_day()
    test_pdt_caps_same_day_round_trips()
    print('\nall simulator tests passed')
