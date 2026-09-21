"""
Regression tests for live-quote exit pricing (owner directive 2026-09-21).

Anchored on the real incident: on 2026-09-21 the hourly feed had no same-day
bar until mid-morning, so the engine read MSTR's Friday close of 153.9086 as
the current price against a 166.296 fill and emitted stop_loss at -7.45%.
MSTR was actually trading at 165.895, down 0.24%. The order was not placed;
these tests pin the behaviour that makes that impossible.

    python -m bot.strategy.test_quote_pricing
"""

import json

import numpy as np
import pandas as pd

from . import engine

STALE_CLOSE = 153.9086   # MSTR, Friday 2026-09-18 close
LIVE_QUOTE = 165.895     # MSTR, live at 10:33 ET on 2026-09-21
FILL = 166.296           # what we actually paid


def _bars(last_close: float, n: int = 90) -> pd.DataFrame:
    """A flat-ish series ending at last_close, long enough to clear the
    60-bar minimum so the symbol gets a signal rather than being skipped."""
    idx = pd.date_range('2026-09-01', periods=n, freq='h', tz='UTC')
    close = np.linspace(last_close * 0.98, last_close, n)
    return pd.DataFrame({'open': close, 'high': close * 1.001,
                         'low': close * 0.999, 'close': close,
                         'volume': np.full(n, 1_000_000.0)}, index=idx)


def _snapshot(quotes=None):
    cfg = json.load(open('bot/config.json'))
    snap = {
        'today_et': '2026-09-21',
        'config': cfg,
        'state': {'date_et': '2026-09-21', 'start_of_day_equity': 1428.0,
                  'peak_equity': 1428.0, 'halted_today': False,
                  'positions': {'MSTR': {'entry_date_et': '2026-09-18',
                                         'entry_reason': 'breakout_long',
                                         'entry_confidence': 0.8,
                                         'high_water_mark': FILL,
                                         'be_armed': False, 'cycles_held': 1,
                                         'reversal_streak': 0}}},
        'portfolio': {'equity': 1428.0, 'buying_power': 1.0},
        'positions': [{'symbol': 'MSTR', 'quantity': '2.145210',
                       'shares_available_for_sells': '2.145210',
                       'average_buy_price': str(FILL)}],
        'trade_history': [],
        'bars': {'MSTR': _bars(STALE_CLOSE)},
        'regime_bars': _bars(760.0, n=40),
    }
    if quotes is not None:
        snap['quotes'] = quotes
    return snap


def _exit_reasons(out):
    return {e['symbol']: e['reason'] for e in out['exits']}


def test_stale_bar_alone_would_false_stop():
    """The bug, pinned: with no quote the stale bar still drives the exit."""
    out = engine.run(_snapshot())
    assert _exit_reasons(out).get('MSTR') == 'stop_loss', \
        'expected the documented stale-bar false stop without quotes'
    print('ok  stale bar alone still produces the false stop (bug reproduced)')


def test_live_quote_prevents_false_stop():
    """The fix: a live quote is used instead, and MSTR is not sold."""
    out = engine.run(_snapshot({'MSTR': LIVE_QUOTE}))
    assert 'MSTR' not in _exit_reasons(out), \
        f'MSTR must not exit at {LIVE_QUOTE} vs {FILL}; got {out["exits"]}'
    diag = out['diagnostics']['symbols']['MSTR']
    assert diag['exit_price'] == LIVE_QUOTE
    assert diag['exit_price_source'] == 'quote'
    assert diag['bar_vs_quote_pct'] == round((STALE_CLOSE / LIVE_QUOTE - 1) * 100, 2)
    print(f'ok  live quote prevents the false stop '
          f'(bar drift {diag["bar_vs_quote_pct"]}%)')


def test_real_stop_still_fires_on_quote():
    """A genuine loss must still stop out — the guard must not mute stops."""
    out = engine.run(_snapshot({'MSTR': FILL * 0.93}))
    assert _exit_reasons(out).get('MSTR') == 'stop_loss', \
        'a real -7% loss must still stop out'
    print('ok  a genuine loss still stops out')


def test_bad_quotes_fall_back_to_bar():
    """Junk quotes are dropped rather than trusted."""
    for bad in ({'MSTR': 0}, {'MSTR': -5}, {'MSTR': None},
                {'MSTR': 'nope'}, {'MSTR': float('nan')}):
        out = engine.run(_snapshot(bad))
        diag = out['diagnostics']['symbols']['MSTR']
        assert diag['exit_price_source'] == 'bar', f'{bad} should fall back'
    assert engine.clean_quotes(None) == {}
    print('ok  malformed quotes fall back to the bar close')


def test_entry_seeds_hwm_from_quote(monkeypatched_price=340.0):
    """A new entry's high-water mark comes from the quote, not a pre-entry bar.

    Synthetic bars do not reliably clear the ensemble, so the candidate is
    injected directly into the entry branch by stubbing the signal pass. That
    keeps the assertion on the seeding line itself rather than on whether a
    made-up price series happens to trigger a buy.
    """
    quote = 337.35
    snap = _snapshot({'AAPL': quote})
    snap['portfolio']['buying_power'] = 500.0
    snap['bars']['AAPL'] = _bars(monkeypatched_price)

    real_ensemble = engine.ensemble
    engine.ensemble = lambda *a, **k: (1, 0.99)   # force a high-confidence buy
    real_filter = engine.entry_filter
    engine.entry_filter = lambda *a, **k: (True, '')
    real_ev = engine.entry_ev
    engine.entry_ev = lambda *a, **k: 0.05
    try:
        out = engine.run(snap)
    finally:
        engine.ensemble, engine.entry_filter, engine.entry_ev = \
            real_ensemble, real_filter, real_ev

    entered = [e['symbol'] for e in out['entries']]
    assert 'AAPL' in entered, f'expected an AAPL entry, got {out["entries"]}'
    hwm = out['state_updates']['positions']['AAPL']['high_water_mark']
    assert hwm == quote, (f'entry HWM must seed from the live quote {quote}, '
                          f'not the bar close {monkeypatched_price}; got {hwm}')
    print('ok  entry seeds its high-water mark from the live quote')


def test_entry_hwm_falls_back_without_quote():
    """With no quote the seeding reverts to the bar close, as before."""
    snap = _snapshot()
    snap['portfolio']['buying_power'] = 500.0
    snap['bars']['AAPL'] = _bars(340.0)
    real_ensemble, real_filter, real_ev = \
        engine.ensemble, engine.entry_filter, engine.entry_ev
    engine.ensemble = lambda *a, **k: (1, 0.99)
    engine.entry_filter = lambda *a, **k: (True, '')
    engine.entry_ev = lambda *a, **k: 0.05
    try:
        out = engine.run(snap)
    finally:
        engine.ensemble, engine.entry_filter, engine.entry_ev = \
            real_ensemble, real_filter, real_ev
    hwm = out['state_updates']['positions']['AAPL']['high_water_mark']
    assert abs(hwm - 340.0) < 1e-6, f'expected the bar close, got {hwm}'
    print('ok  entry seeding falls back to the bar close without a quote')


if __name__ == '__main__':
    test_stale_bar_alone_would_false_stop()
    test_live_quote_prevents_false_stop()
    test_real_stop_still_fires_on_quote()
    test_bad_quotes_fall_back_to_bar()
    test_entry_seeds_hwm_from_quote()
    test_entry_hwm_falls_back_without_quote()
    print('\nall live-quote pricing tests passed')
