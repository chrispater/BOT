"""
Tests for cycle state I/O (2026-09-24). Runs in a temp dir; never touches
the real state or log.

    python -m bot.tools.test_cycle_io
"""

import json
import os
import tempfile

from . import cycle_io


def _setup(tmp):
    os.makedirs(os.path.join(tmp, 'bot'))
    json.dump({'date_et': '2026-09-23', 'start_of_day_equity': 1400.0, 'peak_equity': 1431.6,
               'halted_today': False,
               'positions': {'MSTR': {'high_water_mark': 170.18, 'cycles_held': 17}}},
              open(os.path.join(tmp, 'bot', 'state.json'), 'w'))
    open(os.path.join(tmp, 'bot', 'trade_log.jsonl'), 'w').close()


def _in(tmp, fn):
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        return fn()
    finally:
        os.chdir(cwd)


SNAP = {'today_et': '2026-09-24', 'quotes': {'MSTR': 160.0, 'COIN': 196.0},
        'positions': [{'symbol': 'MSTR', 'quantity': '2.0', 'average_buy_price': '166.30'}]}


def test_rollover_once_per_day():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        r1 = _in(tmp, lambda: cycle_io.rollover('2026-09-24', 1522.06))
        r2 = _in(tmp, lambda: cycle_io.rollover('2026-09-24', 1500.00))
        st = json.load(open(os.path.join(tmp, 'bot', 'state.json')))
        assert r1['rolled'] and not r2['rolled']
        assert st['start_of_day_equity'] == 1522.06 and st['peak_equity'] == 1522.06
    print('ok  rollover resets start-of-day and peak once per ET day, never twice')


def test_persist_entry_seeds_hwm_at_fill_and_logs():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        dec = {'exits': [], 'entries': [{'symbol': 'COIN'}], 'diagnostics': {'regime': 'sideways'},
               'state_updates': {'peak_equity': 1522.06, 'positions': {
                   'MSTR': {'high_water_mark': 170.18, 'cycles_held': 18},
                   'COIN': {'high_water_mark': 195.87, 'cycles_held': 0}}}}
        fills = {'time_et': '11:32', 'equity': 1521.96, 'buying_power': 101.0,
                 'orders': [{'symbol': 'COIN', 'side': 'buy', 'ref_id': 'r1', 'state': 'filled',
                             'quantity': '0.509869', 'avg_price': '196.1286', 'dollar_amount': '100.00'}]}
        lines = _in(tmp, lambda: cycle_io.persist(fills, dec, SNAP))
        st = json.load(open(os.path.join(tmp, 'bot', 'state.json')))
        assert st['positions']['COIN']['high_water_mark'] == 196.1286
        assert st['positions']['MSTR']['cycles_held'] == 18
        assert [l.get('event') or l.get('type') for l in lines] == ['buy', 'cycle_summary']
        assert lines[-1]['live_marks_pct']['MSTR'] == round((160 / 166.3 - 1) * 100, 2)
    print('ok  a filled entry seeds its high-water mark at the fill; engine state is merged')


def test_rejected_entry_dropped_and_sell_pnl_logged():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        dec = {'exits': [{'symbol': 'MSTR'}], 'entries': [{'symbol': 'COIN'}], 'diagnostics': {},
               'state_updates': {'peak_equity': 1500.0, 'positions': {
                   'COIN': {'high_water_mark': 195.87, 'cycles_held': 0}}}}
        fills = {'orders': [
            {'symbol': 'MSTR', 'side': 'sell', 'ref_id': 's', 'state': 'filled',
             'quantity': '2.0', 'avg_price': '158.0'},
            {'symbol': 'COIN', 'side': 'buy', 'ref_id': 'b', 'state': 'rejected'}]}
        lines = _in(tmp, lambda: cycle_io.persist(fills, dec, SNAP))
        st = json.load(open(os.path.join(tmp, 'bot', 'state.json')))
        assert 'COIN' not in st['positions'] and 'MSTR' not in st['positions']
        sell = lines[0]
        assert sell['event'] == 'sell' and sell['pnl_pct'] == round((158 / 166.3 - 1) * 100, 4)
    print('ok  a rejected entry is not recorded as a position; sells log pnl_pct from avg cost')


def test_failed_exit_keeps_position():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        dec = {'exits': [{'symbol': 'MSTR'}], 'entries': [], 'diagnostics': {},
               'state_updates': {'peak_equity': 1500.0, 'positions': {}}}
        fills = {'orders': [{'symbol': 'MSTR', 'side': 'sell', 'ref_id': 's', 'state': 'rejected'}]}
        _in(tmp, lambda: cycle_io.persist(fills, dec, SNAP))
        st = json.load(open(os.path.join(tmp, 'bot', 'state.json')))
        assert st['positions']['MSTR']['high_water_mark'] == 170.18
    print('ok  an exit that did not fill leaves the position and its state in place')


def test_refs_are_fresh_uuids():
    import uuid
    r = cycle_io.refs(3)
    assert len(set(r)) == 3 and all(str(uuid.UUID(x)) == x for x in r)
    print('ok  refs gives distinct UUID ref_ids')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('\nall cycle I/O tests passed')
