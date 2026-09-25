"""
Tests for the crypto lane's snapshot builder and engine settlement switch
(2026-09-23). No network.

    python -m bot.tools.test_crypto_lane
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd

from bot.strategy import engine
from bot.strategy.run_cycle import bars_to_df

from .build_crypto_snapshot import btc_daily, build, merged_config, sleeve

CFG = json.load(open('bot/config.json'))


def _exit_cfg():
    cfg = dict(CFG['strategy'])
    cfg.update(CFG['risk'])
    return cfg


def test_same_day_take_profit_not_deferred_for_crypto():
    cfg = _exit_cfg()
    meta = {'high_water_mark': 100.0, 'cycles_held': 0}
    ok_eq, why_eq, _ = engine.exit_decision(meta, 100.0, 111.0, 0, 0.5, cfg, held_today=True)
    assert (ok_eq, why_eq) == (False, 'take_profit'), (ok_eq, why_eq)
    cfg['instant_settlement'] = True
    held_today = True and not cfg.get('instant_settlement')
    ok_c, why_c, _ = engine.exit_decision(meta, 100.0, 111.0, 0, 0.5, cfg, held_today=held_today)
    assert (ok_c, why_c) == (True, 'take_profit'), (ok_c, why_c)
    print('ok  a same-day take profit is deferred for equities (GFV) but not for crypto')


def test_sleeve_caps_spending():
    pos = [{'symbol': 'BTC', 'quantity': '0.001', 'average_buy_price': '80000'}]
    target, spend = sleeve(1400.0, pos, {'BTC': 84000.0}, buying_power=900.0, pct=25)
    assert target == 350.0 and abs(spend - (350.0 - 84.0)) < 1e-9, (target, spend)
    _, spend2 = sleeve(1400.0, [], {}, buying_power=20.0, pct=25)
    assert spend2 == 20.0
    _, spend3 = sleeve(1400.0, [{'symbol': 'ETH', 'quantity': '1', 'average_buy_price': '400'}],
                       {'ETH': 400.0}, buying_power=500.0, pct=25)
    assert spend3 == 0.0
    print('ok  the lane can spend only up to its 25% sleeve, and never more than buying power')


def test_merged_config_isolates_universe():
    c = json.loads(json.dumps(CFG))
    c['crypto']['strategy'] = {'validation_cost_pct': 0.4}
    m = merged_config(c)
    assert m['universe'] == c['crypto']['universe'] and m['universe_expansion_enabled'] is False
    assert m['strategy']['validation_cost_pct'] == 0.4 and m['strategy']['instant_settlement'] is True
    assert CFG['strategy'].get('instant_settlement') is None, 'equity config must be untouched'
    assert engine.active_universe(m) == set(c['crypto']['universe'])
    print('ok  crypto overrides apply to the crypto lane only; equity config untouched')


def _hourly(n=1600, seed=1, start='2026-07-01'):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq='h', tz='UTC')
    close = 80000 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return [{'begins_at': t.strftime('%Y-%m-%dT%H:%M:%SZ'), 'open_price': str(c * 0.999),
             'high_price': str(c * 1.003), 'low_price': str(c * 0.997), 'close_price': str(c),
             'volume': float(rng.uniform(50, 150)), 'session': '24_7'} for t, c in zip(idx, close)]


def test_btc_regime_is_daily_and_excludes_today():
    from bot.strategy.bars import clean_rows
    df = clean_rows(_hourly(n=24 * 5 + 7))
    d = btc_daily(df)
    assert len(d) == 5 and d.index[-1] < df.index[-1].normalize(), d.index
    print('ok  BTC regime bars are daily and leave out the unfinished day')


def test_end_to_end_engine_run():
    with tempfile.TemporaryDirectory() as tmp:
        hist = os.path.join(tmp, 'h.json')
        json.dump({'data': {'results': [{'symbol': s, 'bars': _hourly(seed=i)}
                                        for i, s in enumerate(['BTC', 'ETH'])]}}, open(hist, 'w'))
        spec = {'today_et': '2026-09-06', 'history_file': hist, 'total_equity': 1400.0,
                'buying_power': 600.0, 'positions': [], 'quotes': {}}
        cfg = json.loads(json.dumps(CFG))
        cfg['crypto']['universe'] = ['BTC', 'ETH']
        snap, rep = build(spec, cfg, {}, [])
        assert rep['sleeve_equity'] == 350.0 and rep['spendable'] == 350.0, rep
        snap['bars'] = {s: bars_to_df(r) for s, r in snap['bars'].items()}
        snap['regime_bars'] = bars_to_df(snap['regime_bars'])
        dec = engine.run(snap)
        spent = sum(float(e['dollar_amount']) for e in dec['entries'])
        assert spent <= 350.0 + 1e-6, spent
        assert all(e['symbol'] in ('BTC', 'ETH') for e in dec['entries'])
        assert set(dec['diagnostics']['symbols']) == {'BTC', 'ETH'}
    print(f'ok  the unchanged engine runs a crypto snapshot end to end ({len(dec["entries"])} entries, within sleeve)')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('\nall crypto lane tests passed')
