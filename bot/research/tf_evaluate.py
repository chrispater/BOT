"""
Stage B of the timeframe study (see timeframes/PLAN.md): fixed timeframe vs
point-in-time adaptive per-symbol timeframe vs hindsight upper bound, in one
simulated portfolio per asset class, through the unchanged simulator.

    python -m bot.research.tf_evaluate equity|crypto
"""

import sys

import numpy as np
import pandas as pd

from .experiments import EXPANDED
from .simulate import Params, load_signals, prepare, project, run

SP = '/tmp/claude-0/-home-user-BOT/59c0a99d-d529-57f6-bcb4-f09f32cd2414/scratchpad'
CRYPTO = tuple('BTC ETH SOL XRP DOGE LTC BCH LINK AVAX ADA DOT UNI XLM SHIB ATOM'.split())

SETUPS = {
    # name: (sig dir, bar hours, shift ts to bar end by hours) — the pre-change
    # hourly replays were stamped at bar START; the tf_precompute ones at END.
    'equity': {'universe': tuple(EXPANDED), 'cost_rt': 0.002, 'slip': 0.001, 'equity': 1441.11,
               'account': 'cash',
               'tfs': {'1h': (f'{SP}/bt/sig', 1.0, 1.0),
                       '2h': (f'{SP}/tf/sig_eq_2h', 2.0, 0.0),
                       '1d': (f'{SP}/tf/sig_eq_1d', 6.5, 0.0)}},
    'crypto': {'universe': CRYPTO, 'cost_rt': 0.004, 'slip': 0.002, 'equity': 355.0,
               'account': 'crypto',
               'tfs': {'1h': (f'{SP}/bt/sigc', 1.0, 1.0),
                       '4h': (f'{SP}/tf/sig_c_4h', 4.0, 0.0),
                       '1d': (f'{SP}/tf/sig_c_1d', 24.0, 0.0)}},
}


def load_tf(sigdir, universe, bar_hours, shift_h, tf):
    s = load_signals(sigdir, universe)
    if shift_h:
        s.index = s.index + pd.Timedelta(hours=shift_h)
    s['bar_hours'] = bar_hours
    s['tf'] = tf
    return s


def score(frame, cost_rt):
    """Validated net OOS expectancy per hour of holding horizon (PLAN.md 1b)."""
    ok = frame['ml_trained'] & (frame['val_folds'] >= 2) & (frame['val_n'] >= 40)
    net = frame['gross_both'] - cost_rt
    sc = net / (5.0 * frame['bar_hours'])
    return sc.where(ok & (net > 0))


def adaptive(frames: dict, cost_rt: float) -> pd.DataFrame:
    """Keep each symbol's rows only from the timeframe chosen point-in-time.

    At a row's decision time t, every timeframe's latest validation (from its
    most recent retrain at or before t) is compared; the symbol trades the
    best validated one, or nothing.
    """
    allrows = pd.concat(frames.values()).sort_index(kind='stable')
    keep = []
    for sym, g in allrows.groupby('sym'):
        g = g.copy()
        g['score'] = score(g, cost_rt)
        cols = {}
        for tf in frames:
            s = g['score'].where(g['tf'] == tf)
            # a timeframe's score is known from its own rows; carry it forward in time
            known = g['tf'] == tf
            cols[tf] = s.where(known).groupby(level=0).last().reindex(g.index.unique()).ffill()
        sc = pd.DataFrame(cols)
        best = sc.fillna(-np.inf).idxmax(axis=1).where(sc.notna().any(axis=1))
        chosen = best.reindex(g.index)
        keep.append(g[g['tf'].to_numpy() == chosen.to_numpy()])
    return pd.concat(keep).sort_index(kind='stable') if keep else allrows.iloc[:0]


def evaluate(prep, universe, setup, start, end, label):
    base = dict(universe=universe, gate='net', val_cost=setup['cost_rt'], entry_gate='validated',
                expectancy_block=True, expectancy_expiry_days=3, account=setup['account'],
                slippage=setup['slip'], start_equity=setup['equity'], start=start, end=end)
    days = pd.date_range(start, end, freq='D')
    mid = str(days[len(days) // 2].date())
    f = run(prep, Params(label, **base))
    a = run(prep, Params(label, **dict(base, end=mid)))
    b = run(prep, Params(label, **dict(base, start=mid)))
    c = run(prep, Params(label, **dict(base, slippage=setup['slip'] * 2)))
    pj = project(f['daily_returns']) if len(f['daily_returns']) > 10 else None
    return {'variant': label, 'ret': f['ret'], 'H1': a['ret'], 'H2': b['ret'], 'x2cost': c['ret'],
            'maxdd': f['max_dd'], 'trades': f['trades'], 'win': f['win_rate'],
            'p_loss70': pj['p_loss'] if pj else np.nan, 'trades_list': f.get('trade_list')}


def fmt(r):
    w = f"{r['win'] * 100:4.0f}%" if r['win'] is not None and not pd.isna(r['win']) else '   —'
    return (f"{r['variant']:34}{r['ret']:+8.1%}{r['H1']:+8.1%}{r['H2']:+8.1%}{r['x2cost']:+9.1%}"
            f"{r['maxdd']:+8.1%}{r['trades']:7}{w:>6}{r['p_loss70']:9.0%}")


def main():
    cls = sys.argv[1]
    st = SETUPS[cls]
    U = st['universe']
    frames = {}
    for tf, (d, bh, sh) in st['tfs'].items():
        try:
            frames[tf] = load_tf(d, U, bh, sh, tf)
        except ValueError:
            print(f'{tf}: no signals'); continue
    spans = {tf: (f['date'].min(), f['date'].max(), f['sym'].nunique()) for tf, f in frames.items()}
    for tf, (a, b, n) in spans.items():
        print(f'{tf}: {a} -> {b}, {n} symbols')
    start = str(max(v[0] for v in spans.values()))
    end = str(min(v[1] for v in spans.values()))
    print(f'\ncommon window {start} -> {end}\n')
    hdr = f"{'variant':34}{'return':>8}{'H1':>8}{'H2':>8}{'@2xcost':>9}{'maxDD':>8}{'trades':>7}{'win':>6}{'P(loss70)':>9}"
    print(hdr)
    results = {}
    preps = {tf: prepare(f) for tf, f in frames.items()}
    for tf in frames:
        results[tf] = evaluate(preps[tf], U, st, start, end, f'fixed {tf}')
        print(fmt(results[tf]))
    ad = adaptive(frames, st['cost_rt'])
    results['adaptive'] = evaluate(prepare(ad), U, st, start, end, 'adaptive per-symbol (point-in-time)')
    print(fmt(results['adaptive']))
    share = ad[(ad['date'] >= pd.Timestamp(start).date())].groupby('tf')['sym'].size()
    print('   adaptive rows by timeframe:', share.to_dict())

    # Hindsight upper bound: each symbol on the timeframe whose own single-symbol
    # run made the most over the window. Uses the future; never adoptable.
    best = {}
    for sym in U:
        top, arg = -np.inf, None
        for tf, f in frames.items():
            fs = f[f['sym'] == sym]
            if fs.empty:
                continue
            r = run(prepare(fs), Params('hs', universe=(sym,), gate='net', val_cost=st['cost_rt'],
                                          entry_gate='validated', expectancy_block=False,
                                          account=st['account'], slippage=st['slip'],
                                          start_equity=st['equity'], start=start, end=end))
            if r['ret'] > top:
                top, arg = r['ret'], tf
        if arg:
            best[sym] = arg
    hs = pd.concat([frames[tf][frames[tf]['sym'] == s] for s, tf in best.items()]).sort_index(kind='stable')
    r = evaluate(prepare(hs), U, st, start, end, 'hindsight best TF (upper bound)')
    print(fmt(r))
    print('   hindsight picks:', pd.Series(best).value_counts().to_dict())

    # Longest honest window for each fixed timeframe on its own.
    print('\nEach fixed timeframe over its own full out-of-sample window:')
    print(hdr)
    for tf, f in frames.items():
        a, b, _ = spans[tf]
        print(fmt(evaluate(preps[tf], U, st, str(a), str(b), f'fixed {tf} ({a}..{b})')))


if __name__ == '__main__':
    main()
