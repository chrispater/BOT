"""
Point-in-time signal replay at any bar timeframe (timeframe study, 2026-09-23).

The same stage-A replay as precompute.py (same engine model, features,
walk-forward validation and setups, imported from bot/strategy), generalised
from "hourly bars, 60 calendar days of history, retrain daily" to:

  * any bar length (--bar-hours) and a label threshold fixed per timeframe
    by the pre-registered rule in timeframes/PLAN.md,
  * a training window measured in BARS (--history),
  * retraining every K bars (--retrain) instead of every day.

Rows are stamped at the bar's END, the moment a live cycle could act on it,
so signals from different timeframes interleave correctly in one simulated
portfolio. The fill is the next bar's open; for equity daily bars that is the
next session's 9:30 ET open.

    python -m bot.research.tf_precompute <bars.pkl> <outdir> --bar-hours 24 \
        --thresh 0.0196 --history 500 --retrain 5 --test-start 2025-09-23 \
        [--equity-daily] [--workers 4]
"""

import argparse
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd

from bot.strategy import engine
from bot.strategy.indicators import calculate_indicators
from bot.strategy.setups import detect_setups

from .precompute import FORWARD, SL, RecordingMLStream, _val_stats

ET = 'America/New_York'


def bar_times(index: pd.DatetimeIndex, bar_hours: float, equity_daily: bool):
    """(decision time = bar end, fill time of the NEXT bar) for each bar."""
    if equity_daily:
        dates = index.tz_convert('UTC').normalize().tz_localize(None)
        close = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(ET) + pd.Timedelta(hours=16) for d in dates]).tz_convert('UTC')
        opens = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(ET) + pd.Timedelta(hours=9, minutes=30) for d in dates]).tz_convert('UTC')
        next_fill = pd.Series(opens, index=index).shift(-1)
        return close, next_fill
    end = index + pd.Timedelta(hours=bar_hours)
    next_fill = pd.Series(index, index=index).shift(-1)
    return end, next_fill


def process_symbol(args):
    sym, df, out_path, p = args
    if os.path.exists(out_path):
        return sym, 'cached'
    df = df[df['volume'] > 0].sort_index()
    ends, next_fill = bar_times(df.index, p['bar_hours'], p['equity_daily'])
    next_open = df['open'].shift(-1)
    test_start = pd.Timestamp(p['test_start'], tz='UTC')
    first = int(np.searchsorted(ends, test_start))
    first = max(first, p['history'])
    rows = []
    for b0 in range(first, len(df), p['retrain']):
        b1 = min(b0 + p['retrain'], len(df))
        sub = df.iloc[b0 - p['history']:b1]
        n_train = p['history']
        try:
            ind = calculate_indicators(sub)
        except Exception:
            continue
        ml = RecordingMLStream(FORWARD, p['thresh'], SL)
        try:
            ml.train(ind.iloc[:n_train])
        except Exception:
            ml.trained = False
        vs = _val_stats(ml)
        feats = list(ml.model.feature_names_in_) if (ml.trained and ml.model is not None
                                                     and hasattr(ml.model, 'feature_names_in_')) else None
        for k in range(n_train, len(ind)):
            view = ind.iloc[:k + 1]
            i = b0 + (k - n_train)
            ml_cls, ml_prob = 0, 0.5
            if ml.trained and ml.model is not None:
                try:
                    X = view[feats].iloc[[-1]] if feats else view.iloc[[-1]]
                    proba = ml.model.predict_proba(X)[0]
                    best = int(np.argmax(proba))
                    ml_cls, ml_prob = int(ml.model.classes_[best]), float(proba[best])
                except Exception:
                    pass
            se_sig, se_conf, _ = engine.indicator_score(view)
            fired = tuple((s['signal'], s['confidence'], s['name']) for s in detect_setups(view))
            row = view.iloc[-1]
            end = ends[i]
            rows.append({
                'ts': end, 'date': end.tz_convert(ET).date(), 'close': float(row['close']),
                'next_open': float(next_open.iloc[i]) if pd.notna(next_open.iloc[i]) else np.nan,
                'next_ts': next_fill.iloc[i],
                'ml_trained': bool(ml.trained), 'ml_cls': ml_cls, 'ml_prob': ml_prob,
                'se_sig': int(se_sig), 'se_conf': float(se_conf), 'fired': fired,
                'adx': float(row['adx']) if pd.notna(row.get('adx')) else np.nan,
                'vr': float(row['volume_ratio']) if pd.notna(row.get('volume_ratio')) else np.nan,
                'bar_hours': p['bar_hours'],
                **vs,
            })
    out = pd.DataFrame(rows).set_index('ts') if rows else pd.DataFrame()
    out.to_pickle(out_path)
    return sym, len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bars'), ap.add_argument('outdir')
    ap.add_argument('--bar-hours', type=float, required=True)
    ap.add_argument('--thresh', type=float, required=True)
    ap.add_argument('--history', type=int, required=True)
    ap.add_argument('--retrain', type=int, required=True)
    ap.add_argument('--test-start', required=True)
    ap.add_argument('--equity-daily', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    p = {'bar_hours': a.bar_hours, 'thresh': a.thresh, 'history': a.history,
         'retrain': a.retrain, 'test_start': a.test_start, 'equity_daily': a.equity_daily}
    os.makedirs(a.outdir, exist_ok=True)
    bars = pickle.load(open(a.bars, 'rb'))
    jobs = [(s, df, os.path.join(a.outdir, f'{s}.pkl'), p) for s, df in sorted(bars.items())]
    with Pool(a.workers) as pool:
        for sym, n in pool.imap_unordered(process_symbol, jobs):
            print(f'{sym:6} {n}', flush=True)


if __name__ == '__main__':
    main()
