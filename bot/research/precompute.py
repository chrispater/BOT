"""
Stage A of the backtest: point-in-time signal replay.

For every symbol and every trading day D in the out-of-sample window, this
retrains the live engine's own model exactly as a cycle on D would see it —
on the trailing `history_days` of bars that had CLOSED before D's session —
and records, for every bar of D, the raw inputs the engine's decision logic
consumes. Nothing here decides a trade; stage B (simulate.py) does that, so
one expensive pass can serve every variant.

Faithfulness to live, and the deliberate approximations:
  * Model, labels, features, walk-forward splits, setups and the indicator
    vote are imported from bot/strategy — not reimplemented.
  * Live retrains every hourly cycle; this retrains once per day on bars
    through the previous close. Up to ~6 hours of model staleness intraday.
  * Indicators are computed over [window start, end of D] with the same
    60-day warm-up the live fetch has, so EMA/rolling warm-up matches live.
  * The validation pass records the per-bar OOS predictions and forward
    returns (not just the gross mean) so stage B can evaluate cost-aware,
    long-only and short-side variants of the gate without refitting.

No look-ahead: every quantity stored for a bar uses only bars at or before
it, and the model used on D never saw D. Labels need `forward_periods` bars
of future, and MLStream.train already drops those rows.

    python -m bot.research.precompute <bars.pkl> <outdir> [workers]
"""

import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from bot.strategy import engine
from bot.strategy.indicators import calculate_indicators
from bot.strategy.ml_model import MLStream, purged_walk_forward_splits
from bot.strategy.setups import detect_setups

HISTORY_DAYS = 60
FORWARD, THRESH, SL = 5, 0.004, 0.05


class RecordingMLStream(MLStream):
    """MLStream whose walk-forward pass also keeps the raw OOS arrays.

    The verdict it sets on self.validated is the live rule, unchanged; the
    extra arrays exist only so alternative gates can be scored offline.
    """

    def _validate_walk_forward(self, df, feats):
        preds_all, fut_all, folds_run = [], [], 0
        for train_idx, test_idx in purged_walk_forward_splits(len(df), embargo=self.forward_periods):
            train_sub = df.iloc[train_idx]
            mask = train_sub['signal'] != 0
            X_full = train_sub.loc[mask, feats]
            y = train_sub.loc[mask, 'signal'].astype(int)
            if len(X_full) < 20 or y.nunique() < 2:
                continue
            ff = [c for c in feats if X_full[c].notna().any()]
            test_sub = df.iloc[test_idx]
            if test_sub.empty:
                continue
            try:
                m = self._make_model()
                m.fit(X_full[ff], y)
                preds = m.predict(test_sub[ff])
            except Exception:
                continue
            fut = test_sub['future_return'].to_numpy()
            ok = ~np.isnan(fut)
            if not ok.any():
                continue
            preds_all.extend(preds[ok].tolist())
            fut_all.extend(fut[ok].tolist())
            folds_run += 1
        self.oos_preds = np.array(preds_all, dtype=float)
        self.oos_fut = np.array(fut_all, dtype=float)
        self.oos_folds = folds_run
        n = len(preds_all)
        if folds_run < self.MIN_FOLDS or n < self.MIN_POOLED_TRADES:
            return False, 'insufficient'
        gross = float(np.mean(self.oos_preds * self.oos_fut))
        return gross > 0, f'gross {gross:.4f}'


def _val_stats(ml):
    p = getattr(ml, 'oos_preds', np.array([]))
    f = getattr(ml, 'oos_fut', np.array([]))
    long_m, short_m = p == 1, p == -1
    return {
        'val_folds': getattr(ml, 'oos_folds', 0),
        'val_n': int(len(p)),
        'gross_both': float(np.mean(p * f)) if len(p) else np.nan,
        'long_gross': float(np.mean(f[long_m])) if long_m.any() else np.nan,
        'n_long': int(long_m.sum()),
        'short_gross': float(np.mean(-f[short_m])) if short_m.any() else np.nan,
        'n_short': int(short_m.sum()),
    }


def process_symbol(args):
    sym, df, out_path = args
    if os.path.exists(out_path):
        return sym, 'cached'
    et_dates = df.index.tz_convert('America/New_York').date
    days = sorted(set(et_dates))
    first_ts = df.index[0]
    # Next bar's open, for fills — the next bar in the symbol's own series,
    # which for the last bar of a day is the following morning's first bar.
    next_open = df['open'].shift(-1)
    next_ts = pd.Series(df.index, index=df.index).shift(-1)

    rows = []
    for d in days:
        day_mask = et_dates == d
        day_idx = np.flatnonzero(day_mask)
        d_start = df.index[day_idx[0]]
        if d_start - pd.Timedelta(days=HISTORY_DAYS) < first_ts:
            continue  # not enough history yet for a live-equivalent window
        win_mask = (df.index >= d_start - pd.Timedelta(days=HISTORY_DAYS)) & (df.index < d_start)
        n_train = int(win_mask.sum())
        if n_train < 60:
            continue
        sub = pd.concat([df[win_mask], df[day_mask]])
        try:
            ind = calculate_indicators(sub)
        except Exception:
            continue
        ml = RecordingMLStream(FORWARD, THRESH, SL)
        try:
            ml.train(ind.iloc[:n_train])
        except Exception:
            ml.trained = False
        vs = _val_stats(ml)
        feats = [c for c in ml.model.feature_names_in_] if (ml.trained and ml.model is not None
                                                             and hasattr(ml.model, 'feature_names_in_')) else None
        for k in range(n_train, len(ind)):
            view = ind.iloc[:k + 1]
            ts = ind.index[k]
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
            rows.append({
                'ts': ts, 'date': d, 'close': float(row['close']),
                'next_open': float(next_open.get(ts, np.nan)),
                'next_ts': next_ts.get(ts, pd.NaT),
                'ml_trained': bool(ml.trained), 'ml_cls': ml_cls, 'ml_prob': ml_prob,
                'se_sig': int(se_sig), 'se_conf': float(se_conf), 'fired': fired,
                'adx': float(row['adx']) if pd.notna(row.get('adx')) else np.nan,
                'vr': float(row['volume_ratio']) if pd.notna(row.get('volume_ratio')) else np.nan,
                **vs,
            })
    out = pd.DataFrame(rows).set_index('ts') if rows else pd.DataFrame()
    out.to_pickle(out_path)
    return sym, len(out)


def main():
    bars_path, outdir = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(outdir, exist_ok=True)
    bars = pickle.load(open(bars_path, 'rb'))
    jobs = [(s, df, os.path.join(outdir, f'{s}.pkl')) for s, df in sorted(bars.items())]
    with Pool(workers) as pool:
        for sym, n in pool.imap_unordered(process_symbol, jobs):
            print(f'{sym:6} {n}', flush=True)


if __name__ == '__main__':
    main()
