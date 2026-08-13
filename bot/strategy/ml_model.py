"""
ML signal stream, ported from CryptoQuantScanner backend/trading_service.py
(create_labels / train_model / predict_signal). Per-symbol model, retrained
fresh each cycle from the fetched bars — no persistence, no drift.

Uses HistGradientBoostingClassifier: NaN-native (no imputer/scaler),
well-calibrated probabilities, pure sklearn.

Validation gate ported from the claude/review-trading-bot-ZMKDp branch's
Market Intelligence Engine (backend/mie/validation.py): purged walk-forward
cross-validation, so a model only drives entries/exits after it has shown a
positive-expectancy edge on data it wasn't fit on. The rest of that engine
(persistent observation store, live order-book cost model, nearest-neighbor
corroboration) needs infrastructure this bot doesn't have and was left out —
this bot is stateless between hourly cycles and only has OHLCV bars, not a
continuous tape. Narrows entries/exits only, same as the original: a symbol
that fails validation degrades to no-signal, it never forces a trade.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .indicators import FEATURE_COLUMNS


def purged_walk_forward_splits(n: int, n_splits: int = 4, embargo: int = 5,
                               min_train: int = 100):
    """
    Yield (train_idx, test_idx) over a chronologically ordered range of length
    n. Each test block is a contiguous future segment; train is an expanding
    window of everything before it, truncated by `embargo` bars immediately
    preceding the test block so no label's forward-return window (which reads
    `embargo` bars into the future) leaks test-period information into
    training. Port of the crypto project's validation.py, sized down for
    ~300-bar hourly equity histories instead of a continuously-growing tape.
    """
    if n <= min_train + embargo + 1:
        return
    usable = n - min_train
    fold_size = max(1, usable // n_splits)
    for i in range(n_splits):
        test_start = min_train + i * fold_size
        test_end = n if i == n_splits - 1 else min(n, min_train + (i + 1) * fold_size)
        if test_start >= test_end:
            continue
        train_end = max(0, test_start - embargo)
        if train_end < min_train // 2:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def create_labels(df: pd.DataFrame, forward_periods: int = 5,
                  threshold: float = 0.004, sl_price: float = 0.05) -> pd.DataFrame:
    """
    "Clean signal" labeling: label LONG only if price ends up more than
    `threshold` after `forward_periods` bars AND never dips below the
    stop-loss boundary in between (the trade survives the exit rules).
    Mirrored for SHORT. Falls back to plain threshold labeling when clean
    labels are too sparse or one-sided.
    """
    sl_price = max(sl_price, 0.005)
    close = df['close'].values.astype(float)
    low = df['low'].values.astype(float)
    high = df['high'].values.astype(float)
    n = len(df)

    future_return = np.full(n, np.nan)
    future_min = np.full(n, np.nan)
    future_max = np.full(n, np.nan)
    for i in range(n - forward_periods):
        c = close[i]
        if c <= 0:
            continue
        future_return[i] = close[i + forward_periods] / c - 1
        future_min[i] = low[i + 1:i + forward_periods + 1].min() / c - 1
        future_max[i] = high[i + 1:i + forward_periods + 1].max() / c - 1

    df = df.copy()
    df['future_return'] = future_return
    df['signal'] = 0
    df.loc[(df['future_return'] > threshold) & (pd.Series(future_min, index=df.index) > -sl_price), 'signal'] = 1
    df.loc[(df['future_return'] < -threshold) & (pd.Series(future_max, index=df.index) < sl_price), 'signal'] = -1

    labeled = int((df['signal'] != 0).sum())
    n_long = int((df['signal'] == 1).sum())
    n_short = int((df['signal'] == -1).sum())
    if labeled < 50 or min(n_long, n_short) < 15:
        df['signal'] = 0
        df.loc[df['future_return'] > threshold, 'signal'] = 1
        df.loc[df['future_return'] < -threshold, 'signal'] = -1
    return df


class MLStream:
    """Train-and-predict wrapper for one symbol's bar history."""

    MIN_POOLED_TRADES = 40   # OOS bars behind the validation verdict before it's trusted
    MIN_FOLDS = 2            # require the edge to show up across >1 fold, not one lucky window

    def __init__(self, forward_periods: int = 5, threshold: float = 0.004,
                 sl_price: float = 0.05, seed: int = 42):
        self.forward_periods = forward_periods
        self.threshold = threshold
        self.sl_price = sl_price
        self.seed = seed
        self.model = None
        self.trained = False
        self.validated = False
        self.validation_summary = 'not run'

    def _make_model(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=0.1,
            class_weight='balanced', random_state=self.seed,
        )

    def train(self, df_ind: pd.DataFrame) -> bool:
        """df_ind: indicator-enriched bars (closed candles only)."""
        df = create_labels(df_ind, self.forward_periods, self.threshold, self.sl_price)
        # Last forward_periods rows have unknown outcomes — excluded via NaN label window
        df = df.iloc[:-self.forward_periods] if len(df) > self.forward_periods else df
        feats = [c for c in FEATURE_COLUMNS if c in df.columns]
        mask = df['signal'] != 0
        X = df.loc[mask, feats]
        y = df.loc[mask, 'signal'].astype(int)
        # Require enough samples and both classes, else the classifier would
        # predict one direction at bogus confidence every cycle.
        if len(X) < 30 or y.nunique() < 2:
            self.trained = False
            self.validated = False
            self.validation_summary = 'not trained (insufficient labeled samples)'
            return False
        self.model = self._make_model()
        # Thinly traded listings can leave a feature column constant, which
        # sklearn's binner rejects — degrade to untrained instead of crashing.
        try:
            self.model.fit(X, y)
        except Exception:
            self.model = None
            self.trained = False
            self.validated = False
            self.validation_summary = 'not trained (fit error)'
            return False
        self.trained = True
        self.validated, self.validation_summary = self._validate_walk_forward(df, feats)
        return True

    def _validate_walk_forward(self, df: pd.DataFrame, feats: list) -> tuple:
        """
        Purged walk-forward CV: refit the same architecture on each fold's
        expanding train window, predict on every bar of the held-out test
        block (mirroring live predict() — always a directional call, not just
        on labeled rows), and pool (predicted_direction * future_return)
        across folds. Passes only with enough pooled OOS bars, from more than
        one fold, at positive pooled expectancy.
        """
        pooled_returns = []
        folds_run = 0
        for train_idx, test_idx in purged_walk_forward_splits(len(df), embargo=self.forward_periods):
            train_sub = df.iloc[train_idx]
            train_mask = train_sub['signal'] != 0
            X_train_full = train_sub.loc[train_mask, feats]
            y_train = train_sub.loc[train_mask, 'signal'].astype(int)
            if len(X_train_full) < 20 or y_train.nunique() < 2:
                continue
            # Coverage gate (port of MIE's select_trainable_features): a short
            # early train window won't have warmup for long-lookback features
            # like ema200_distance, leaving them entirely NaN — sklearn's
            # binner rejects an all-NaN column outright. Drop those for this
            # fold only, same as the live model gets the full feature set
            # once there's enough history for every column to have real data.
            feats_fold = [c for c in feats if X_train_full[c].notna().any()]
            X_train = X_train_full[feats_fold]
            test_sub = df.iloc[test_idx]
            if test_sub.empty:
                continue
            try:
                fold_model = self._make_model()
                fold_model.fit(X_train, y_train)
                preds = fold_model.predict(test_sub[feats_fold])
            except Exception:
                continue
            fut_ret = test_sub['future_return'].to_numpy()
            valid = ~np.isnan(fut_ret)
            if not valid.any():
                continue
            pooled_returns.extend((preds[valid] * fut_ret[valid]).tolist())
            folds_run += 1

        pooled_trades = len(pooled_returns)
        if folds_run < self.MIN_FOLDS or pooled_trades < self.MIN_POOLED_TRADES:
            return False, f'not validated ({pooled_trades} OOS bars, {folds_run} folds — need {self.MIN_POOLED_TRADES}/{self.MIN_FOLDS})'
        expectancy = float(np.mean(pooled_returns))
        if expectancy <= 0:
            return False, f'not validated (negative OOS expectancy {expectancy:.4f} over {pooled_trades} bars, {folds_run} folds)'
        return True, f'validated (expectancy {expectancy:.4f} over {pooled_trades} bars, {folds_run} folds)'

    def predict(self, df_ind: pd.DataFrame):
        """Predict on the last closed bar. Returns (signal in {-1,0,1}, confidence 0-1).
        Trained-but-unvalidated models degrade to no-signal — this only narrows
        entries/exits, it never manufactures a trade the walk-forward gate
        didn't clear."""
        if not self.trained or not self.validated or self.model is None:
            return 0, 0.5
        feats = [c for c in FEATURE_COLUMNS if c in df_ind.columns]
        X_last = df_ind[feats].iloc[[-1]]
        proba = self.model.predict_proba(X_last)[0]
        classes = list(self.model.classes_)
        best = int(np.argmax(proba))
        return int(classes[best]), float(proba[best])
