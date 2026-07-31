"""
ML signal stream, ported from CryptoQuantScanner backend/trading_service.py
(create_labels / train_model / predict_signal). Per-symbol model, retrained
fresh each cycle from the fetched bars — no persistence, no drift.

Uses HistGradientBoostingClassifier: NaN-native (no imputer/scaler),
well-calibrated probabilities, pure sklearn.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .indicators import FEATURE_COLUMNS


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

    def __init__(self, forward_periods: int = 5, threshold: float = 0.004,
                 sl_price: float = 0.05, seed: int = 42):
        self.forward_periods = forward_periods
        self.threshold = threshold
        self.sl_price = sl_price
        self.seed = seed
        self.model = None
        self.trained = False

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
            return False
        self.model = HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=0.1,
            class_weight='balanced', random_state=self.seed,
        )
        # Thinly traded listings can leave a feature column constant, which
        # sklearn's binner rejects — degrade to untrained instead of crashing.
        try:
            self.model.fit(X, y)
        except Exception:
            self.model = None
            self.trained = False
            return False
        self.trained = True
        return True

    def predict(self, df_ind: pd.DataFrame):
        """Predict on the last closed bar. Returns (signal in {-1,0,1}, confidence 0-1)."""
        if not self.trained or self.model is None:
            return 0, 0.5
        feats = [c for c in FEATURE_COLUMNS if c in df_ind.columns]
        X_last = df_ind[feats].iloc[[-1]]
        proba = self.model.predict_proba(X_last)[0]
        classes = list(self.model.classes_)
        best = int(np.argmax(proba))
        return int(classes[best]), float(proba[best])
