import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from backend.mie.feature_engine import FeatureEngine


@pytest.fixture(autouse=True)
def isolated_mie_db(monkeypatch):
    """Every test gets its own throwaway SQLite file — no cross-test state."""
    path = tempfile.mktemp(suffix='.sqlite3')
    monkeypatch.setenv('MIE_DB_PATH', path)
    yield path
    try:
        os.remove(path)
    except OSError:
        pass


def make_ohlcv(n=500, seed=0, freq='5min', start_price=100.0, drift=0.0, vol=0.3):
    rng = np.random.RandomState(seed)
    idx = pd.date_range('2026-01-01', periods=n, freq=freq, tz='UTC')
    price = start_price + np.cumsum(rng.randn(n) * vol + drift)
    price = np.maximum(price, 1.0)
    df = pd.DataFrame({
        'open': price + rng.randn(n) * 0.05,
        'high': price + np.abs(rng.randn(n) * 0.2),
        'low': price - np.abs(rng.randn(n) * 0.2),
        'close': price,
        'volume': np.abs(rng.randn(n) * 1000 + 5000),
    }, index=idx)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    return df


def make_predictable_ohlcv(n=4000, seed=7, bar_seconds=60, horizon_sec=180, effect=0.006):
    """
    OHLCV whose forward `horizon_sec` cumulative return is genuinely predictable
    from a feature we hand back separately (`signal`) — used to prove the
    validation gate CAN pass when a real, horizon-matched edge exists.
    """
    rng = np.random.RandomState(seed)
    idx = pd.date_range('2026-01-01', periods=n, freq=f'{bar_seconds}s', tz='UTC')
    signal = rng.randn(n)
    k = max(1, round(horizon_sec / bar_seconds))
    fwd = -effect * signal
    ret = rng.randn(n) * 0.0015
    for i in range(n - k):
        for j in range(1, k + 1):
            ret[i + j] += fwd[i] / k
    price = 100 * np.cumprod(1 + ret)
    df = pd.DataFrame({
        'open': price * (1 + rng.randn(n) * 0.0002),
        'high': price * (1 + np.abs(rng.randn(n)) * 0.0006),
        'low': price * (1 - np.abs(rng.randn(n)) * 0.0006),
        'close': price,
        'volume': np.abs(rng.randn(n) * 1000 + 5000),
    }, index=idx)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    return df, signal


class SignalInjectingFeatureEngine(FeatureEngine):
    """
    Test double that appends an externally-known 'synthetic_signal' feature
    alongside everything the real FeatureEngine computes from OHLCV, on BOTH
    the historical (build_frame) and live (build_state) paths.

    This exists to test the edge model / engine honestly: a feature only
    present in a hand-built training frame but absent from build_state's live
    output would make the model appear to have an edge in training that it
    can never actually exercise in decide() — an artifact of the test
    fabricating data, not a property of the engine. Wiring the same signal
    into both paths, indexed by timestamp, is what a real live order-book or
    derivatives feature does implicitly (see feature_engine.py's live-only
    groups) and is what this test double reproduces explicitly and
    deterministically.
    """

    def __init__(self, signal: pd.Series, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal = signal

    def build_frame(self, df, references=None, keep_prices=True):
        feats = super().build_frame(df, references, keep_prices)
        feats['synthetic_signal'] = self.signal.reindex(feats.index)
        return feats

    def build_state(self, symbol, df, references=None, book=None, trades=None, derivs=None):
        state = super().build_state(symbol, df, references, book, trades, derivs)
        val = self.signal.get(state.as_of)
        if val is not None and pd.notna(val):
            state.features['synthetic_signal'] = float(val)
        return state
