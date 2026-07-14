"""
Indicator + feature engineering, ported from CryptoQuantScanner
backend/trading_service.py calculate_indicators(). Pure pandas/numpy —
no talib/pandas_ta dependency. Input: DataFrame with columns
open/high/low/close/volume and a DatetimeIndex.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position',
    'atr', 'adx', 'stoch_k', 'stoch_d', 'cci', 'mfi',
    'roc', 'mom', 'trend_sma', 'volatility', 'volume_ratio',
    'vwap_distance', 'vwap_slope',
    'obv_slope', 'ad_slope', 'cmf',
    'volume_price_confirm', 'volume_divergence',
    'breakout_proximity', 'breakout_quality',
    'vol_weighted_mom', 'vol_weighted_roc',
    'bb_squeeze', 'in_squeeze',
    'vol_adj_adx', 'directional_volume',
    'ema200_distance', 'ema200_slope',
    'ret_3', 'ret_5', 'ret_10', 'ret_accel',
    'rsi_delta3', 'macd_hist_delta3', 'adx_delta3',
    'bb_position_delta3', 'volume_ratio_delta3',
    'candle_streak', 'green_frac_10', 'range_pos_5', 'vol_regime',
]


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), n)
    loss = _wilder((-delta).clip(lower=0), n)
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def _atr(h, l, c, n: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return _wilder(tr, n)


def _adx(h, l, c, n: int = 14) -> pd.Series:
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    atr = _atr(h, l, c, n)
    plus_di = 100 * _wilder(plus_dm, n) / (atr + 1e-10)
    minus_di = 100 * _wilder(minus_dm, n) / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return _wilder(dx, n)


def _stoch(h, l, c, k: int = 14, smooth: int = 3, d: int = 3):
    ll = l.rolling(k).min()
    hh = h.rolling(k).max()
    fast_k = 100 * (c - ll) / (hh - ll + 1e-10)
    slow_k = fast_k.rolling(smooth).mean()
    slow_d = slow_k.rolling(d).mean()
    return slow_k, slow_d


def _cci(h, l, c, n: int = 20) -> pd.Series:
    tp = (h + l + c) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-10)


def _mfi(h, l, c, v, n: int = 14) -> pd.Series:
    tp = (h + l + c) / 3
    mf = tp * v
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    ratio = pos.rolling(n).sum() / (neg.rolling(n).sum() + 1e-10)
    return 100 - 100 / (1 + ratio)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    o, h, l, c, v = (df[k].astype(float) for k in ('open', 'high', 'low', 'close', 'volume'))

    df['sma_10'] = c.rolling(10).mean()
    df['sma_20'] = c.rolling(20).mean()
    df['ema_12'] = _ema(c, 12)
    df['ema_26'] = _ema(c, 26)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = _ema(df['macd'], 9)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['rsi'] = _rsi(c, 14)

    df['bb_middle'] = c.rolling(20).mean()
    bb_std = c.rolling(20).std(ddof=0)
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    df['atr'] = _atr(h, l, c, 14)
    df['adx'] = _adx(h, l, c, 14)
    df['stoch_k'], df['stoch_d'] = _stoch(h, l, c)
    df['cci'] = _cci(h, l, c, 20)
    df['mfi'] = _mfi(h, l, c, v, 14)
    df['roc'] = c.pct_change(10) * 100
    df['mom'] = c - c.shift(10)
    df['trend_sma'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
    df['returns'] = c.pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['volume_sma'] = v.rolling(20).mean()
    df['volume_ratio'] = v / (df['volume_sma'] + 1e-10)

    tp = (h + l + c) / 3
    df['vwap'] = (tp * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-10)
    df['vwap_distance'] = (c - df['vwap']) / (df['vwap'] + 1e-10)
    df['vwap_slope'] = df['vwap'].pct_change(periods=5) * 100

    df['obv'] = (np.sign(df['returns'].fillna(0)) * v).cumsum()
    df['obv_slope'] = df['obv'].pct_change(periods=5).replace([np.inf, -np.inf], 0) * 100

    mfm = ((c - l) - (h - c)) / (h - l + 1e-10)
    df['ad'] = (mfm * v).cumsum()
    df['ad_slope'] = df['ad'].pct_change(periods=5).replace([np.inf, -np.inf], 0) * 100
    df['cmf'] = (mfm * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-10)

    price_ch5 = c.pct_change(periods=5)
    vol_ch5 = v.pct_change(periods=5)
    df['volume_price_confirm'] = np.sign(price_ch5) * np.sign(vol_ch5)
    exp_vol = price_ch5.abs() * df['volume_sma']
    act_vol = (v - df['volume_sma']).abs()
    df['volume_divergence'] = (act_vol - exp_vol) / (df['volume_sma'] + 1e-10)

    df['donchian_high'] = h.rolling(20).max()
    df['donchian_low'] = l.rolling(20).min()
    df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2
    dc_range = df['donchian_high'] - df['donchian_low']
    df['breakout_proximity'] = (c - df['donchian_mid']) / (dc_range / 2 + 1e-10)
    df['is_new_high'] = (c >= df['donchian_high'].shift(1)).astype(int)
    df['is_new_low'] = (c <= df['donchian_low'].shift(1)).astype(int)
    df['breakout_quality'] = (
        df['is_new_high'] * (1 + df['volume_ratio'].clip(0, 2) - 1)
        - df['is_new_low'] * (1 + df['volume_ratio'].clip(0, 2) - 1)
    )

    df['vol_weighted_mom'] = (df['returns'] * df['volume_ratio']).rolling(10).sum()
    df['vol_weighted_roc'] = df['vol_weighted_mom'].pct_change(periods=5).replace([np.inf, -np.inf], 0) * 100

    df['ema200'] = _ema(c, 200)
    df['ema200_distance'] = (c - df['ema200']) / (df['ema200'] + 1e-10)
    df['ema200_slope'] = df['ema200'].pct_change(periods=10) * 100

    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
    df['bb_squeeze'] = df['bb_width'] / (df['bb_width'].rolling(50).mean() + 1e-10)
    df['keltner_upper'] = df['ema_12'] + 2 * df['atr']
    df['keltner_lower'] = df['ema_12'] - 2 * df['atr']
    df['in_squeeze'] = ((df['bb_lower'] > df['keltner_lower'])
                        & (df['bb_upper'] < df['keltner_upper'])).astype(int)

    df['vol_adj_adx'] = df['adx'] * df['volume_ratio'].clip(0.5, 2)
    up_vol = pd.Series(np.where(c > o, v, 0.0), index=df.index)
    dn_vol = pd.Series(np.where(c < o, v, 0.0), index=df.index)
    df['directional_volume'] = ((up_vol.rolling(10).sum() - dn_vol.rolling(10).sum())
                                / (v.rolling(10).sum() + 1e-10))

    df['ret_3'] = c.pct_change(periods=3)
    df['ret_5'] = c.pct_change(periods=5)
    df['ret_10'] = c.pct_change(periods=10)
    df['ret_accel'] = df['returns'] - df['returns'].shift(1)
    df['rsi_delta3'] = df['rsi'] - df['rsi'].shift(3)
    df['macd_hist_delta3'] = df['macd_hist'] - df['macd_hist'].shift(3)
    df['adx_delta3'] = df['adx'] - df['adx'].shift(3)
    df['bb_position_delta3'] = df['bb_position'] - df['bb_position'].shift(3)
    df['volume_ratio_delta3'] = df['volume_ratio'] - df['volume_ratio'].shift(3)

    _dir = np.sign(df['returns'].fillna(0))
    _run_grp = (_dir != _dir.shift()).cumsum()
    _run_len = _dir.groupby(_run_grp).cumcount() + 1
    df['candle_streak'] = (_run_len * _dir).clip(-10, 10)
    df['green_frac_10'] = (df['returns'] > 0).rolling(10).mean()

    _hi5 = h.rolling(5).max()
    _lo5 = l.rolling(5).min()
    df['range_pos_5'] = (c - _lo5) / (_hi5 - _lo5 + 1e-10)
    df['vol_regime'] = df['volatility'] / (df['volatility'].rolling(50).mean() + 1e-10)

    return df
