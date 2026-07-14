"""
Deterministic technical setups, ported from CryptoQuantScanner
backend/trading_service.py detect_setups(). Evaluated on the LAST CLOSED
candle. In this long-only cash-account port, signal=+1 setups are entry
candidates; signal=-1 setups never open shorts — they act as entry vetoes
and exit pressure on held longs.
"""

import math
import pandas as pd


def _g(row, col, default):
    v = row.get(col, default)
    try:
        return default if pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return default


def detect_setups(df: pd.DataFrame) -> list:
    """df: indicator-enriched bars, closed candles only. Returns
    [{'signal': ±1, 'confidence': float, 'name': str}, ...]"""
    setups = []
    if df is None or len(df) < 30:
        return setups
    row = df.iloc[-1]
    prev = df.iloc[-2]

    rsi = _g(row, 'rsi', 50)
    rsi_prev2 = _g(df.iloc[-3], 'rsi', 50)
    vol = _g(row, 'volume_ratio', 1.0)
    rng5 = _g(row, 'range_pos_5', 0.5)
    ema_dist = _g(row, 'ema200_distance', 0.0)
    adx = _g(row, 'adx', 20)
    close = _g(row, 'close', 0)
    prev_close = _g(prev, 'close', 0)
    vol_bonus = 0.04 if vol >= 2.0 else (0.02 if vol >= 1.5 else 0.0)

    # 1. RSI oversold bounce (long) — hooked up off the low, not a falling knife
    if (rsi < 32 and rsi > rsi_prev2 + 1.5 and rng5 > 0.25
            and ema_dist > -0.05 and close > prev_close):
        setups.append({'signal': 1, 'confidence': 0.74 + vol_bonus, 'name': 'rsi_bounce'})
    # RSI overbought fade (bearish pressure)
    if (rsi > 68 and rsi < rsi_prev2 - 1.5 and rng5 < 0.75
            and ema_dist < 0.05 and close < prev_close):
        setups.append({'signal': -1, 'confidence': 0.74 + vol_bonus, 'name': 'rsi_fade'})

    # 2. RSI divergence over the last 20 closed candles
    seg = df.tail(20)
    if len(seg) >= 20 and seg['rsi'].notna().all():
        lo_r, lo_p = seg['low'].iloc[-7:], seg['low'].iloc[:-7]
        rs_r, rs_p = seg['rsi'].iloc[-7:], seg['rsi'].iloc[:-7]
        hi_r, hi_p = seg['high'].iloc[-7:], seg['high'].iloc[:-7]
        if (lo_r.min() < lo_p.min() and rs_r.min() > rs_p.min() + 2
                and rsi < 45 and close > prev_close):
            setups.append({'signal': 1, 'confidence': 0.78 + vol_bonus, 'name': 'bull_divergence'})
        if (hi_r.max() > hi_p.max() and rs_r.max() < rs_p.max() - 2
                and rsi > 55 and close < prev_close):
            setups.append({'signal': -1, 'confidence': 0.78 + vol_bonus, 'name': 'bear_divergence'})

    # 3. Breakout — new 20-bar extreme on real volume with trend strength
    if _g(row, 'is_new_high', 0) >= 1 and vol >= 1.5 and adx >= 18:
        setups.append({'signal': 1, 'confidence': 0.76 + vol_bonus, 'name': 'breakout_long'})
    if _g(row, 'is_new_low', 0) >= 1 and vol >= 1.5 and adx >= 18:
        setups.append({'signal': -1, 'confidence': 0.76 + vol_bonus, 'name': 'breakout_down'})

    # 4. Shakeout — wick through the prior range edge, close back inside
    dc_low_prev = _g(prev, 'donchian_low', float('nan'))
    dc_high_prev = _g(prev, 'donchian_high', float('nan'))
    if not math.isnan(dc_low_prev) and vol >= 1.3:
        if _g(row, 'low', 0) < dc_low_prev and close > dc_low_prev:
            setups.append({'signal': 1, 'confidence': 0.78 + vol_bonus, 'name': 'shakeout_spring'})
    if not math.isnan(dc_high_prev) and vol >= 1.3:
        if _g(row, 'high', 0) > dc_high_prev and close < dc_high_prev:
            setups.append({'signal': -1, 'confidence': 0.78 + vol_bonus, 'name': 'shakeout_upthrust'})

    return setups


def best_setup(df: pd.DataFrame, ml_signal: int, ml_conf: float):
    """
    Resolve fired setups against the ML opinion (port of _best_setup).
      - Conflicting directions in the same candle -> stand down.
      - ML strongly disagrees (opposite at >=80%) -> veto.
      - ML agrees -> +0.08 confidence boost.
      - Multiple same-direction setups stack +0.03 each.
    Returns (signal, confidence, name) or (0, 0.0, None).
    """
    fired = detect_setups(df)
    if not fired:
        return 0, 0.0, None
    longs = [s for s in fired if s['signal'] == 1]
    shorts = [s for s in fired if s['signal'] == -1]
    if longs and shorts:
        return 0, 0.0, None
    group = longs or shorts
    sig = group[0]['signal']
    if ml_signal == -sig and ml_conf >= 0.80:
        return 0, 0.0, None
    conf = max(s['confidence'] for s in group) + 0.03 * (len(group) - 1)
    if ml_signal == sig:
        conf += 0.08
    conf = min(0.93, conf)
    name = '+'.join(s['name'] for s in group)
    return sig, conf, name
