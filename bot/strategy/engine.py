"""
Decision engine — long-only cash-account port of CryptoQuantScanner
backend/trading_service.py (ensemble, governors, sizing, exit engine).

Consumes a snapshot dict (bars, portfolio, positions, state, config,
trade history), emits a decisions dict (exits, entries, state updates,
halt flags, diagnostics). Pure function of its input — all broker I/O
happens outside (the agent session fetches data and places orders).

Crypto-only features intentionally dropped: leverage (all thresholds are
plain price %), shorts (bearish signals veto entries / pressure exits),
maker-taker fees (replaced by a slippage buffer), funding, 24/7 sessions.
"""

import numpy as np
import pandas as pd

from .indicators import calculate_indicators
from .ml_model import MLStream
from .setups import best_setup, detect_setups


# ── Signal-engine scoring stream (port of signal_engine.py generate_signal) ──

def indicator_score(df: pd.DataFrame):
    """Weighted indicator vote on the last closed bar.
    Returns (signal in {-1,0,1}, confidence 0-1, reasons list)."""
    if df is None or len(df) < 25:
        return 0, 0.0, []
    row, prev = df.iloc[-1], df.iloc[-2]
    bull, bear, reasons = 0, 0, []

    rsi = row.get('rsi', np.nan)
    if pd.notna(rsi):
        if rsi < 30:
            bull += 25; reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            bear += 25; reasons.append(f"RSI overbought ({rsi:.1f})")

    m, s = row.get('macd', np.nan), row.get('macd_signal', np.nan)
    mp, sp = prev.get('macd', np.nan), prev.get('macd_signal', np.nan)
    if pd.notna(m) and pd.notna(s) and pd.notna(mp) and pd.notna(sp):
        if mp <= sp and m > s:
            bull += 30; reasons.append("MACD bullish crossover")
        elif mp >= sp and m < s:
            bear += 30; reasons.append("MACD bearish crossover")

    vr = row.get('volume_ratio', np.nan)
    if pd.notna(vr) and vr > 2.5:
        bull += 15; bear += 15; reasons.append(f"Volume spike ({vr:.1f}x)")

    bbp, bbw = row.get('bb_position', np.nan), row.get('bb_width', np.nan)
    if pd.notna(bbp) and pd.notna(bbw):
        if bbp < 0.0:
            bull += 20; reasons.append("BB lower band bounce")
        elif bbp > 1.0:
            bear += 20; reasons.append("BB upper band rejection")

    st = row.get('stoch_k', np.nan)
    if pd.notna(st):
        if st < 20:
            bull += 10; reasons.append("Stochastic oversold")
        elif st > 80:
            bear += 10; reasons.append("Stochastic overbought")

    closes = df['close'].tail(20)
    if len(closes) == 20 and closes.std() > 0:
        z = (closes.iloc[-1] - closes.mean()) / closes.std()
        if z < -2:
            bull += 15; reasons.append(f"Price anomaly (z={z:.2f})")
        elif z > 2:
            bear += 15; reasons.append(f"Price anomaly (z={z:.2f})")

    if bull > bear and bull >= 40:
        return 1, min(bull, 100) / 100.0, reasons
    if bear > bull and bear >= 40:
        return -1, min(bear, 100) / 100.0, reasons
    return 0, 0.0, reasons


def ensemble(ml_signal, ml_conf, se_signal, se_conf):
    """Blend ML with the indicator-score second opinion (port of _ensemble_signal).
    Agreement: weighted average x1.2 boost. Disagreement: ML signal at x0.70."""
    if se_signal == 0:
        return ml_signal, ml_conf
    if se_signal == ml_signal:
        return ml_signal, min(0.99, (ml_conf * 0.6 + se_conf * 0.4) * 1.2)
    return ml_signal, ml_conf * 0.70


# ── Regime / governors / sizing helpers ──────────────────────────────────────

def market_regime(regime_bars: pd.DataFrame) -> str:
    """SPY daily EMA50 slope + price position (replaces the BTC 4h regime)."""
    if regime_bars is None or len(regime_bars) < 55:
        return 'sideways'
    close = regime_bars['close'].astype(float)
    ema50 = close.ewm(span=50, adjust=False).mean()
    price_vs_ema = (close.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1]
    ema_slope = (ema50.iloc[-1] - ema50.iloc[-10]) / ema50.iloc[-10]
    if price_vs_ema > 0.02 and ema_slope > 0.001:
        return 'bull'
    if price_vs_ema < -0.02 and ema_slope < -0.001:
        return 'bear'
    return 'sideways'


def regime_multiplier(regime: str) -> float:
    """Long-only: bull +20%, bear -60%, sideways -20%."""
    return {'bull': 1.2, 'bear': 0.4}.get(regime, 0.8)


def drawdown_scale(equity: float, peak_equity: float) -> float:
    """Graduated size scale by drawdown from peak: 1.0 / 0.75 / 0.5 / 0.25."""
    if peak_equity <= 0:
        return 1.0
    dd = (peak_equity - equity) / peak_equity
    if dd <= 0.05:
        return 1.00
    if dd <= 0.10:
        return 0.75
    if dd <= 0.15:
        return 0.50
    return 0.25


def adaptive_scale(closed_trades: list) -> float:
    """Aggression throttle from recent realized performance, bounded [0.75, 1.30]."""
    recent = closed_trades[-20:]
    if len(recent) < 8:
        return 1.0
    wins = sum(1 for t in recent if t['pnl_pct'] > 0)
    wr = wins / len(recent)
    avg_roi = float(np.mean([t['pnl_pct'] / 100.0 for t in recent]))
    scale = 1.0
    if wr >= 0.65:
        scale += 0.20
    elif wr >= 0.55:
        scale += 0.10
    elif wr < 0.45:
        scale -= 0.15
    elif wr < 0.50:
        scale -= 0.08
    if avg_roi > 0.01:
        scale += 0.10
    elif avg_roi < 0.0:
        scale -= 0.10
    return max(0.75, min(1.30, scale))


def performance_floor(closed_trades: list) -> bool:
    """True = block new entries (win rate < 40% over the last 20 closed trades)."""
    recent = closed_trades[-20:]
    if len(recent) < 20:
        return False
    wins = sum(1 for t in recent if t['pnl_pct'] > 0)
    return wins / len(recent) < 0.40


def half_kelly_fraction(closed_trades: list, fallback: float, cap: float) -> float:
    """Half-Kelly position fraction from the last 50 closed trades; falls back
    to `fallback` until 20 trades are on record. Bounds: [0.05, cap]."""
    closed = closed_trades[-50:]
    if len(closed) < 20:
        return fallback
    wins = [t['pnl_pct'] / 100.0 for t in closed if t['pnl_pct'] > 0]
    losses = [abs(t['pnl_pct']) / 100.0 for t in closed if t['pnl_pct'] < 0]
    if not wins or not losses:
        return fallback
    p = len(wins) / len(closed)
    b = (sum(wins) / len(wins)) / max(sum(losses) / len(losses), 1e-6)
    kelly = p - (1 - p) / max(b, 1e-6)
    return max(0.05, min(cap, kelly * 0.5)) if kelly > 0 else 0.05


def volatility_multiplier(df: pd.DataFrame) -> float:
    """Inverse-ATR size scale, 0.5x-1.5x. Baseline 0.5% ATR/price — typical
    for hourly equity bars (the crypto original used 0.15% on 5m bars)."""
    if 'atr' not in df.columns or df['atr'].isna().all():
        return 1.0
    atr, price = df['atr'].iloc[-1], df['close'].iloc[-1]
    if price <= 0 or atr <= 0 or pd.isna(atr):
        return 1.0
    return max(0.5, min(1.5, 0.005 / (atr / price + 1e-10)))


def entry_filter(df: pd.DataFrame, adx_threshold: float, min_volume_ratio: float):
    """Quality gate: ADX >= threshold, volume_ratio >= floor. NaN ADX blocks."""
    row = df.iloc[-1]
    adx = row.get('adx', np.nan)
    vol = row.get('volume_ratio', np.nan)
    adx = 0.0 if pd.isna(adx) else float(adx)
    vol = 1.0 if pd.isna(vol) else float(vol)
    if adx < adx_threshold:
        return False, f"ADX {adx:.1f} < {adx_threshold}"
    if vol < min_volume_ratio:
        return False, f"volume_ratio {vol:.2f} < {min_volume_ratio}"
    return True, ""


def entry_ev(confidence: float, closed_trades: list, tp_pct: float,
             sl_pct: float, slippage_per_side: float) -> float:
    """Expected value per trade. Uses realized win/loss averages once >=10
    trades are on record; before that, nominal TP (capped 10%) and SL."""
    p = max(0.01, min(0.99, confidence))
    closed = closed_trades[-30:]
    wins = [t['pnl_pct'] / 100.0 for t in closed if t['pnl_pct'] > 0]
    losses = [abs(t['pnl_pct']) / 100.0 for t in closed if t['pnl_pct'] < 0]
    if len(closed) >= 10 and wins and losses:
        return p * float(np.mean(wins)) - (1 - p) * float(np.mean(losses))
    slip = 2 * slippage_per_side
    net_win = min(tp_pct, 0.10) - slip
    net_loss = sl_pct + slip
    return p * net_win - (1 - p) * net_loss


# ── Exit engine (port of _exit_logic, long-only, no leverage/fees) ───────────

def exit_decision(pos_meta: dict, avg_cost: float, price: float,
                  signal: int, confidence: float, cfg: dict, held_today: bool):
    """
    Returns (should_exit, reason, updated_meta). Priority:
      1. Stop loss / take profit — hard boundaries.
      2. Breakeven floor — once armed past the trigger, a winner can't
         round-trip into a loser.
      3. Trailing stop — retrace from the high-water mark while in profit.
      4. Time stop — never armed within time_stop_hours of market time.
      5. Signal reversal — entry-grade bearish confidence, min hold,
         2 consecutive cycles (hysteresis).
    Cash-account GFV guard: same-day exits are allowed ONLY for stop_loss
    (and the daily halt, handled by the caller).
    """
    meta = dict(pos_meta)
    pnl_pct = (price - avg_cost) / avg_cost if avg_cost > 0 else 0.0

    sl = cfg['per_position_stop_loss_pct'] / 100.0
    tp = cfg['per_position_take_profit_pct'] / 100.0

    if pnl_pct <= -sl:
        return True, 'stop_loss', meta

    deferred = held_today  # non-stop exits wait until the next trading day

    if pnl_pct >= tp:
        return (not deferred), 'take_profit', meta

    be_trigger = cfg['breakeven_arm_pct'] / 100.0
    be_floor = cfg['breakeven_floor_pct'] / 100.0
    if not meta.get('be_armed') and pnl_pct >= be_trigger:
        meta['be_armed'] = True
    if meta.get('be_armed') and pnl_pct <= be_floor:
        return (not deferred), 'breakeven_floor', meta

    hwm = max(float(meta.get('high_water_mark', avg_cost)), price)
    meta['high_water_mark'] = hwm
    trail = cfg['trailing_stop_pct'] / 100.0
    if pnl_pct > 0 and price <= hwm * (1 - trail):
        return (not deferred), 'trailing_stop', meta

    meta['cycles_held'] = int(meta.get('cycles_held', 0)) + 1
    if not meta.get('be_armed') and meta['cycles_held'] >= cfg['time_stop_hours']:
        return (not deferred), 'time_stop', meta

    if signal == -1 and confidence >= cfg['min_confidence'] \
            and meta['cycles_held'] >= cfg['min_hold_hours']:
        streak = int(meta.get('reversal_streak', 0)) + 1
        meta['reversal_streak'] = streak
        if streak >= cfg['reversal_confirm_cycles']:
            return (not deferred), 'signal_reversal', meta
    else:
        meta['reversal_streak'] = 0

    return False, '', meta


# ── Main cycle ────────────────────────────────────────────────────────────────

def run(snapshot: dict) -> dict:
    cfg_all = snapshot['config']
    cfg = dict(cfg_all['strategy'])
    cfg.update(cfg_all['risk'])  # flat lookup for the exit engine
    risk = cfg_all['risk']
    state = snapshot['state']
    portfolio = snapshot['portfolio']
    today_et = snapshot['today_et']
    closed_trades = [t for t in snapshot.get('trade_history', [])
                     if t.get('event') == 'sell' and t.get('pnl_pct') is not None]

    equity = float(portfolio['equity'])
    buying_power = float(portfolio['buying_power'])
    peak_equity = max(float(state.get('peak_equity') or 0.0), equity)

    decisions = {'exits': [], 'entries': [], 'halt': False, 'halt_reason': '',
                 'state_updates': {'peak_equity': peak_equity, 'positions': {}},
                 'diagnostics': {'symbols': {}}}

    # ── Daily loss stop ──────────────────────────────────────────────────────
    sod = float(state.get('start_of_day_equity') or equity)
    day_dd = (sod - equity) / sod if sod > 0 else 0.0
    if day_dd >= risk['daily_loss_stop_pct'] / 100.0:
        decisions['halt'] = True
        decisions['halt_reason'] = f"daily loss {day_dd:.1%} >= {risk['daily_loss_stop_pct']}%"
        for p in snapshot.get('positions', []):
            decisions['exits'].append({'symbol': p['symbol'], 'reason': 'daily_stop',
                                       'quantity': p['shares_available_for_sells']})
        return decisions

    # ── Hard drawdown stop (entries only; exits still managed below) ────────
    hard_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
    entries_blocked = None
    if hard_dd > cfg['max_drawdown_pct'] / 100.0:
        entries_blocked = f"drawdown {hard_dd:.1%} > max {cfg['max_drawdown_pct']}%"
    elif performance_floor(closed_trades):
        entries_blocked = "win rate < 40% over last 20 trades"

    # ── Regime ───────────────────────────────────────────────────────────────
    regime = market_regime(snapshot.get('regime_bars'))
    r_mult = regime_multiplier(regime)
    decisions['diagnostics']['regime'] = regime

    # ── Give-back protection: day up >30% → halve position slots ────────────
    max_pos = risk['max_open_positions']
    if sod > 0 and (equity - sod) / sod > 0.30:
        max_pos = max(1, max_pos // 2)

    held = {p['symbol']: p for p in snapshot.get('positions', [])}
    state_positions = state.get('positions', {})

    # ── Per-symbol signal computation ────────────────────────────────────────
    signals = {}
    for symbol, df in snapshot['bars'].items():
        if df is None or len(df) < 60:
            decisions['diagnostics']['symbols'][symbol] = {'skipped': 'insufficient bars'}
            continue
        # One symbol's bad data must never kill the whole cycle — degrade to
        # no-signal (held positions still get exit management via avg_cost).
        try:
            df_ind = calculate_indicators(df)
            ml = MLStream(cfg['forward_periods'], cfg['label_threshold'],
                          risk['per_position_stop_loss_pct'] / 100.0)
            ml.train(df_ind)
            ml_signal, ml_conf = ml.predict(df_ind)
            se_signal, se_conf, se_reasons = indicator_score(df_ind)
            signal, confidence = ensemble(ml_signal, ml_conf, se_signal, se_conf)

            setup_name = None
            if symbol not in held and (signal == 0 or confidence < cfg['min_confidence']):
                s_sig, s_conf, s_name = best_setup(df_ind, signal, confidence)
                if s_sig != 0 and s_conf >= cfg['setup_min_confidence']:
                    signal, confidence, setup_name = s_sig, s_conf, s_name

            signals[symbol] = {'signal': signal, 'confidence': confidence,
                               'setup': setup_name, 'df': df_ind,
                               'price': float(df_ind['close'].iloc[-1]),
                               'ml': (ml_signal, round(ml_conf, 3), ml.trained),
                               'se': (se_signal, round(se_conf, 3))}
            decisions['diagnostics']['symbols'][symbol] = {
                'signal': signal, 'confidence': round(confidence, 3),
                'ml_signal': ml_signal, 'ml_conf': round(ml_conf, 3),
                'ml_trained': ml.trained, 'ml_validated': ml.validated,
                'ml_validation': ml.validation_summary, 'se_signal': se_signal,
                'setup': setup_name, 'price': signals[symbol]['price'],
                'reasons': se_reasons[:4]}
        except Exception as e:
            signals.pop(symbol, None)
            decisions['diagnostics']['symbols'][symbol] = {
                'skipped': f'error: {type(e).__name__}: {e}'}

    # ── Exits ────────────────────────────────────────────────────────────────
    for symbol, pos in held.items():
        meta = dict(state_positions.get(symbol, {}))
        meta.setdefault('entry_date_et', today_et)
        sig = signals.get(symbol, {})
        avg_cost = float(pos.get('average_buy_price') or 0.0)
        price = sig.get('price') or avg_cost
        should_exit, reason, meta = exit_decision(
            meta, avg_cost, price, sig.get('signal', 0), sig.get('confidence', 0.0),
            cfg, held_today=(meta.get('entry_date_et') == today_et))
        if should_exit:
            decisions['exits'].append({'symbol': symbol, 'reason': reason,
                                       'quantity': pos['shares_available_for_sells'],
                                       'pnl_pct_est': round((price - avg_cost) / avg_cost * 100, 2)
                                       if avg_cost > 0 else None})
        else:
            decisions['state_updates']['positions'][symbol] = meta

    # ── Entries ──────────────────────────────────────────────────────────────
    if entries_blocked:
        decisions['diagnostics']['entries_blocked'] = entries_blocked
        return decisions

    exiting = {e['symbol'] for e in decisions['exits']}
    open_count = len([s for s in held if s not in exiting])
    slots = max(0, max_pos - open_count)
    cash_left = buying_power - cfg_all['cash_reserve_usd']

    candidates = [(sym, s) for sym, s in signals.items()
                  if s['signal'] == 1 and s['confidence'] >= cfg['min_confidence']
                  and sym not in held]
    candidates.sort(key=lambda kv: kv[1]['confidence'], reverse=True)

    for symbol, s in candidates:
        if slots <= 0 or cash_left < cfg_all['min_order_usd']:
            break
        ok, why = entry_filter(s['df'], cfg['adx_threshold'], cfg['min_volume_ratio'])
        if not ok:
            decisions['diagnostics']['symbols'][symbol]['entry_blocked'] = why
            continue
        ev = entry_ev(s['confidence'], closed_trades,
                      risk['per_position_take_profit_pct'] / 100.0,
                      risk['per_position_stop_loss_pct'] / 100.0,
                      cfg['slippage_pct_per_side'] / 100.0)
        if ev <= 0:
            decisions['diagnostics']['symbols'][symbol]['entry_blocked'] = f"EV {ev:.4f} <= 0"
            continue
        # Owner directive 2026-07-16: flat sizing at max_position_pct of equity.
        # The Kelly/adaptive/volatility/regime shrink multipliers are bypassed
        # for sizing (still computed in diagnostics); drawdown protection now
        # acts only through the max_drawdown entry block and daily stop.
        size = min(equity * risk['max_position_pct_of_equity'] / 100.0, cash_left)
        size = float(np.floor(size * 100) / 100)
        if size < cfg_all['min_order_usd']:
            decisions['diagnostics']['symbols'][symbol]['entry_blocked'] = f"size ${size:.2f} < min"
            continue
        decisions['entries'].append({
            'symbol': symbol, 'dollar_amount': f"{size:.2f}",
            'confidence': round(s['confidence'], 3),
            'reason': s['setup'] or 'ml_ensemble', 'ev': round(ev, 4)})
        decisions['state_updates']['positions'][symbol] = {
            'entry_date_et': today_et, 'entry_reason': s['setup'] or 'ml_ensemble',
            'entry_confidence': round(s['confidence'], 3),
            'high_water_mark': s['price'], 'be_armed': False,
            'cycles_held': 0, 'reversal_streak': 0}
        slots -= 1
        cash_left -= size

    return decisions
