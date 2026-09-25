"""
Stage B of the backtest: portfolio replay of the engine's decisions.

Consumes stage A's point-in-time signal records and replays the live
decision logic bar by bar, calling the engine's own functions wherever they
exist (exit_decision, entry_filter, entry_ev, ensemble, recent_expectancy)
so the simulator cannot drift from the code that trades.

Execution model — chosen to mirror how the live loop actually trades:
  * A decision is taken after each hourly bar closes, on bars through that
    bar. It fills at the NEXT bar's open, +/- slippage. For the day's last
    bar that is the next morning's open, which is the live morning cycle:
    stale bars, a fresh quote, and the overnight gap in the fill.
  * Exit conditions are evaluated against that next open — the "quote" the
    live cycle now prices exits from — so polled stops gap exactly as live.
  * Entry gap haircut uses (next open vs signal close), as live does.

Account models:
  * cash    — proceeds settle next trading day; buying power is settled cash;
              non-stop exits defer on the entry day (the engine's GFV guard).
  * margin  — proceeds available immediately; no GFV deferral, but the
              pattern-day-trader rule caps same-day round trips at 3 per 5
              trading days under $25k. A same-day stop still executes and is
              counted as a violation if it breaches the cap.
  * leverage > 1 (margin only) — gross exposure up to L x equity; borrowed
              cash accrues interest daily.
Shorts (margin only) reuse exit_decision through a mirrored price
(2*avg - p), which maps a short's P&L, high-water mark and trail exactly
onto the long-only logic.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot.strategy import engine

ET = 'America/New_York'


@dataclass
class Params:
    name: str = 'baseline'
    universe: tuple = ()                 # symbols eligible for ENTRY (held ones still managed)
    start: str = None                    # first decision date (ET), inclusive
    end: str = None
    gate: str = 'gross'                  # ML validation rule: gross | net | net_long
    val_cost: float = 0.002              # round-trip cost charged inside net gates
    entry_gate: str = None               # None | 'validated' | 'tier'
    tier_min: float = 0.0                # min side-specific net OOS expectancy for 'tier'
    static_universe: tuple = None        # fixed entry list (item 2, static form)
    account: str = 'cash'                # cash | margin | crypto
    leverage: float = 1.0
    margin_rate: float = 0.06
    shorts: bool = False
    slippage: float = 0.001              # per side, applied to every fill
    time_stop_hours: float = None
    expectancy_block: bool = True
    expectancy_expiry_days: int = None   # None = the original permanent latch
    max_pos_pct: float = 25.0
    max_open: int = 15
    start_equity: float = 1441.11
    cfg_overrides: dict = field(default_factory=dict)


BASE_CFG = {
    'min_confidence': 0.6, 'setup_min_confidence': 0.72, 'adx_threshold': 20,
    'min_volume_ratio': 0.65, 'trailing_stop_pct': 4.0, 'breakeven_arm_pct': 2.5,
    'breakeven_floor_pct': 0.5, 'min_hold_hours': 3, 'reversal_confirm_cycles': 2,
    'max_drawdown_pct': 25, 'per_position_stop_loss_pct': 5,
    'per_position_take_profit_pct': 10, 'daily_loss_stop_pct': 10,
    'entry_max_gap_ratio': 1.0, 'entry_gap_min_scale': 0.5,
    'min_order_usd': 5.0, 'cash_reserve_usd': 1.0,
}


def resolve_setups(fired, ml_signal, ml_conf):
    """engine.setups.best_setup, applied to a precomputed fired-setup list."""
    if not fired:
        return 0, 0.0, None
    longs = [s for s in fired if s[0] == 1]
    shorts = [s for s in fired if s[0] == -1]
    if longs and shorts:
        return 0, 0.0, None
    group = longs or shorts
    sig = group[0][0]
    if ml_signal == -sig and ml_conf >= 0.80:
        return 0, 0.0, None
    conf = max(s[1] for s in group) + 0.03 * (len(group) - 1)
    if ml_signal == sig:
        conf += 0.08
    return sig, min(0.93, conf), '+'.join(s[2] for s in group)


def _side_net(r, side, cost):
    if side == 1:
        return (r['long_gross'] - cost) if r['n_long'] >= 20 and pd.notna(r['long_gross']) else np.nan
    return (r['short_gross'] - cost) if r['n_short'] >= 20 and pd.notna(r['short_gross']) else np.nan


def gate_ok(r, p):
    """Is this symbol's ML stream usable today under the chosen rule?"""
    if not r['ml_trained'] or r['val_folds'] < 2 or r['val_n'] < 40:
        return False
    if p.gate == 'gross':
        return r['gross_both'] > 0
    if p.gate == 'net':
        return r['gross_both'] - p.val_cost > 0
    if p.gate == 'net_long':
        v = _side_net(r, 1, p.val_cost)
        return pd.notna(v) and v > 0
    raise ValueError(p.gate)


def entry_allowed(r, p, side, ml_ok):
    if p.static_universe is not None and r['sym'] not in p.static_universe:
        return False
    if p.entry_gate is None:
        return True
    if p.entry_gate == 'validated':
        if side == 1:
            return ml_ok
        v = _side_net(r, -1, p.val_cost)
        return ml_ok and pd.notna(v) and v > 0
    if p.entry_gate == 'tier':
        v = _side_net(r, side, p.val_cost)
        return ml_ok and pd.notna(v) and v >= p.tier_min
    raise ValueError(p.entry_gate)


def load_signals(sigdir, symbols):
    frames = []
    for s in symbols:
        try:
            df = pd.read_pickle(f'{sigdir}/{s}.pkl')
        except FileNotFoundError:
            continue
        if df.empty:
            continue
        df = df.copy()
        df['sym'] = s
        frames.append(df)
    allsig = pd.concat(frames).sort_index()
    allsig['fill_date'] = pd.to_datetime(allsig['next_ts'], utc=True).dt.tz_convert(ET).dt.date
    return allsig


def prepare(allsig: pd.DataFrame) -> list:
    """Group signal rows by bar time once, as plain dicts — reused by every variant."""
    sig = allsig[allsig['next_open'].notna()]
    out = []
    for ts, g in sig.groupby(level=0, sort=True):
        rows = g.to_dict('records')
        out.append((ts, rows[0]['fill_date'], rows[0]['date'], rows))
    return out


def run(prepared: list, p: Params) -> dict:
    cfg = dict(BASE_CFG)
    cfg['time_stop_hours'] = p.time_stop_hours
    cfg.update(p.cfg_overrides)
    slip = p.slippage
    margin = p.account == 'margin'
    # Crypto (Robinhood, 2026-09-23 crypto lane): proceeds are spendable at
    # once and there is no GFV or pattern-day-trader rule, so same-day exits
    # are never deferred. No borrowing: leverage and shorts stay margin-only.
    instant = margin or p.account == 'crypto'
    lev = p.leverage if margin else 1.0
    shorts = p.shorts and margin

    lo = pd.Timestamp(p.start).date() if p.start else None
    hi = pd.Timestamp(p.end).date() if p.end else None
    steps = [x for x in prepared if (lo is None or x[2] >= lo) and (hi is None or x[2] <= hi)]

    cash_settled = p.start_equity
    unsettled = []                       # (settle_date, amount)
    pos = {}                             # sym -> dict(qty, avg, side, meta, entry_date)
    last_px = {}
    trades, day_trades = [], []          # day_trades: list of dates
    pdt_violations = 0
    cur_day, sod, peak, halted = None, None, None, False
    daily_equity = []
    blocked_by_expectancy = 0
    decisions = 0
    trading_days = sorted({x[1] for x in steps})
    day_index = {d: i for i, d in enumerate(trading_days)}
    exposure_samples = []

    def equity():
        cash = cash_settled + sum(a for _, a in unsettled)
        return cash + sum(v['qty'] * last_px.get(s, v['avg']) for s, v in pos.items())

    def gross_exposure():
        return sum(abs(v['qty'] * last_px.get(s, v['avg'])) for s, v in pos.items())

    def next_trading_day(d):
        i = day_index.get(d)
        return trading_days[i + 1] if i is not None and i + 1 < len(trading_days) else d

    def close_position(sym, fill_px, reason, ts, d):
        nonlocal cash_settled
        v = pos.pop(sym)
        if v['side'] == 1:
            proceeds = v['qty'] * fill_px
            pnl = (fill_px - v['avg']) / v['avg'] * 100
        else:
            proceeds = v['qty'] * fill_px          # qty < 0: buying back is a cash outflow
            pnl = (v['avg'] - fill_px) / v['avg'] * 100
        if instant or v['side'] == -1:
            cash_settled += proceeds
        else:
            unsettled.append((next_trading_day(d), proceeds))
        usd = (fill_px - v['avg']) * v['qty']
        trades.append({'sym': sym, 'side': v['side'], 'reason': reason, 'pnl_pct': pnl,
                       'pnl_usd': usd, 'entry_date': v['entry_date'], 'exit_date': d,
                       'exit_ts': ts, 'event': 'sell'})
        if v['entry_date'] == d:
            day_trades.append(d)

    def pdt_count(d):
        i = day_index[d]
        window = set(trading_days[max(0, i - 4):i + 1])
        return sum(1 for x in day_trades if x in window)

    for ts, d, _, bar in steps:
        for r in bar:
            last_px[r['sym']] = r['close']

        # ── day rollover (live convention: sod and peak reset to open equity)
        if d != cur_day:
            if cur_day is not None:
                daily_equity.append((cur_day, equity()))
            cur_day = d
            matured = [a for sd, a in unsettled if sd <= d]
            unsettled = [(sd, a) for sd, a in unsettled if sd > d]
            cash_settled += sum(matured)
            if margin and cash_settled < 0:
                cash_settled -= (-cash_settled) * p.margin_rate / 252
            sod = equity()
            peak = sod
            halted = False
        if halted:
            continue
        decisions += 1
        E = equity()
        peak = max(peak, E)
        exposure_samples.append(gross_exposure() / E if E > 0 else 0)

        # ── daily loss stop: flatten at the next open and stand down for the day
        if sod > 0 and (sod - E) / sod >= cfg['daily_loss_stop_pct'] / 100:
            for r in bar:
                if r['sym'] in pos:
                    px = r['next_open'] * (1 - slip if pos[r['sym']]['side'] == 1 else 1 + slip)
                    close_position(r['sym'], px, 'daily_stop', ts, d)
            halted = True
            continue

        closed = [t for t in trades if t['event'] == 'sell']
        entries_blocked = None
        if peak > 0 and (peak - E) / peak > cfg['max_drawdown_pct'] / 100:
            entries_blocked = 'drawdown'
        elif p.expectancy_block and engine.expectancy_block(closed, d, p.expectancy_expiry_days):
            entries_blocked = 'expectancy'
            blocked_by_expectancy += 1
        max_pos = p.max_open
        if sod > 0 and (E - sod) / sod > 0.30:
            max_pos = max(1, max_pos // 2)

        # ── signals for every symbol that printed this bar
        sigs = {}
        for r in bar:
            ml_ok = gate_ok(r, p)
            ml_sig, ml_conf = (r['ml_cls'], r['ml_prob']) if ml_ok else (0, 0.5)
            s, c = engine.ensemble(ml_sig, ml_conf, r['se_sig'], r['se_conf'])
            setup = None
            if r['sym'] not in pos and (s == 0 or c < cfg['min_confidence']):
                ss, sc, sn = resolve_setups(r['fired'], s, c)
                if ss != 0 and sc >= cfg['setup_min_confidence']:
                    s, c, setup = ss, sc, sn
            sigs[r['sym']] = (s, c, setup, ml_ok, r)

        # ── exits (live: quote-priced; here the next open is the quote)
        for sym in list(pos):
            if sym not in sigs:
                continue
            s, c, _, _, r = sigs[sym]
            v = pos[sym]
            quote = r['next_open']
            same_day = v['entry_date'] == d
            if v['side'] == 1:
                px_eval, sig_eval = quote, s
            else:
                px_eval, sig_eval = 2 * v['avg'] - quote, -s
            defer = same_day and not instant
            ok, reason, meta = engine.exit_decision(v['meta'], v['avg'], px_eval, sig_eval, c,
                                                    cfg, held_today=defer)
            if ok and margin and same_day:
                if reason != 'stop_loss' and pdt_count(d) >= 3:
                    ok = False
                elif reason == 'stop_loss' and pdt_count(d) >= 3:
                    pdt_violations += 1
            v['meta'] = meta
            if ok:
                fill = quote * (1 - slip) if v['side'] == 1 else quote * (1 + slip)
                close_position(sym, fill, reason, ts, d)

        if entries_blocked:
            continue

        # ── entries
        E = equity()
        if margin:
            cash_left = lev * E - gross_exposure() - cfg['cash_reserve_usd']
        else:
            cash_left = cash_settled - cfg['cash_reserve_usd']
        slots = max(0, max_pos - len(pos))
        cands = []
        for sym, (s, c, setup, ml_ok, r) in sigs.items():
            if sym in pos or sym not in p.universe or c < cfg['min_confidence']:
                continue
            if s == 1 or (s == -1 and shorts):
                if entry_allowed(r, p, s, ml_ok):
                    cands.append((c, sym, s, setup, r))
        cands.sort(key=lambda x: -x[0])
        for c, sym, side, setup, r in cands:
            if slots <= 0 or cash_left < cfg['min_order_usd']:
                break
            adx = 0.0 if pd.isna(r['adx']) else r['adx']
            vr = 1.0 if pd.isna(r['vr']) else r['vr']
            if adx < cfg['adx_threshold'] or vr < cfg['min_volume_ratio']:
                continue
            ev = engine.entry_ev(c, [t for t in trades if t['event'] == 'sell'],
                                 cfg['per_position_take_profit_pct'] / 100,
                                 cfg['per_position_stop_loss_pct'] / 100, slip)
            if ev <= 0:
                continue
            quote, sp = r['next_open'], r['close']
            gap = ((quote - sp) / sp) * side if sp > 0 else 0.0
            scale = 1.0
            if gap > 0:
                if gap >= ev * cfg['entry_max_gap_ratio']:
                    continue
                scale = max((ev - gap) / ev, cfg['entry_gap_min_scale'])
            size = min(E * p.max_pos_pct / 100 * scale, cash_left)
            size = math.floor(size * 100) / 100
            if size < cfg['min_order_usd']:
                continue
            fill = quote * (1 + slip) if side == 1 else quote * (1 - slip)
            qty = size / fill * side
            cash_settled -= qty * fill                  # long: pay; short: receive
            hwm = quote if side == 1 else 2 * fill - quote
            pos[sym] = {'qty': qty, 'avg': fill, 'side': side, 'entry_date': d,
                        'meta': {'entry_date_et': str(d), 'high_water_mark': hwm,
                                 'be_armed': False, 'cycles_held': 0, 'reversal_streak': 0}}
            cash_left -= size
            slots -= 1

    if cur_day is not None:
        daily_equity.append((cur_day, equity()))
    return summarize(p, trades, daily_equity, pdt_violations, blocked_by_expectancy,
                     decisions, exposure_samples, pos)


def summarize(p, trades, daily_equity, pdt_violations, blocked, decisions, expo, open_pos):
    eq = pd.Series({d: e for d, e in daily_equity})
    start, end = p.start_equity, float(eq.iloc[-1]) if len(eq) else p.start_equity
    rets = eq.pct_change().fillna(eq.iloc[0] / start - 1 if len(eq) else 0)
    dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
    t = pd.DataFrame(trades)
    by_reason = {}
    if len(t):
        for k, g in t.groupby('reason'):
            by_reason[k] = (len(g), round(g['pnl_pct'].mean(), 3), round(g['pnl_usd'].sum(), 2))
    mid = len(eq) // 2
    h1 = float(eq.iloc[mid - 1] / start - 1) if mid > 0 else np.nan
    h2 = float(eq.iloc[-1] / eq.iloc[mid - 1] - 1) if mid > 0 else np.nan
    return {
        'name': p.name, 'days': len(eq), 'start': start, 'end': round(end, 2),
        'ret': end / start - 1, 'h1': h1, 'h2': h2, 'max_dd': float(dd),
        'trades': len(t), 'win_rate': float((t['pnl_pct'] > 0).mean()) if len(t) else np.nan,
        'mean_trade': float(t['pnl_pct'].mean()) if len(t) else np.nan,
        'pnl_usd': float(t['pnl_usd'].sum()) if len(t) else 0.0,
        'shorts': int((t['side'] == -1).sum()) if len(t) else 0,
        'exposure': float(np.mean(expo)) if expo else 0.0,
        'pdt_violations': pdt_violations, 'blocked_by_expectancy': blocked,
        'decisions': decisions, 'by_reason': by_reason, 'daily_returns': rets.values,
        'equity_curve': eq, 'trades_df': t, 'open_at_end': len(open_pos),
    }


def project(daily_returns, days=70, n=20000, block=5, seed=7, target=1_000_000 / 1441.11):
    """Block-bootstrap the variant's own daily returns forward `days` sessions."""
    r = np.asarray(daily_returns, dtype=float)
    if len(r) < block * 2:
        return None
    rng = np.random.default_rng(seed)
    nb = math.ceil(days / block)
    starts = rng.integers(0, len(r) - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :days]
    mult = np.prod(1 + r[idx], axis=1)
    return {
        'median': float(np.median(mult)), 'p10': float(np.percentile(mult, 10)),
        'p90': float(np.percentile(mult, 90)), 'p99': float(np.percentile(mult, 99)),
        'max': float(mult.max()), 'p_loss': float((mult < 1).mean()),
        'p_2x': float((mult >= 2).mean()), 'p_10x': float((mult >= 10).mean()),
        'p_target': float((mult >= target).mean()),
    }
