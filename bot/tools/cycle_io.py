"""
Fixed, reviewable state I/O for a trading cycle (2026-09-24).

Cycles used to rewrite bot/state.json and append to bot/trade_log.jsonl with
throwaway inline scripts. Each one was new code, so each needed a fresh
permission approval, and a cycle waiting on that approval with nobody there
stalled after the engine had already decided. These subcommands are the only
state writes a cycle needs, so they can be pre-approved once.

    python -m bot.tools.cycle_io rollover <today_et> <total_value>
    python -m bot.tools.cycle_io persist <fills.json>

fills.json (written by the cycle after executing decisions.json):
  {"time_et": "HH:MM", "equity": float, "buying_power": float,
   "orders": [{"symbol", "side": "buy"|"sell", "ref_id", "state",
               "quantity", "avg_price", "dollar_amount", "reason"?}],
   "skips": [{"ts", "reason", "note"}],        # optional
   "note": "..."}                              # optional
It reads decisions.json and input.json from the repo root (the cycle's own
outputs), so positions, marks and pnl come from the same snapshot the engine
saw rather than being retyped.
"""

import json
import sys
from datetime import datetime, timezone

STATE = 'bot/state.json'
LOG = 'bot/trade_log.jsonl'


def _load(path):
    return json.load(open(path))


def _save_state(st):
    with open(STATE, 'w') as f:
        json.dump(st, f, indent=2)
        f.write('\n')


def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def rollover(today_et: str, total_value: float) -> dict:
    """Runbook step 1: new ET day -> reset start-of-day and peak to the open value."""
    st = _load(STATE)
    if st.get('date_et') == today_et:
        return {'rolled': False, 'date_et': today_et}
    st.update({'date_et': today_et, 'start_of_day_equity': total_value,
               'peak_equity': total_value, 'halted_today': False})
    _save_state(st)
    return {'rolled': True, 'date_et': today_et, 'start_of_day_equity': total_value}


def persist(fills: dict, decisions: dict, snapshot: dict) -> list:
    """Runbook step 5: merge engine state, correct for what actually filled, log."""
    st = _load(STATE)
    positions = dict(decisions['state_updates']['positions'])
    held = {p['symbol']: p for p in snapshot.get('positions', [])}
    quotes = snapshot.get('quotes', {})
    lines = []
    for s in fills.get('skips', []):
        lines.append({'ts': s.get('ts', _now()), 'event': 'skip', 'reason': s['reason'],
                      'note': s.get('note', '')})
    for o in fills.get('orders', []):
        filled = o.get('state') == 'filled'
        sym = o['symbol']
        rec = {'ts': o.get('ts', _now()), 'event': o['side'], 'symbol': sym,
               'price_ref': o.get('avg_price'), 'reason': o.get('reason'),
               'ref_id': o.get('ref_id'), 'order_state': o.get('state'),
               'quantity': o.get('quantity')}
        if o['side'] == 'buy':
            rec['dollar_amount'] = o.get('dollar_amount')
            if filled and sym in positions and o.get('avg_price'):
                positions[sym]['high_water_mark'] = float(o['avg_price'])  # seed HWM at the fill
            elif not filled:
                positions.pop(sym, None)  # rejected entry is not a position
        else:
            avg = float(held.get(sym, {}).get('average_buy_price') or 0)
            if filled and avg > 0 and o.get('avg_price'):
                px = float(o['avg_price'])
                rec['pnl_pct'] = round((px / avg - 1) * 100, 4)
                rec['pnl_usd'] = round((px - avg) * float(o.get('quantity') or 0), 4)
            if not filled and sym in st.get('positions', {}):
                positions[sym] = st['positions'][sym]  # failed exit: still held
        lines.append(rec)
    st['positions'] = positions
    st['peak_equity'] = decisions['state_updates']['peak_equity']
    st['last_run_utc'] = _now()
    _save_state(st)
    marks = {s: round((float(quotes[s]) / float(p['average_buy_price']) - 1) * 100, 2)
             for s, p in held.items() if s in quotes and float(p.get('average_buy_price') or 0) > 0}
    n = len([o for o in fills.get('orders', []) if o.get('state') == 'filled'])
    lines.append({'type': 'cycle_summary', 'timestamp_utc': _now(), 'date_et': snapshot['today_et'],
                  'time_et': fills.get('time_et'),
                  'regime': decisions.get('diagnostics', {}).get('regime'),
                  'exits': len(decisions['exits']), 'entries': len(decisions['entries']),
                  'actions': n, 'equity': fills.get('equity'),
                  'buying_power_raw': fills.get('buying_power'),
                  'open_positions': sorted(positions), 'live_marks_pct': marks,
                  'note': fills.get('note', '')})
    with open(LOG, 'a') as f:
        for x in lines:
            f.write(json.dumps(x) + '\n')
    return lines


def main():
    cmd = sys.argv[1]
    if cmd == 'rollover':
        print(json.dumps(rollover(sys.argv[2], float(sys.argv[3]))))
    elif cmd == 'persist':
        lines = persist(_load(sys.argv[2]), _load('decisions.json'), _load('input.json'))
        print(json.dumps(lines[-1]))
    else:
        raise SystemExit(f'unknown command {cmd}')


if __name__ == '__main__':
    main()
