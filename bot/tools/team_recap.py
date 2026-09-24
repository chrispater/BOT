"""
Team scoreboard and morning recap toward the $1MM goal (owner request 2026-09-24).

"We're a team": the bot trades stocks in the Agentic account; the owner trades
crypto by hand in the Investing account. This tracks both, read-only, and
answers one question each morning: how fast are we compounding toward $1MM?

Two numbers, kept apart on purpose:
  * value       — what the accounts are worth, deposits included. Progress
                  toward $1MM counts every dollar, however it got there.
  * performance — the change in value NOT explained by deposits or transfers.
                  This is the compounding rate, and the only fair way to say
                  whether the pace is improving.

Robinhood exposes no transfer history, so flows are inferred between
snapshots from what trades cannot explain:
  cash flow   = change in cash − (sell proceeds − buy costs)
  asset flow  = Σ (change in quantity − net filled quantity) × current price
Small unexplained amounts (< $0.50, rounding) are ignored. Dividends, interest
and staking rewards land in flows, not performance — a small, conservative
bias. A transfer between the two accounts nets to zero at the team level.

    python -m bot.tools.team_recap snapshot <spec.json>   # append today's snapshot
    python -m bot.tools.team_recap report                 # print the recap

spec.json (built each morning from read-only MCP calls):
  {"ts_utc": "...Z",
   "accounts": {"<key>": {"label": str, "total_value": float, "cash": float,
                "holdings": {"SYM": {"qty": float, "price": float}},
                "fills": [{"symbol", "side": "buy"|"sell", "qty", "notional"}],
                "realized": {"7d": float, "30d": float}}},        # optional
   "other_accounts_usd": float}                                  # optional manual total
"""

import json
import math
import sys
from datetime import date, datetime, timedelta, timezone

GOAL = 1_000_000.0
DEADLINE = date(2026, 12, 31)
BOARD = 'bot/team/scoreboard.jsonl'   # gitignored: holds the owner's other account (public repo)
MIN_FLOW = 0.50
MIN_PACE_DAYS = 3


def _ts(s):
    return datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)


def load_board(path=BOARD):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return []


def account_step(prev, cur):
    """Flow and performance P&L of one account between two snapshots."""
    if prev is None:
        return {'flow': 0.0, 'pnl': None}
    fills = cur.get('fills') or []
    trade_cash = sum(f['notional'] if f['side'] == 'sell' else -f['notional'] for f in fills)
    ext_cash = (cur['cash'] - prev['cash']) - trade_cash
    net_qty = {}
    for f in fills:
        net_qty[f['symbol']] = net_qty.get(f['symbol'], 0.0) + (f['qty'] if f['side'] == 'buy' else -f['qty'])
    ext_asset = 0.0
    for s in set(cur['holdings']) | set(prev['holdings']):
        q1 = cur['holdings'].get(s, {}).get('qty', 0.0)
        q0 = prev['holdings'].get(s, {}).get('qty', 0.0)
        px = cur['holdings'].get(s, {}).get('price') or prev['holdings'].get(s, {}).get('price') or 0.0
        ext_asset += ((q1 - q0) - net_qty.get(s, 0.0)) * px
    flow = ext_cash + ext_asset
    if abs(flow) < MIN_FLOW:
        flow = 0.0
    pnl = cur['total_value'] - prev['total_value'] - flow
    return {'flow': round(flow, 2), 'pnl': round(pnl, 2),
            'ext_cash': round(ext_cash, 2), 'ext_asset': round(ext_asset, 2)}


def make_record(spec, board):
    last = board[-1] if board else None
    accounts, team_v0, team_v1, team_flow, team_pnl, have_pnl = {}, 0.0, 0.0, 0.0, 0.0, False
    for key, cur in spec['accounts'].items():
        prev = last['accounts'].get(key) if last else None
        step = account_step(prev, cur)
        accounts[key] = {'label': cur.get('label', key), 'total_value': round(cur['total_value'], 2),
                         'cash': round(cur['cash'], 2), 'holdings': cur['holdings'],
                         'realized': cur.get('realized'), **step}
        team_v1 += cur['total_value']
        if prev is not None:
            team_v0 += prev['total_value']
            team_flow += step['flow']
            team_pnl += step['pnl']
            have_pnl = True
    r = None
    if have_pnl:
        base = team_v0 + 0.5 * team_flow   # Modified Dietz: flows assumed mid-period
        r = team_pnl / base if base > 0 else None
    return {'ts_utc': spec['ts_utc'], 'accounts': accounts,
            'other_accounts_usd': spec.get('other_accounts_usd', 0.0),
            'team': {'value': round(team_v1, 2), 'flow': round(team_flow, 2),
                     'pnl': round(team_pnl, 2) if have_pnl else None,
                     'r': round(r, 6) if r is not None else None}}


def window_stats(board, days):
    """Chained performance return, flows and span over steps starting within the trailing `days`."""
    if len(board) < 2:
        return None
    end = _ts(board[-1]['ts_utc'])
    first_idx = next((i for i in range(len(board) - 1)
                      if (end - _ts(board[i]['ts_utc'])).total_seconds() <= days * 86400 + 3600), None)
    if first_idx is None:
        return None
    recs = board[first_idx + 1:]
    span = (end - _ts(board[first_idx]['ts_utc'])).total_seconds() / 86400
    if span <= 0:
        return None
    twr = 1.0
    for b in recs:
        if b['team']['r'] is not None:
            twr *= 1 + b['team']['r']
    flows = sum(b['team']['flow'] for b in recs)
    return {'twr': twr - 1, 'span_days': span, 'daily': twr ** (1 / span) - 1,
            'flow_per_day': flows / span, 'pnl': sum(b['team']['pnl'] or 0 for b in recs)}


def days_to_goal(value, daily, flow_per_day=0.0, cap_days=365 * 60):
    """Days until value reaches GOAL at a daily rate plus a daily contribution."""
    if value >= GOAL:
        return 0
    if flow_per_day <= 0 and daily <= 0:
        return None
    if flow_per_day <= 0:
        return math.log(GOAL / value) / math.log(1 + daily)
    v = value
    for d in range(1, cap_days + 1):
        v = v * (1 + daily) + flow_per_day
        if v >= GOAL:
            return d
    return None


def required_daily(value, today, deadline=DEADLINE):
    n = (deadline - today).days
    return (GOAL / value) ** (1 / n) - 1 if n > 0 and value > 0 else None


def _eta(days, today):
    if days is None:
        return 'not closing in at this pace'
    if days > 365 * 60:
        return 'more than 60 years'
    d = today.toordinal() + int(math.ceil(days))
    when = date.fromordinal(d)
    return f"{days / 365.25:.1f} years (≈ {when:%b %Y})" if days > 365 else f"{days:.0f} days (≈ {when:%b %d, %Y})"


def report(board):
    if not board:
        return 'No snapshots yet.'
    last = board[-1]
    today = _ts(last['ts_utc']).date()
    tracked = last['team']['value']
    other = last.get('other_accounts_usd') or 0.0
    value = tracked + other
    lines = [f"**Team recap — {today:%a %b %d}**", '',
             f"**Team value: ${value:,.2f}** — {value / GOAL:.2%} of $1MM"]
    parts = [f"{a['label']} ${a['total_value']:,.2f}" for a in last['accounts'].values()]
    if other:
        parts.append(f"other accounts ${other:,.2f} (owner-reported)")
    lines.append('  ' + ' + '.join(parts))
    t = last['team']
    if t['pnl'] is not None:
        flow_txt = f"; deposits/transfers {t['flow']:+,.2f}" if t['flow'] else ''
        lines.append(f"Since last recap: performance {t['pnl']:+,.2f} ({(t['r'] or 0):+.2%}){flow_txt}")
        for a in last['accounts'].values():
            if a['pnl'] is not None:
                f_txt = f", flow {a['flow']:+,.2f}" if a['flow'] else ''
                lines.append(f"  {a['label']}: {a['pnl']:+,.2f}{f_txt}")
    lines.append('')
    lines.append('**Pace toward $1MM** (performance only, deposits excluded)')
    # One day of crypto moves extrapolated to a year is noise, so no average (or ETA)
    # until MIN_PACE_DAYS are tracked; longer windows appear once they differ.
    tracked = (_ts(last['ts_utc']) - _ts(board[0]['ts_utc'])).total_seconds() / 86400
    shown = False
    for label, days, need in (('7-day', 7, MIN_PACE_DAYS), ('30-day', 30, 7.25),
                              ('since tracking began', 3650, 30.25)):
        if tracked < need - 0.25:
            continue
        w = window_stats(board, days)
        if not w:
            continue
        shown = True
        eta = _eta(days_to_goal(value, w['daily']), today)
        line = (f"  {label} ({w['span_days']:.0f}d): {w['twr']:+.2%} total, "
                f"{w['daily']:+.2%}/day avg → $1MM in {eta}")
        if w['flow_per_day'] > 0:
            eta_c = _eta(days_to_goal(value, max(w['daily'], 0.0), w['flow_per_day']), today)
            line += f"; with your ${w['flow_per_day']:,.0f}/day deposits: {eta_c}"
        lines.append(line)
    if not shown:
        start = _ts(board[0]['ts_utc']).date() + timedelta(days=MIN_PACE_DAYS)
        lines.append(f"  Building history — {len(board)} snapshot(s) so far; "
                     f"the pace average starts {start:%a %b %d}.")
    req = required_daily(value, today)
    if req is not None:
        lines.append(f"  To reach $1MM by {DEADLINE:%b %d}: {req:+.2%} every day "
                     f"({(DEADLINE - today).days} days left)")
    rl = [f"{a['label']} 7d {a['realized']['7d']:+,.2f} / 30d {a['realized']['30d']:+,.2f}"
          for a in last['accounts'].values() if a.get('realized')]
    if rl:
        lines.append('')
        lines.append('**Realized P&L (broker-reported, closed trades only)**')
        lines.extend('  ' + x for x in rl)
    if len(board) >= 2:
        movers = []
        for key, a in last['accounts'].items():
            prev_h = board[-2]['accounts'].get(key, {}).get('holdings', {})
            for s, h in a['holdings'].items():
                p0 = prev_h.get(s, {}).get('price')
                if p0 and h.get('price'):
                    movers.append((h['price'] / p0 - 1, s))
        if movers:
            movers.sort(reverse=True)
            top = ', '.join(f"{s} {m:+.1%}" for m, s in movers[:3])
            low = ', '.join(f"{s} {m:+.1%}" for m, s in movers[-2:][::-1])
            lines.append('')
            lines.append(f"Movers since last recap: best {top}; worst {low}")
    return '\n'.join(lines)


def main():
    cmd = sys.argv[1]
    if cmd == 'snapshot':
        spec = json.load(open(sys.argv[2]))
        board = load_board()
        rec = make_record(spec, board)
        import os
        os.makedirs('bot/team', exist_ok=True)
        with open(BOARD, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(json.dumps(rec['team']))
    elif cmd == 'report':
        print(report(load_board()))
    else:
        raise SystemExit(f'unknown command {cmd}')


if __name__ == '__main__':
    main()
