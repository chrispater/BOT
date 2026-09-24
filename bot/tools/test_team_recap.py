"""
Tests for the team scoreboard (2026-09-24): deposits and transfers must never
be counted as performance, and trades must never be counted as deposits.

    python -m bot.tools.test_team_recap
"""

from datetime import date

from .team_recap import (GOAL, account_step, days_to_goal, make_record, report,
                         required_daily, window_stats)


def _acct(cash, holdings, total=None, fills=None):
    tv = total if total is not None else cash + sum(h['qty'] * h['price'] for h in holdings.values())
    return {'label': 'x', 'total_value': tv, 'cash': cash, 'holdings': holdings, 'fills': fills or []}


def test_cash_deposit_is_flow_not_performance():
    prev = _acct(1.0, {'COIN': {'qty': 1.0, 'price': 200.0}})
    cur = _acct(101.0, {'COIN': {'qty': 1.0, 'price': 200.0}})
    s = account_step(prev, cur)
    assert s['flow'] == 100.0 and s['pnl'] == 0.0, s
    print('ok  a $100 deposit is a flow, not a $100 gain')


def test_trade_is_neither_flow_nor_pnl():
    prev = _acct(1000.0, {})
    cur = _acct(900.0, {'XRP': {'qty': 50.0, 'price': 2.0}},
                fills=[{'symbol': 'XRP', 'side': 'buy', 'qty': 50.0, 'notional': 100.0}])
    s = account_step(prev, cur)
    assert s['flow'] == 0.0 and s['pnl'] == 0.0, s
    print('ok  buying $100 of a coin is neither a deposit nor a gain')


def test_price_move_is_performance():
    prev = _acct(0.0, {'SOL': {'qty': 10.0, 'price': 100.0}})
    cur = _acct(0.0, {'SOL': {'qty': 10.0, 'price': 110.0}})
    s = account_step(prev, cur)
    assert s['flow'] == 0.0 and s['pnl'] == 100.0, s
    print('ok  a price move on held coins is performance')


def test_coin_transfer_in_is_flow():
    prev = _acct(0.0, {'QNT': {'qty': 1.0, 'price': 100.0}})
    cur = _acct(0.0, {'QNT': {'qty': 3.0, 'price': 100.0}})
    s = account_step(prev, cur)
    assert s['flow'] == 200.0 and s['pnl'] == 0.0, s
    print('ok  coins transferred in with no order are a flow at current price')


def test_profit_taking_sale_keeps_gain_as_performance():
    prev = _acct(0.0, {'ONDO': {'qty': 100.0, 'price': 1.0}})
    cur = _acct(130.0, {}, fills=[{'symbol': 'ONDO', 'side': 'sell', 'qty': 100.0, 'notional': 130.0}])
    s = account_step(prev, cur)
    assert s['flow'] == 0.0 and s['pnl'] == 30.0, s
    print('ok  selling after a +30% move books +$30 performance, $0 flow')


def test_transfer_between_accounts_nets_to_zero_for_team():
    b0 = make_record({'ts_utc': '2026-09-23T12:30:00Z', 'accounts': {
        'a': _acct(500.0, {}), 'b': _acct(1.0, {})}}, [])
    b1 = make_record({'ts_utc': '2026-09-24T12:30:00Z', 'accounts': {
        'a': _acct(400.0, {}), 'b': _acct(101.0, {})}}, [b0])
    assert b1['accounts']['a']['flow'] == -100.0 and b1['accounts']['b']['flow'] == 100.0
    assert b1['team']['flow'] == 0.0 and b1['team']['pnl'] == 0.0 and b1['team']['r'] == 0.0
    print('ok  moving $100 between the two accounts is zero flow and zero gain for the team')


def test_pace_chains_returns_and_ignores_deposits():
    # 10 coins at $100 rising 1%/day; a $500 cash deposit on day 23 stays in cash.
    board = [make_record({'ts_utc': '2026-09-20T12:30:00Z',
                          'accounts': {'a': _acct(0.0, {'X': {'qty': 10.0, 'price': 100.0}})}}, [])]
    px, cash = 100.0, 0.0
    for d in range(21, 25):
        px *= 1.01
        cash += 500.0 if d == 23 else 0.0
        board.append(make_record({'ts_utc': f'2026-09-{d}T12:30:00Z',
                                  'accounts': {'a': _acct(cash, {'X': {'qty': 10.0, 'price': px}})}}, board))
    w = window_stats(board, 7)
    assert abs(w['span_days'] - 4.0) < 1e-9, w
    assert abs(w['flow_per_day'] - 125.0) < 1e-6, w
    # Idle deposited cash dilutes the pace a little (Modified Dietz) but the $500 itself is never growth.
    assert 0.007 < w['daily'] < 0.0101, w
    assert w['pnl'] < 50, w
    print(f"ok  pace = {w['daily']:.3%}/day from four +1% days; the $500 deposit is not counted as growth")


def test_goal_math():
    n = days_to_goal(10_000.0, 0.01)
    assert abs((1.01 ** n) * 10_000.0 - GOAL) < 1.0
    assert days_to_goal(10_000.0, -0.001) is None
    assert days_to_goal(10_000.0, 0.0, 1000.0) == 990
    r = required_daily(10_000.0, date(2026, 12, 21))
    assert abs((1 + r) ** 10 * 10_000.0 - GOAL) < 1e-3
    print('ok  time-to-goal and required-daily-rate math')


def test_report_first_day_has_no_pace_claims():
    board = [make_record({'ts_utc': '2026-09-24T16:30:00Z', 'accounts': {'a': _acct(7000.0, {})}}, [])]
    text = report(board)
    assert 'Building history' in text and 'years' not in text.split('Building history')[0], text
    print('ok  day one reports value and the required pace, but claims no average yet')


def test_no_average_until_three_days_then_seven_day_line():
    two = [make_record({'ts_utc': '2026-09-24T12:30:00Z', 'accounts': {'a': _acct(7000.0, {})}}, [])]
    two.append(make_record({'ts_utc': '2026-09-25T12:30:00Z', 'accounts': {'a': _acct(7000.0, {})}}, two))
    text = report(two)
    assert 'Building history' in text and '→ $1MM' not in text, text
    board = [make_record({'ts_utc': '2026-09-20T12:30:00Z',
                          'accounts': {'a': _acct(0.0, {'X': {'qty': 10.0, 'price': 100.0}})}}, [])]
    for d in range(21, 25):
        board.append(make_record({'ts_utc': f'2026-09-{d}T12:30:00Z', 'accounts': {
            'a': _acct(0.0, {'X': {'qty': 10.0, 'price': 100.0 * 1.01 ** (d - 20)}})}}, board))
    text = report(board)
    assert '7-day (4d)' in text and '30-day' not in text and 'Building history' not in text, text
    print('ok  one day of moves is never extrapolated; the 7-day average appears after 3 days')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('\nall team recap tests passed')
