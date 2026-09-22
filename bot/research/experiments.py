"""
Variant grid for the replication backtest (owner request 2026-09-22:
"backtest items 1-5 to find our path forward").

The variants below were fixed BEFORE any result was seen, so the grid is a
test of the five proposals as stated, not a search for the best-looking
number. The one row that is chosen after the fact ('best observed') is
labelled as such, because anything picked by looking at results on this
window is optimistically biased on this window.

Items, as proposed:
  1  charge costs inside the validation gate
  2  cut the universe to cost-surviving names
  3  concentrate on the top-expectancy tier
  4  escape T+1 (margin account, 1x — no borrowing)
  5  shorts (margin) / inverse ETFs (cash-account-compatible)
plus: universe expansion, leverage, slippage, and the time-stop check.

    python -m bot.research.experiments <sigdir> <outfile>
"""

import json
import pickle
import sys

import numpy as np
import pandas as pd

from bot.research.simulate import Params, load_signals, prepare, project, run

BASE = ('IBIT ETHA COIN MSTR NVDA TSLA QQQ SPY ONEQ DIA XRP XRPC XRPZ TOXR GXRP '
        'XRPR XRPI XXRP UXRP XRPT MSFT GOOGL META AMD USO XLE UUP TLT JPM V WMT AAPL AMZN').split()
ADD_SINGLE = 'PLTR AVGO NFLX MU SMCI HOOD MARA RIOT SHOP ARM CRWD UBER'.split()
ADD_ETF = 'IWM SMH XLF GLD SLV ARKK KRE'.split()
ADD_LEV = 'TQQQ SOXL TSLL NVDL CONL'.split()
ADD_INV = 'SQQQ SOXS SH PSQ TSLQ SBIT NVD SPXU'.split()
EXPANDED = BASE + ADD_SINGLE + ADD_ETF + ADD_LEV + ADD_INV

OOS_START = '2026-02-20'
SEL_END, TEST_START = '2026-03-19', '2026-03-20'   # item 2 static: select, then test


def static_selection(allsig, symbols, cost=0.002):
    """Item 2 in its static form, chosen honestly: names whose mean net OOS
    expectancy over the SELECTION month was positive. Tested only after it."""
    s = allsig[(allsig['date'] >= pd.Timestamp(OOS_START).date())
               & (allsig['date'] <= pd.Timestamp(SEL_END).date())]
    daily = s.groupby(['sym', 'date'])['gross_both'].first().reset_index()
    m = daily.groupby('sym')['gross_both'].mean() - cost
    return tuple(sorted(x for x in m.index if x in symbols and m[x] > 0)), m


def grid(sel_base):
    b = dict(universe=tuple(BASE), start=OOS_START)
    stack1 = dict(b, gate='net')
    stack2 = dict(stack1, entry_gate='validated')
    stack3 = dict(stack2, entry_gate='tier', tier_min=0.005)
    stack4 = dict(stack3, account='margin')
    stack5 = dict(stack4, shorts=True)
    return [
        # ── reference points
        Params('V0  live replica', **b),
        Params('V0t live + 14h time stop', time_stop_hours=14, **b),
        Params('V0x live, no expectancy block', expectancy_block=False, **b),
        # ── each item alone, against V0
        Params('I1  cost-aware ML gate', **dict(b, gate='net')),
        Params('I1L cost-aware long-side gate', **dict(b, gate='net_long')),
        Params('I2  enter validated names only', **dict(b, gate='net', entry_gate='validated')),
        Params('I3a tier >= 0.5% net', **dict(b, gate='net', entry_gate='tier', tier_min=0.005)),
        Params('I3b tier >= 1.0% net', **dict(b, gate='net', entry_gate='tier', tier_min=0.010)),
        Params('I4  margin 1x (no T+1)', **dict(b, account='margin')),
        Params('I5a shorts (margin)', **dict(b, account='margin', shorts=True)),
        Params('I5b + inverse ETFs (cash)', **dict(b, universe=tuple(BASE + ADD_INV))),
        # ── cumulative stack, items applied in order
        Params('S1  +1', **stack1),
        Params('S2  +1+2', **stack2),
        Params('S3  +1+2+3', **stack3),
        Params('S4  +1+2+3+4', **stack4),
        Params('S5  +1+2+3+4+5', **stack5),
        # ── universe expansion
        Params('E0  live, 65 symbols', **dict(b, universe=tuple(EXPANDED))),
        Params('E3  S3 on 65 symbols', **dict(stack3, universe=tuple(EXPANDED))),
        Params('E5  S5 on 65 symbols', **dict(stack5, universe=tuple(EXPANDED))),
        # ── leverage and costs, on the full stack
        Params('L2  S5 at 2x leverage', **dict(stack5, leverage=2.0)),
        Params('L2e E5 at 2x leverage', **dict(stack5, universe=tuple(EXPANDED), leverage=2.0)),
        Params('C2  S5 at 0.2%/side slip', **dict(stack5, slippage=0.002)),
    ], [
        # item 2, static, honest split: selected on Feb20-Mar19, tested after
        Params('T0  live replica (test window)', universe=tuple(BASE), start=TEST_START),
        Params('T2  static cost-survivors', universe=tuple(BASE), start=TEST_START,
               static_universe=sel_base),
    ]


def row(r, proj):
    return {k: r[k] for k in ('name', 'days', 'end', 'ret', 'h1', 'h2', 'max_dd', 'trades',
                               'win_rate', 'mean_trade', 'pnl_usd', 'shorts', 'exposure',
                               'pdt_violations', 'blocked_by_expectancy', 'by_reason')} | {
        'proj': proj}


def main():
    sigdir, outfile = sys.argv[1], sys.argv[2]
    allsig = load_signals(sigdir, EXPANDED)
    prepared = prepare(allsig)
    sel_base, sel_scores = static_selection(allsig, BASE)
    main_grid, split_grid = grid(sel_base)
    results = []
    for p in main_grid + split_grid:
        r = run(prepared, p)
        results.append(row(r, project(r['daily_returns'])))
        print(f"{p.name:34} end ${r['end']:>10,.2f}  {r['ret']:+8.2%}  dd {r['max_dd']:+7.2%}  "
              f"trades {r['trades']:4}", flush=True)
        pickle.dump(r, open(f"{outfile}.{p.name.split()[0]}.pkl", 'wb'))
    json.dump({'static_selection': list(sel_base),
               'selection_scores': {k: float(v) for k, v in sel_scores.items()},
               'results': results}, open(outfile, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
