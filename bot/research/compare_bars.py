"""
Re-validate the deployed strategy on corrected bars (2026-09-23).

The replication backtest ran on the broker's hourly bars, which are
incomplete ~29% of the time and omit the 9:30 open. This replays the
deployed configuration (cost-aware gate, validated-only entries, 3-day
expectancy cooldown, 65 symbols) on both bar sources over the SAME window —
the span where real 5-minute history exists — so the only difference
between the rows is the data.

    python -m bot.research.compare_bars <sig_hourly_dir> <sig_5min_dir> [<sig_5min_clock_dir>]

The optional third source rebuilds complete hourly bars from 5-minute data
on the broker's own clock grid (10:00…15:00 ET, opening half hour left out),
which separates the two changes the 9:30 aggregation makes at once:
completeness and anchoring.
"""

import sys

from bot.research.experiments import BASE, EXPANDED
from bot.research.simulate import Params, load_signals, prepare, project, run


def main():
    hourly_dir, five_dir = sys.argv[1], sys.argv[2]
    s5 = load_signals(five_dir, EXPANDED)
    start = str(s5['date'].min())
    end = str(s5['date'].max())
    days = sorted(s5['date'].unique())
    mid = str(days[len(days) // 2])
    preps = {'broker hourly': prepare(load_signals(hourly_dir, EXPANDED)),
             '5-min aggregated': prepare(s5)}
    if len(sys.argv) > 3:
        preps['5-min, clock grid'] = prepare(load_signals(sys.argv[3], EXPANDED))
    print(f'common window {start} -> {end} ({len(days)} trading days), halves split at {mid}\n')

    deployed = dict(universe=tuple(EXPANDED), gate='net', entry_gate='validated',
                    expectancy_block=True, expectancy_expiry_days=3)
    live_before = dict(universe=tuple(BASE), expectancy_block=True)
    rows = [('deployed config', deployed), ('pre-change engine', live_before)]

    print(f"{'config':20}{'bars':18}{'return':>9}{'H1':>8}{'H2':>8}{'@0.2%':>8}{'maxDD':>8}"
          f"{'trades':>7}{'win%':>6}{'median 70d':>12}{'P(loss)':>8}")
    for label, kw in rows:
        for src, prep in preps.items():
            base = dict(kw, start=start, end=end)
            f = run(prep, Params(label, **base))
            a = run(prep, Params(label, **dict(base, end=mid)))
            b = run(prep, Params(label, **dict(base, start=mid)))
            c = run(prep, Params(label, **dict(base, slippage=0.002)))
            p = project(f['daily_returns'])
            print(f"{label:20}{src:18}{f['ret']:+9.1%}{a['ret']:+8.1%}{b['ret']:+8.1%}{c['ret']:+8.1%}"
                  f"{f['max_dd']:+8.1%}{f['trades']:7}{(f['win_rate'] or 0) * 100:6.0f}"
                  f"{p['median'] * 1441.11 if p else float('nan'):12,.0f}{p['p_loss'] if p else float('nan'):8.0%}")
        print()


if __name__ == '__main__':
    main()
