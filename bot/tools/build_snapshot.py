"""
Build run_cycle's input.json from saved MCP tool-result files.

The historicals responses exceed the tool-result token cap and land on disk,
so each cycle's snapshot is stitched together here rather than read back
into context. This used to live outside the repository, which is how it
shipped without a filter for the broker's synthetic filler bars; it is
versioned now so the data path is reviewed and tested like the engine.

    python -m bot.tools.build_snapshot <spec.json>

spec.json:
  {"today_et": "YYYY-MM-DD",
   "bar_files": [path, ...],      # get_equity_historicals results, any interval
   "regime_file": path,           # SPY daily results
   "equity": float, "buying_power": float,
   "positions": [{symbol, quantity, shares_available_for_sells, average_buy_price}],
   "quotes": {symbol: last_trade_price},
   "now_utc": optional ISO time, for tests}

`strategy.bar_source` in config.json selects the bar path:
  "hour"    — the broker's hourly bars, as fetched (filler removed).
  "5minute" — 5-minute bars aggregated here into 9:30-anchored hourly bars
              (see bot/strategy/bars.py for why), with any bucket that has
              not finished yet dropped.
"""

import json
import sys
from collections import defaultdict

import pandas as pd

from bot.strategy.bars import aggregate_hourly, clean_rows, to_snapshot_rows

REPO_CONFIG = 'bot/config.json'
REPO_STATE = 'bot/state.json'
REPO_LOG = 'bot/trade_log.jsonl'


def completed(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Drop aggregated buckets that have not closed yet (15:30 is 30 minutes)."""
    if df.empty:
        return df
    et = df.index.tz_convert('America/New_York')
    dur = pd.to_timedelta([30 if (t.hour, t.minute) == (15, 30) else 60 for t in et], unit='min')
    return df[(df.index + dur) <= now]


def build(spec: dict, config: dict) -> dict:
    source = config['strategy'].get('bar_source', 'hour')
    now = pd.Timestamp(spec.get('now_utc') or pd.Timestamp.now(tz='UTC'))
    if now.tzinfo is None:
        now = now.tz_localize('UTC')

    raw = defaultdict(list)
    filler = defaultdict(int)
    for path in spec['bar_files']:
        for r in json.load(open(path))['data']['results']:
            bars = r.get('bars') or []
            filler[r['symbol']] += sum(1 for b in bars if b.get('interpolated'))
            raw[r['symbol']].extend(bars)

    bars = {}
    for sym, rows in raw.items():
        df = clean_rows(rows)
        if source == '5minute':
            df = completed(aggregate_hourly(df), now)
        if not df.empty:
            bars[sym] = to_snapshot_rows(df)

    regime = clean_rows(json.load(open(spec['regime_file']))['data']['results'][0]['bars'])

    snapshot = {
        'today_et': spec['today_et'],
        'config': config,
        'state': json.load(open(REPO_STATE)),
        'portfolio': {'equity': spec['equity'], 'buying_power': spec['buying_power']},
        'positions': spec['positions'],
        'trade_history': [json.loads(l) for l in open(REPO_LOG) if l.strip()],
        'bars': bars,
        'regime_bars': to_snapshot_rows(regime.iloc[-120:]),
        'quotes': spec.get('quotes', {}),
    }
    report = {'bar_source': source, 'symbols': len(bars),
              'filler_dropped': {s: n for s, n in filler.items() if n},
              'latest_bar': max((v[-1]['t'] for v in bars.values()), default=None)}
    return snapshot, report


def main():
    spec = json.load(open(sys.argv[1]))
    snapshot, report = build(spec, json.load(open(REPO_CONFIG)))
    json.dump(snapshot, open('input.json', 'w'))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
