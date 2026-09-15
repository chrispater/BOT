"""
CLI wrapper. Run from the repo root:

    python -m bot.strategy.run_cycle <input.json> <output.json>

Input snapshot schema (built by the agent session from MCP tool results):
{
  "today_et": "YYYY-MM-DD",
  "config": {...},                     # contents of bot/config.json
  "state": {...},                      # contents of bot/state.json
  "portfolio": {"equity": 50.0, "buying_power": 48.0},
  "positions": [{"symbol": "IBIT", "quantity": "0.21",
                 "shares_available_for_sells": "0.21",
                 "average_buy_price": "58.90"}],
  "trade_history": [{"event": "sell", "pnl_pct": 1.2, ...}, ...],  # trade_log.jsonl lines
  "bars": {"IBIT": [{"t": "2026-07-01T14:30:00Z", "o": 1, "h": 1,
                     "l": 1, "c": 1, "v": 100}, ...], ...},        # hourly, oldest first
  "regime_bars": [{...}]                                           # SPY daily, oldest first
}

Output: decisions dict from engine.run() — exits, entries, halt,
state_updates, diagnostics.
"""

import json
import sys

import numpy as np
import pandas as pd

from . import engine


def bars_to_df(rows):
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.rename(columns={'t': 'ts', 'o': 'open', 'h': 'high',
                            'l': 'low', 'c': 'close', 'v': 'volume'})
    for col in ('open', 'high', 'low', 'close', 'volume'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df = df.dropna(subset=['close']).set_index('ts').sort_index()
    return df


def main():
    np.random.seed(42)
    inp, outp = sys.argv[1], sys.argv[2]
    with open(inp) as f:
        snapshot = json.load(f)

    now = pd.Timestamp.now(tz='UTC')
    bars = {}
    for sym, rows in snapshot.get('bars', {}).items():
        df = bars_to_df(rows)
        if df is None or df.empty:
            continue
        # Drop the still-forming hourly candle — its close/high/low aren't final
        if now - df.index[-1] < pd.Timedelta(minutes=55):
            df = df.iloc[:-1]
        bars[sym] = df
    snapshot['bars'] = bars
    snapshot['regime_bars'] = bars_to_df(snapshot.get('regime_bars'))

    decisions = engine.run(snapshot)

    with open(outp, 'w') as f:
        json.dump(decisions, f, indent=2, default=str)
    print(json.dumps({'exits': len(decisions['exits']),
                      'entries': len(decisions['entries']),
                      'halt': decisions['halt']}))


if __name__ == '__main__':
    main()
