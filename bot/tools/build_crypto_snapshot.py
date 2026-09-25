"""
Build the crypto lane's run_cycle input from a crypto_history file (2026-09-23).

The crypto lane runs the SAME engine as equities (bot/strategy/engine.py),
with the differences expressed as data rather than a second code path:

  * universe      config.crypto.universe; no equity expansion list.
  * overrides     config.crypto.strategy / config.crypto.risk are merged over
                  the equity blocks (e.g. a higher validation cost for the
                  wider crypto spread).
  * settlement    instant_settlement=True: no GFV deferral of same-day exits.
  * regime        BTC daily bars resampled from the hourly series, in place
                  of SPY daily.
  * sleeve        sizing sees only the crypto sleeve: sleeve_pct of total
                  account value. Buying power is capped so the lane can never
                  spend past its sleeve, whatever the account holds in cash.
  * books         its own state and trade log, so the two lanes never write
                  the same files.

    python -m bot.tools.build_crypto_snapshot <spec.json>

spec.json:
  {"today_et": "YYYY-MM-DD",
   "history_file": path,              # bot.tools.crypto_history output
   "total_equity": float,             # get_portfolio total_value
   "buying_power": float,             # crypto_buying_power
   "positions": [{symbol, quantity, shares_available_for_sells, average_buy_price}],
   "quotes": {symbol: price}}
"""

import json
import sys

import pandas as pd

from bot.strategy.bars import clean_rows, to_snapshot_rows

REPO_CONFIG = 'bot/config.json'
CRYPTO_STATE = 'bot/crypto_state.json'
CRYPTO_LOG = 'bot/crypto_trade_log.jsonl'


def merged_config(config: dict) -> dict:
    c = config['crypto']
    out = json.loads(json.dumps(config))
    out['strategy'].update(c.get('strategy') or {})
    out['strategy']['instant_settlement'] = True
    out['risk'].update(c.get('risk') or {})
    out['universe'] = list(c['universe'])
    out['universe_expansion'] = []
    out['universe_expansion_enabled'] = False
    return out


def sleeve(total_equity: float, positions: list, quotes: dict, buying_power: float, pct: float):
    """(sleeve_equity, spendable) for the crypto sleeve."""
    target = total_equity * pct / 100.0
    held = sum(float(p['quantity']) * float(quotes.get(p['symbol'], p['average_buy_price']))
               for p in positions)
    spendable = max(0.0, min(buying_power, target - held))
    return target, spendable


def btc_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    d = hourly.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                   'close': 'last', 'volume': 'sum'}).dropna()
    return d.iloc[:-1]  # today's day is still forming


def load_json(path, default):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return default


def load_log(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return []


def build(spec: dict, config: dict, state: dict, log: list):
    c = config['crypto']
    raw = json.load(open(spec['history_file']))['data']['results']
    frames = {r['symbol']: clean_rows(r['bars']) for r in raw}
    bars = {s: to_snapshot_rows(df) for s, df in frames.items() if not df.empty}
    regime = btc_daily(frames['BTC']) if 'BTC' in frames else None

    sleeve_equity, spendable = sleeve(spec['total_equity'], spec['positions'],
                                      spec.get('quotes', {}), spec['buying_power'],
                                      float(c['sleeve_pct']))
    snapshot = {
        'today_et': spec['today_et'],
        'config': merged_config(config),
        'state': state,
        'portfolio': {'equity': sleeve_equity, 'buying_power': spendable},
        'positions': spec['positions'],
        'trade_history': log,
        'bars': bars,
        'regime_bars': to_snapshot_rows(regime.iloc[-120:]) if regime is not None else [],
        'quotes': spec.get('quotes', {}),
    }
    report = {'lane': 'crypto', 'symbols': len(bars), 'sleeve_equity': round(sleeve_equity, 2),
              'spendable': round(spendable, 2),
              'latest_bar': max((v[-1]['t'] for v in bars.values()), default=None)}
    return snapshot, report


def main():
    spec = json.load(open(sys.argv[1]))
    config = json.load(open(REPO_CONFIG))
    if not config.get('crypto', {}).get('enabled'):
        print(json.dumps({'lane': 'crypto', 'skipped': 'config.crypto.enabled is false'}))
        return
    snapshot, report = build(spec, config, load_json(CRYPTO_STATE, {}), load_log(CRYPTO_LOG))
    json.dump(snapshot, open('crypto_input.json', 'w'))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
