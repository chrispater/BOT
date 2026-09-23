"""
Hourly crypto OHLCV history for the crypto lane (2026-09-23).

Robinhood's connector has live crypto quotes but no crypto bar history, so
history comes from a public exchange feed via ccxt — the fetch approach used
by the owner's CryptoQuantScanner (backend/trading_service.py). Only the
fetch is borrowed; every trading decision stays in bot/strategy/.

Two deliberate differences from the scanner's fetcher:
  * The exchange is configurable (`crypto.history_exchange`), because the
    scanner's BloFin perpetuals and a spot USD venue price differently.
  * A failed fetch RAISES. The scanner silently substitutes simulated bars on
    any error; a live trading loop must never trade on invented prices.

Output is written in the same shape as a get_equity_historicals result file
({"data": {"results": [{"symbol", "bars": [...]}]}}, broker field names), so
bot/tools/build_snapshot.py consumes it unchanged.

    python -m bot.tools.crypto_history <out.json> [days]
"""

import json
import sys
import time

import pandas as pd

HOUR_MS = 3_600_000


class HistoryError(RuntimeError):
    pass


def make_exchange(name: str):
    import ccxt
    if name not in ccxt.exchanges:
        raise HistoryError(f'unknown ccxt exchange {name!r}')
    ex = getattr(ccxt, name)({'enableRateLimit': True})
    # ccxt disables trust_env on its requests session, which bypasses the
    # HTTPS_PROXY / REQUESTS_CA_BUNDLE the cloud environment routes egress
    # through; the direct path is refused even for allowlisted hosts.
    ex.session.trust_env = True
    return ex


def fetch_hourly(exchange, pair: str, days: int, now_ms: int = None, page: int = 300) -> list:
    """Closed hourly bars for `pair` over the last `days`, oldest first.

    Pages forward from `since` until caught up. The bar still forming at
    `now_ms` is dropped: a partial bar would read as a full hour to the engine.
    """
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    since = now_ms - days * 24 * HOUR_MS
    rows = {}
    while since < now_ms:
        try:
            batch = exchange.fetch_ohlcv(pair, '1h', since=since, limit=page)
        except Exception as e:  # network, rate limit, unknown pair
            raise HistoryError(f'{pair}: fetch failed: {type(e).__name__}: {e}') from e
        if not batch:
            break
        for t, o, h, l, c, v in batch:
            rows[int(t)] = (o, h, l, c, v)
        last = int(batch[-1][0])
        if last + HOUR_MS <= since:  # exchange ignored `since`; avoid looping forever
            break
        since = last + HOUR_MS
    closed = sorted(t for t in rows if t + HOUR_MS <= now_ms)
    if not closed:
        raise HistoryError(f'{pair}: no closed bars returned')
    return [{'begins_at': pd.Timestamp(t, unit='ms', tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
             'open_price': str(rows[t][0]), 'high_price': str(rows[t][1]),
             'low_price': str(rows[t][2]), 'close_price': str(rows[t][3]),
             'volume': float(rows[t][4] or 0.0), 'session': '24_7'} for t in closed]


def pair_for(symbol: str, crypto_cfg: dict) -> str:
    """Robinhood symbol (e.g. 'BTC') -> exchange pair (e.g. 'BTC/USD')."""
    overrides = crypto_cfg.get('pair_overrides') or {}
    return overrides.get(symbol) or crypto_cfg['pair_template'].format(base=symbol)


def build(crypto_cfg: dict, days: int, exchange=None, now_ms: int = None) -> dict:
    exchange = exchange or make_exchange(crypto_cfg['history_exchange'])
    results = []
    for sym in crypto_cfg['universe']:
        bars = fetch_hourly(exchange, pair_for(sym, crypto_cfg), days, now_ms=now_ms)
        results.append({'symbol': sym, 'interval': 'hour', 'bounds': '24_7', 'bars': bars})
    return {'data': {'results': results}}


def main():
    out = sys.argv[1]
    cfg = json.load(open('bot/config.json'))['crypto']
    days = int(sys.argv[2]) if len(sys.argv) > 2 else int(cfg.get('history_days', 60))
    payload = build(cfg, days)
    json.dump(payload, open(out, 'w'))
    for r in payload['data']['results']:
        print(f"{r['symbol']:6} {len(r['bars']):5} bars  last {r['bars'][-1]['begins_at']}")


if __name__ == '__main__':
    main()
