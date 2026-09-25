"""
Tests for the crypto history fetcher (2026-09-23). No network: a fake
exchange stands in for ccxt.

    python -m bot.tools.test_crypto_history
"""

import json

from bot.strategy.bars import clean_rows

from .crypto_history import HOUR_MS, HistoryError, build, fetch_hourly, pair_for

T0 = 1_790_000_000_000 - (1_790_000_000_000 % HOUR_MS)  # an hour boundary


class FakeExchange:
    """Serves `n` consecutive hourly bars from T0, `page` at a time."""

    def __init__(self, n=10, page=4, fail=False, ignore_since=False):
        self.n, self.page, self.fail, self.ignore_since = n, page, fail, ignore_since
        self.calls = 0

    def fetch_ohlcv(self, pair, tf, since=None, limit=None):
        self.calls += 1
        if self.fail:
            raise ConnectionError('403 Forbidden')
        assert tf == '1h'
        start = T0 if (self.ignore_since or since is None) else max(since, T0)
        bars = []
        t = start
        while t < T0 + self.n * HOUR_MS and len(bars) < min(limit, self.page):
            i = (t - T0) // HOUR_MS
            bars.append([t, 100 + i, 101 + i, 99 + i, 100.5 + i, 10.0])
            t += HOUR_MS
        return bars


def test_pages_until_caught_up():
    ex = FakeExchange(n=10, page=4)
    now = T0 + 10 * HOUR_MS              # all 10 bars closed
    bars = fetch_hourly(ex, 'BTC/USD', days=1, now_ms=now)
    assert len(bars) == 10, len(bars)
    assert ex.calls >= 3, ex.calls
    print('ok  pages forward across several calls and returns every closed bar')


def test_forming_bar_dropped():
    ex = FakeExchange(n=10, page=20)
    now = T0 + 9 * HOUR_MS + 1_800_000   # bar 9 is half-formed
    bars = fetch_hourly(ex, 'BTC/USD', days=1, now_ms=now)
    assert len(bars) == 9, len(bars)
    print('ok  the hour still forming is dropped, not passed off as a full bar')


def test_failure_raises_never_simulates():
    try:
        fetch_hourly(FakeExchange(fail=True), 'BTC/USD', days=1, now_ms=T0 + 5 * HOUR_MS)
    except HistoryError as e:
        assert '403' in str(e)
        print('ok  a failed fetch raises (the scanner fell back to simulated bars)')
        return
    raise AssertionError('expected HistoryError')


def test_exchange_ignoring_since_terminates():
    ex = FakeExchange(n=3, page=3, ignore_since=True)
    bars = fetch_hourly(ex, 'BTC/USD', days=1, now_ms=T0 + 3 * HOUR_MS)
    assert len(bars) == 3 and ex.calls <= 3, (len(bars), ex.calls)
    print('ok  an exchange that ignores `since` cannot trap the fetch in a loop')


def test_pair_mapping():
    cfg = {'pair_template': '{base}/USD', 'pair_overrides': {'DOGE': 'DOGE/USDT'}}
    assert pair_for('BTC', cfg) == 'BTC/USD'
    assert pair_for('DOGE', cfg) == 'DOGE/USDT'
    print('ok  Robinhood symbols map to exchange pairs, with per-symbol overrides')


def test_output_is_snapshot_compatible():
    cfg = {'universe': ['BTC', 'ETH'], 'pair_template': '{base}/USD'}
    payload = build(cfg, days=1, exchange=FakeExchange(n=6, page=10), now_ms=T0 + 6 * HOUR_MS)
    json.loads(json.dumps(payload))
    for r in payload['data']['results']:
        df = clean_rows(r['bars'])
        assert len(df) == 6 and df['close'].iloc[-1] == 105.5, df.tail(1)
    print('ok  output parses with the same bar loader the equity snapshot uses')


def test_config_block_is_dormant():
    cfg = json.load(open('bot/config.json'))
    assert cfg['crypto']['enabled'] is False
    print('ok  config.crypto ships disabled')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('\nall crypto history tests passed')
