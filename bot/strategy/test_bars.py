"""
Tests for bar hygiene and the 5-minute -> hourly aggregation (2026-09-23).

    python -m bot.strategy.test_bars
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd

from .bars import aggregate_hourly, clean_rows, to_snapshot_rows


def _five_min_session(day: str, base=100.0, vol=1000):
    """A full regular session of 5-minute bars (78), as broker rows."""
    start = pd.Timestamp(f'{day} 09:30', tz='America/New_York')
    rows = []
    for i in range(78):
        t = start + pd.Timedelta(minutes=5 * i)
        o = base + i * 0.1
        rows.append({'begins_at': t.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
                     'open_price': f'{o:.4f}', 'close_price': f'{o + 0.05:.4f}',
                     'high_price': f'{o + 0.2:.4f}', 'low_price': f'{o - 0.2:.4f}',
                     'volume': vol, 'session': 'reg'})
    return rows


def test_seven_buckets_anchored_at_open():
    h = aggregate_hourly(clean_rows(_five_min_session('2026-09-16')))
    et = [t.strftime('%H:%M') for t in h.index.tz_convert('America/New_York')]
    assert et == ['09:30', '10:30', '11:30', '12:30', '13:30', '14:30', '15:30'], et
    assert list(h['parts']) == [12, 12, 12, 12, 12, 12, 6], list(h['parts'])
    print('ok  a session aggregates to 7 buckets from 9:30 ET; the last is the 30-minute close')


def test_ohlcv_is_exact():
    rows = _five_min_session('2026-09-16')
    h = aggregate_hourly(clean_rows(rows))
    first = h.iloc[0]
    parts = rows[:12]
    assert abs(first['open'] - float(parts[0]['open_price'])) < 1e-9
    assert abs(first['close'] - float(parts[-1]['close_price'])) < 1e-9
    assert abs(first['high'] - max(float(p['high_price']) for p in parts)) < 1e-9
    assert abs(first['low'] - min(float(p['low_price']) for p in parts)) < 1e-9
    assert first['volume'] == 12 * 1000
    assert h['volume'].sum() == 78 * 1000, 'every 5-minute part lands in exactly one bucket'
    print('ok  open/high/low/close/volume are exact and every part is counted once')


def test_anchoring_survives_dst():
    """Winter (EST, UTC-5) and summer (EDT, UTC-4) must both anchor at 9:30 ET."""
    for day, utc_open in (('2026-01-15', '14:30'), ('2026-07-15', '13:30')):
        h = aggregate_hourly(clean_rows(_five_min_session(day)))
        assert h.index[0].strftime('%H:%M') == utc_open, (day, h.index[0])
        assert h.index[0].tz_convert('America/New_York').strftime('%H:%M') == '09:30'
    print('ok  9:30 ET anchoring holds across daylight saving time')


def test_open_half_hour_is_included():
    """The broker's hourly series starts at 10:00 ET; this one must not."""
    h = aggregate_hourly(clean_rows(_five_min_session('2026-09-16')))
    first_open = h.index[0].tz_convert('America/New_York')
    assert (first_open.hour, first_open.minute) == (9, 30)
    print('ok  the 9:30-10:00 opening half hour is inside the first bucket')


def test_missing_half_no_longer_erases_volume():
    """The broker defect, reproduced: an hourly bar carrying only one half.

    Aggregating the 5-minute parts recovers the full hour, where the broker's
    own hourly bar kept half of it.
    """
    rows = _five_min_session('2026-09-16', vol=1000)
    truth = aggregate_hourly(clean_rows(rows)).iloc[1]['volume']
    broker_style_half = sum(r['volume'] for r in rows[12:18])    # 30 of 60 minutes
    assert truth == 12_000 and broker_style_half == 6_000
    print(f'ok  aggregated hour carries {truth:,} shares where a half-bar carried {broker_style_half:,}')


def test_filler_bars_are_dropped():
    rows = _five_min_session('2026-09-16')
    rows.append({'begins_at': '2026-09-16T21:00:00Z', 'open_price': '1', 'close_price': '1',
                 'high_price': '1', 'low_price': '1', 'volume': 0, 'interpolated': True})
    rows[5] = dict(rows[5], interpolated=True)
    df = clean_rows(rows)
    assert len(df) == 77, len(df)
    print('ok  interpolated filler bars never reach the engine')


def test_duplicates_collapse():
    rows = _five_min_session('2026-09-16')
    df = clean_rows(rows + rows[:3])
    assert len(df) == 78
    print('ok  overlapping fetch windows do not double-count bars')


def test_snapshot_rows_round_trip():
    h = aggregate_hourly(clean_rows(_five_min_session('2026-09-16')))
    back = clean_rows(to_snapshot_rows(h[['open', 'high', 'low', 'close', 'volume']]))
    assert np.allclose(back['close'].values, h['close'].values)
    assert (back.index == h.index).all()
    print('ok  aggregated bars survive the snapshot row format unchanged')


# ── Snapshot builder ─────────────────────────────────────────────────────────

def _tool_file(tmp, name, results):
    p = os.path.join(tmp, name)
    json.dump({'data': {'results': results}}, open(p, 'w'))
    return p


def _build(source, now_et):
    from bot.tools import build_snapshot as B
    cfg = json.load(open('bot/config.json'))
    cfg['strategy']['bar_source'] = source
    with tempfile.TemporaryDirectory() as tmp:
        rows = _five_min_session('2026-09-16')
        rows.append({'begins_at': '2026-09-16T12:00:00Z', 'open_price': '9', 'close_price': '9',
                     'high_price': '9', 'low_price': '9', 'volume': 0, 'interpolated': True})
        bars_f = _tool_file(tmp, 'b.json', [{'symbol': 'SPY', 'bars': rows}])
        reg_f = _tool_file(tmp, 'r.json', [{'symbol': 'SPY', 'bars': rows[:1]}])
        spec = {'today_et': '2026-09-16', 'bar_files': [bars_f], 'regime_file': reg_f,
                'equity': 1000, 'buying_power': 100, 'positions': [], 'quotes': {},
                'now_utc': pd.Timestamp(f'2026-09-16 {now_et}', tz='America/New_York')
                .tz_convert('UTC').isoformat()}
        return B.build(spec, cfg)


def test_builder_drops_filler_on_hourly_path():
    snap, rep = _build('hour', '16:30')
    assert rep['filler_dropped'] == {'SPY': 1}, rep
    assert all(r['v'] > 0 for r in snap['bars']['SPY'])
    print('ok  builder drops filler on the broker-hourly path too')


def test_builder_drops_unfinished_bucket():
    """At 10:47 ET the 9:30 bucket is done; the 10:30 bucket is still forming."""
    snap, rep = _build('5minute', '10:47')
    et = [pd.Timestamp(r['t']).tz_convert('America/New_York').strftime('%H:%M')
          for r in snap['bars']['SPY']]
    assert et == ['09:30'], et
    print('ok  at 10:47 ET the builder keeps the finished 9:30 bucket and drops 10:30')


def test_builder_first_bar_by_1030():
    """The point of the change: today's first bar exists at 10:30, not 11:00."""
    _, rep = _build('5minute', '10:31')
    assert rep['latest_bar'] == '2026-09-16T13:30:00Z', rep
    print('ok  the session\'s first hourly bar is available at 10:30 ET')


def test_config_bar_source_is_explicit():
    cfg = json.load(open('bot/config.json'))
    assert cfg['strategy'].get('bar_source') in ('hour', '5minute'), cfg['strategy'].get('bar_source')
    print(f"ok  config names its bar source explicitly ({cfg['strategy']['bar_source']})")


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('\nall bar tests passed')
