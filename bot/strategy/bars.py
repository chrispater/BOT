"""
Bar hygiene and client-side hourly aggregation.

The broker's own coarse bars are unreliable. Measured 2026-09-23 against their
finer-grained parts, a coarse bar frequently carries only ONE of its
sub-bars instead of their sum:

  hourly  vs its two 30-minute halves : 71% complete (600 bars, 10 symbols)
  30-min  vs its six 5-minute parts   : 79% complete
  5-min   vs its five 1-minute parts  : ~90% complete

Example: MARA's hourly 14:00 ET bar on 9/22 was exactly its 14:30 half
(2,290 shares) — the first half's 944,439 shares were missing. The error
compounds with each level of aggregation, so the finest practical bar is the
most trustworthy source.

The regular-session hourly series also has a structural gap: it is aligned
to clock hours starting 10:00 ET, so the 9:30-10:00 opening half hour — the
heaviest-volume stretch of the day — is in no hourly bar at all, and the
first bar of a session cannot exist until 11:00 ET. That is the "morning
data lag" logged on 17 consecutive sessions; it was never a delay.

`aggregate_hourly` rebuilds hourly bars from 5-minute bars, anchored at the
9:30 open: 9:30, 10:30, ..., 14:30, and a final 15:30-16:00 half bar. A bad
5-minute part now distorts an hour by roughly 1/12 of its volume instead of
the ~1/2 a missing half costs, the open is included, and the first bar of a
session closes at 10:30.

`clean_rows` also drops the broker's synthetic filler bars (`interpolated`),
which are flat and zero-volume. Before this, they reached the live engine:
44% of TOXR's bars in the 2026-09-23 10:47 snapshot were filler.
"""

import pandas as pd

ET = 'America/New_York'
SESSION_OPEN = pd.Timedelta(hours=9, minutes=30)


def clean_rows(rows) -> pd.DataFrame:
    """Broker bar dicts -> OHLCV DataFrame on a UTC index, filler removed.

    Accepts the broker's field names (begins_at/open_price/...) or the
    compact snapshot names (t/o/h/l/c/v).
    """
    rows = [r for r in (rows or []) if not r.get('interpolated')]
    if not rows:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    df = pd.DataFrame(rows).rename(columns={
        'begins_at': 'ts', 't': 'ts', 'open_price': 'open', 'o': 'open',
        'high_price': 'high', 'h': 'high', 'low_price': 'low', 'l': 'low',
        'close_price': 'close', 'c': 'close', 'v': 'volume'})
    for col in ('open', 'high', 'low', 'close', 'volume'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df = df.dropna(subset=['close']).set_index('ts').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df[['open', 'high', 'low', 'close', 'volume']]


def aggregate_hourly(df5: pd.DataFrame) -> pd.DataFrame:
    """5-minute bars -> hourly bars anchored at 9:30 ET, labelled by start (UTC).

    Buckets per session: 9:30, 10:30, 11:30, 12:30, 13:30, 14:30 (60 min) and
    15:30 (30 min, to the close). Open is the first part's open, close the
    last part's close, high/low the extremes, volume the sum. `parts` counts
    the 5-minute bars that went in, so a thin or partial bucket is visible.
    A still-forming bucket is returned like any other; callers drop it by
    age, as run_cycle already does for the newest bar.
    """
    if df5 is None or df5.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'parts'])
    et = df5.index.tz_convert(ET)
    day = et.normalize()
    since_open = (et - day) - SESSION_OPEN
    bucket_start = day + SESSION_OPEN + since_open.floor('60min')
    g = df5.groupby(bucket_start)
    out = pd.DataFrame({
        'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(),
        'close': g['close'].last(), 'volume': g['volume'].sum(), 'parts': g['close'].size(),
    })
    out.index = out.index.tz_convert('UTC')
    out.index.name = 'ts'
    return out


def to_snapshot_rows(df: pd.DataFrame) -> list:
    """DataFrame -> the compact row format run_cycle's snapshot schema uses."""
    return [{'t': ts.strftime('%Y-%m-%dT%H:%M:%SZ'), 'o': r.open, 'h': r.high,
             'l': r.low, 'c': r.close, 'v': r.volume} for ts, r in df.iterrows()]
