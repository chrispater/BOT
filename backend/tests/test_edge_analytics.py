"""
Regression coverage for edge_analytics.bucket_keys() — the single definition
of what a "condition bucket" is, shared between the code that BUILDS the edge
profile from trade history (analyze_trades/analyze_observations) and the code
that QUERIES it live (TradingService._trade_quality). Those two used to
diverge: the live lookup hand-rolled its own shorter key list that omitted
symbol/side/session/book entirely, so the quality gate could never see
per-coin or per-direction evidence even though the profile had it — every
coin in the same macro regime scored identically regardless of how different
their actual track records were. These tests exist so that gap can't reopen
silently.
"""

from backend.edge_analytics import bucket_keys


def test_bucket_keys_includes_symbol_and_side():
    row = {'regime': 'bull', 'setup_name': None, 'vol_pct': 0.5,
          'hour_utc': 14, 'symbol': 'BTC/USDT:USDT', 'side': 'long'}
    keys = bucket_keys(row)
    assert 'symbol=BTC/USDT:USDT' in keys
    assert 'side=long' in keys


def test_bucket_keys_differentiates_two_symbols():
    """The actual failure mode reported: two different coins, same regime,
    must NOT produce identical bucket-key sets — that's what made every coin
    score identically regardless of its own track record."""
    common = {'regime': 'bull', 'setup_name': None, 'vol_pct': 0.5, 'hour_utc': 14}
    btc_keys = set(bucket_keys({**common, 'symbol': 'BTC/USDT:USDT', 'side': 'long'}))
    eth_keys = set(bucket_keys({**common, 'symbol': 'ETH/USDT:USDT', 'side': 'long'}))

    assert btc_keys != eth_keys
    assert 'symbol=BTC/USDT:USDT' in btc_keys and 'symbol=BTC/USDT:USDT' not in eth_keys
    assert 'symbol=ETH/USDT:USDT' in eth_keys and 'symbol=ETH/USDT:USDT' not in btc_keys
    # But the shared, coin-agnostic keys (regime/vol/session) should still
    # match — the two coins really are in the same macro regime right now.
    shared = {k for k in btc_keys if k.startswith(('regime=', 'vol=', 'session='))}
    assert shared and shared.issubset(eth_keys)


def test_bucket_keys_includes_session_from_hour():
    row = {'regime': 'bull', 'setup_name': None, 'vol_pct': 0.5,
          'hour_utc': 14, 'symbol': 'BTC/USDT:USDT', 'side': 'long'}
    keys = bucket_keys(row)
    assert any(k.startswith('session=') for k in keys)


def test_bucket_keys_includes_context_derived_keys_when_present():
    row = {'regime': 'bull', 'setup_name': None, 'vol_pct': 0.5, 'hour_utc': 14,
          'symbol': 'BTC/USDT:USDT', 'side': 'short',
          'context': {'btc_agree': True, 'book_imbalance': 0.62}}
    keys = bucket_keys(row)
    assert 'btc_agree=True' in keys
    assert 'book=bid_heavy' in keys


def test_bucket_keys_tolerates_missing_context():
    row = {'regime': 'bull', 'setup_name': None, 'vol_pct': 0.5,
          'hour_utc': 14, 'symbol': 'BTC/USDT:USDT', 'side': 'long'}
    keys = bucket_keys(row)   # no 'context' key at all — must not raise
    assert not any(k.startswith('btc_agree=') or k.startswith('book=') for k in keys)
