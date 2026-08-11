import numpy as np
import pandas as pd

from backend.mie.feature_engine import FeatureEngine, group_coverage
from backend.mie.state import GROUP_ORDERFLOW, GROUP_DERIVS
from .conftest import make_ohlcv


def test_build_frame_shape_and_no_all_nan_columns():
    df = make_ohlcv(n=400)
    fe = FeatureEngine()
    feats = fe.build_frame(df, keep_prices=False)
    assert len(feats) == len(df) - 1   # closed_only drops the forming last candle
    # Every column should have SOME real values once past its warmup window.
    tail = feats.iloc[-50:]
    all_nan_cols = [c for c in tail.columns if tail[c].isna().all()]
    assert not all_nan_cols, f"columns never populate: {all_nan_cols}"


def test_no_lookahead_prefix_invariance():
    """
    The defining correctness property of the whole engine: a feature computed
    for bar i must be identical whether it was computed from a frame that ends
    at bar i, or from a much longer frame that happens to include bars after i.
    If it isn't, some feature is peeking at the future — which would make any
    backtest built on it untradeable in reality.
    """
    df = make_ohlcv(n=400, seed=3)
    fe = FeatureEngine(closed_only=False)  # test the raw builder, not the live trim

    full = fe.build_frame(df, keep_prices=False)
    prefix = fe.build_frame(df.iloc[:250], keep_prices=False)

    common_idx = prefix.index
    # Compare on the tail of the overlap only — rolling windows need warmup, and
    # early rows can legitimately still be NaN in one frame's context due solely
    # to that warmup, not lookahead.
    check_idx = common_idx[-50:]
    pd.testing.assert_frame_equal(
        full.loc[check_idx].reset_index(drop=True),
        prefix.loc[check_idx].reset_index(drop=True),
        check_dtype=False, atol=1e-9,
    )


def test_closed_only_drops_forming_candle():
    df = make_ohlcv(n=300, seed=1)
    fe = FeatureEngine(closed_only=True)
    feats = fe.build_frame(df, keep_prices=False)
    assert feats.index[-1] == df.index[-2]


def test_group_coverage_reports_missing_live_only_groups():
    df = make_ohlcv(n=300, seed=2)
    fe = FeatureEngine()
    state = fe.build_state('TEST', df)   # no book/trades/derivs supplied
    assert state.coverage[GROUP_ORDERFLOW] == 0.0
    assert state.coverage[GROUP_DERIVS] == 0.0
    assert state.coverage['ps'] == 1.0


def test_orderbook_features_present_when_book_supplied():
    df = make_ohlcv(n=300, seed=4)
    fe = FeatureEngine()
    book = {'bids': [[99.9, 2], [99.8, 5], [99.7, 10]],
           'asks': [[100.1, 1.5], [100.2, 4], [100.3, 8]]}
    state = fe.build_state('TEST', df, book=book)
    assert state.coverage[GROUP_ORDERFLOW] > 0.0
    assert any(k.startswith('of_') for k in state.features)


def test_build_frame_raises_on_too_little_data():
    df = make_ohlcv(n=10)
    fe = FeatureEngine()
    try:
        fe.build_frame(df)
        assert False, "expected ValueError on too-short input"
    except ValueError:
        pass
