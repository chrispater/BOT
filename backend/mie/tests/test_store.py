import pandas as pd

from backend.mie.feature_engine import FeatureEngine
from backend.mie.store import ObservationStore
from backend.mie.state import MarketState
from .conftest import make_ohlcv


def _record_all_bars(store, fe, df, user_id=1, symbol='TEST'):
    feats = fe.build_frame(df, keep_prices=False)
    for ts, row in feats.iterrows():
        st = MarketState(symbol=symbol, as_of=ts, price=float(df.loc[ts, 'close']),
                         features={k: float(v) for k, v in row.items() if pd.notna(v)})
        store.record(user_id, st)
    return feats


def test_record_is_idempotent_on_unique_key():
    df = make_ohlcv(n=200, seed=5)
    fe = FeatureEngine()
    store = ObservationStore()
    feats = _record_all_bars(store, fe, df)
    n_before = store.count_resolved(symbol='TEST')  # 0, nothing resolved yet
    # Recording the exact same observations again must not create duplicates.
    _record_all_bars(store, fe, df)
    with store._conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    assert total == len(feats)


def test_backfill_outcomes_resolves_and_is_incremental():
    df = make_ohlcv(n=200, seed=6)
    fe = FeatureEngine()
    store = ObservationStore()
    _record_all_bars(store, fe, df)

    updated = store.backfill_outcomes(1, 'TEST', df, bar_seconds=300)
    assert updated > 0
    resolved = store.count_resolved(symbol='TEST')
    assert resolved > 0

    # Calling again with the same data changes nothing further.
    updated_again = store.backfill_outcomes(1, 'TEST', df, bar_seconds=300)
    assert updated_again == 0


def test_training_frame_has_outcome_columns():
    df = make_ohlcv(n=300, seed=8)
    fe = FeatureEngine()
    store = ObservationStore()
    _record_all_bars(store, fe, df)
    store.backfill_outcomes(1, 'TEST', df, bar_seconds=300)

    train_df = store.load_training_frame(symbol='TEST')
    assert not train_df.empty
    assert 'ret_5m' in train_df.columns
    assert 'ps_ret_1' in train_df.columns


def test_similar_observations_returns_nearest_first():
    df = make_ohlcv(n=300, seed=9)
    fe = FeatureEngine()
    store = ObservationStore()
    feats = _record_all_bars(store, fe, df)
    store.backfill_outcomes(1, 'TEST', df, bar_seconds=300)

    query_row = feats.iloc[-1]
    query_state = MarketState(symbol='TEST', as_of=feats.index[-1], price=float(df['close'].iloc[-1]),
                              features={k: float(v) for k, v in query_row.items() if pd.notna(v)})
    neighbors, dists = store.similar_observations(query_state, list(feats.columns), k=5)
    assert len(neighbors) <= 5
    if len(dists) > 1:
        assert (dists[:-1] <= dists[1:]).all()  # sorted ascending by distance
