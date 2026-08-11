import numpy as np

from backend.mie.validation import purged_walk_forward_splits


def test_splits_are_chronological_and_non_overlapping():
    n = 1000
    seen_test = []
    for train_idx, test_idx in purged_walk_forward_splits(n, n_splits=5, embargo=20, min_train=200):
        assert train_idx.max() < test_idx.min()          # train strictly precedes test
        seen_test.append(test_idx)
    all_test = np.concatenate(seen_test)
    assert (np.diff(all_test) > 0).all()                  # test blocks march forward, no overlap


def test_embargo_gap_enforced():
    n = 1000
    embargo = 37
    for train_idx, test_idx in purged_walk_forward_splits(n, n_splits=4, embargo=embargo, min_train=200):
        gap = test_idx.min() - train_idx.max()
        assert gap >= embargo, f"embargo violated: gap={gap} < {embargo}"


def test_too_little_data_yields_no_splits():
    splits = list(purged_walk_forward_splits(n=50, n_splits=5, embargo=20, min_train=200))
    assert splits == []


def test_edge_model_validation_gate_rejects_and_accepts_correctly():
    """
    Integration-level check on the piece that matters most: the validator must
    say NO to a diluted/absent relationship and YES to a real, horizon-matched,
    persistent one — using the exact EdgeModel.fit() path, not a toy re-
    implementation of the logic under test.
    """
    from backend.mie.feature_engine import FeatureEngine
    from backend.mie.outcomes import resolve_outcomes
    from backend.mie.edge_model import EdgeModel
    from .conftest import make_ohlcv, make_predictable_ohlcv

    # No embedded relationship at all -> must not validate.
    df_noise = make_ohlcv(n=3000, seed=11, freq='1min', vol=0.05)
    fe = FeatureEngine(closed_only=False)
    feats = fe.build_frame(df_noise, keep_prices=True)
    outcomes = resolve_outcomes(df_noise, bar_seconds=60, horizons=[180])
    full = feats.join(outcomes)
    candidate_features = [c for c in feats.columns if c not in ('open', 'high', 'low', 'close', 'volume')]
    model = EdgeModel(horizon_sec=180)
    model.fit(full, candidate_features, bar_seconds=60)
    assert model.validated is False

    # Real, horizon-matched, persistent relationship -> must validate.
    df_signal, signal = make_predictable_ohlcv(n=4000, seed=7, bar_seconds=60, horizon_sec=180, effect=0.006)
    feats2 = fe.build_frame(df_signal, keep_prices=True)
    feats2['synthetic_signal'] = signal[:len(feats2)]
    outcomes2 = resolve_outcomes(df_signal, bar_seconds=60, horizons=[180])
    full2 = feats2.join(outcomes2)
    candidate_features2 = ['synthetic_signal'] + [c for c in feats2.columns
                                                  if c not in ('open', 'high', 'low', 'close', 'volume')]
    model2 = EdgeModel(horizon_sec=180, cost_assumption=0.0004)
    report = model2.fit(full2, candidate_features2, bar_seconds=60)
    assert model2.validated is True
    assert report.folds_with_edge == report.n_splits_run  # edge holds in EVERY fold, not one lucky one
    assert report.pooled_trades >= 30
