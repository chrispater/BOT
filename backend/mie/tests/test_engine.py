import pandas as pd

from backend.mie.engine import MarketIntelligenceEngine, EngineConfig
from backend.mie.costs import FeeTier
from backend.mie.state import ACTION_NOTHING, ACTION_LONG, ACTION_SHORT, HORIZONS_SEC
from backend.mie.outcomes import resolve_outcomes
from backend.mie.feature_engine import FeatureEngine
from .conftest import make_ohlcv, make_predictable_ohlcv, SignalInjectingFeatureEngine


def _fit_engine_on_predictable_data(cfg, n=4000, seed=7, bar_seconds=60, horizon_sec=180, effect=0.006):
    """Shared setup: build predictable OHLCV, fit an engine whose live feature
    path can actually reproduce the injected signal (see SignalInjectingFeatureEngine),
    and hand back (engine, df) ready for decide()."""
    df, signal = make_predictable_ohlcv(n=n, seed=seed, bar_seconds=bar_seconds,
                                        horizon_sec=horizon_sec, effect=effect)
    signal_series = pd.Series(signal, index=df.index)
    fe = SignalInjectingFeatureEngine(signal_series, closed_only=False)
    feats = fe.build_frame(df, keep_prices=True)
    outcomes = resolve_outcomes(df, bar_seconds=bar_seconds, horizons=[horizon_sec])
    full = feats.join(outcomes)

    engine = MarketIntelligenceEngine(config=cfg)
    engine.feature_engine = fe   # so decide()'s build_state also sees synthetic_signal
    engine.fit(full, bar_seconds=bar_seconds)
    return engine, df


def test_decide_defaults_to_do_nothing_before_any_fit():
    df = make_ohlcv(n=200, seed=1)
    engine = MarketIntelligenceEngine(config=EngineConfig(horizons=[300]))
    decision = engine.decide('TEST', df)
    assert decision.action == ACTION_NOTHING
    assert decision.is_trade is False


def test_decide_trades_once_fit_with_a_real_edge():
    cfg = EngineConfig(horizons=[180], fee_tier=FeeTier(0.0001, 0.0002), min_quality=40)
    engine, df = _fit_engine_on_predictable_data(cfg)
    assert engine.any_validated

    decision = engine.decide('TEST', df)
    assert decision.action in (ACTION_LONG, ACTION_SHORT)
    assert decision.historical_sample >= cfg.min_pooled_trades
    assert decision.quality >= cfg.min_quality
    assert decision.expectancy_r > 0
    assert not decision.blockers


def test_decide_abstains_in_hostile_regime_even_with_validated_model():
    from backend.mie.state import REGIME_PANIC
    engine, df = _fit_engine_on_predictable_data(EngineConfig(horizons=[180], min_quality=40))

    # Force the regime classifier to always report a hostile regime, regardless
    # of the actual feature values, to isolate the gate under test.
    engine.regime_model.classify = lambda features: (REGIME_PANIC, 0.9)
    decision = engine.decide('TEST', df)
    assert decision.action == ACTION_NOTHING
    assert any('hostile' in b for b in decision.blockers)


def test_quality_and_size_fraction_are_bounded():
    cfg = EngineConfig(horizons=[180], min_quality=40)
    engine, df = _fit_engine_on_predictable_data(cfg, effect=0.01)
    decision = engine.decide('TEST', df)

    assert 0 <= decision.quality <= 100
    if decision.is_trade:
        assert cfg.risk_per_trade_floor <= decision.size_fraction <= cfg.risk_per_trade_cap


def test_engine_records_observation_to_store_when_given_one():
    from backend.mie.store import ObservationStore
    df = make_ohlcv(n=200, seed=2)
    store = ObservationStore()
    engine = MarketIntelligenceEngine(config=EngineConfig(horizons=[300]), store=store)
    engine.decide('TEST', df, user_id=1)
    with store._conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM observations WHERE symbol='TEST'").fetchone()[0]
    assert n == 1


def test_fit_collapses_aliased_horizons_on_coarse_timeframe():
    """
    Regression test for a real bug: on a 5-minute bot, the default HORIZONS_SEC
    (30s/1m/3m/5m/15m) has four values that all round to "1 bar ahead" via
    bars_for_horizon, so fitting one EdgeModel per nominal horizon produced
    four separately-trained-but-byte-identical models — same features, same
    forward-return column — which read (correctly) as broken to a user
    watching the Optimize tab's validation panel show identical stats under
    four different horizon labels. engine.edge_models must never contain more
    entries than there are genuinely distinct bar counts for the fit's
    bar_seconds.
    """
    df = make_ohlcv(n=1500, seed=11, freq='5min')
    fe = FeatureEngine(closed_only=False)
    feats = fe.build_frame(df, keep_prices=True)
    outcomes = resolve_outcomes(df, bar_seconds=300, horizons=HORIZONS_SEC)
    full = feats.join(outcomes)

    engine = MarketIntelligenceEngine(config=EngineConfig(horizons=list(HORIZONS_SEC)))
    engine.fit(full, bar_seconds=300)

    assert sorted(engine.edge_models.keys()) == [300, 900]   # not 5 duplicate entries
    # And the two survivors must be genuinely different fits, not literally
    # the same object reused under two keys.
    assert engine.edge_models[300] is not engine.edge_models[900]


def test_fit_prunes_stale_horizons_after_timeframe_change():
    """A model fit under an old bar_seconds must not linger once a re-fit at a
    different bar_seconds makes it a stale alias — a leftover model reflects
    the wrong bar interval and must never be consulted as if it were current."""
    df = make_ohlcv(n=1500, seed=12, freq='1min')
    fe = FeatureEngine(closed_only=False)
    feats = fe.build_frame(df, keep_prices=True)
    engine = MarketIntelligenceEngine(config=EngineConfig(horizons=list(HORIZONS_SEC)))

    outcomes_1m = resolve_outcomes(df, bar_seconds=60, horizons=HORIZONS_SEC)
    engine.fit(feats.join(outcomes_1m), bar_seconds=60)
    assert sorted(engine.edge_models.keys()) == [60, 180, 300, 900]

    outcomes_15m = resolve_outcomes(df, bar_seconds=900, horizons=HORIZONS_SEC)
    engine.fit(feats.join(outcomes_15m), bar_seconds=900)
    assert sorted(engine.edge_models.keys()) == [900]   # 60/180/300 pruned, not left stale
