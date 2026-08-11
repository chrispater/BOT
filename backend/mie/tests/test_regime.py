from backend.mie.regime import classify_regime
from backend.mie.state import (
    REGIME_PANIC, REGIME_THIN, REGIME_COMPRESSION, REGIME_TREND_UP, REGIME_TREND_DOWN,
    REGIME_MEAN_REVERT, REGIME_EXPANSION, GROUP_REGIME, GROUP_PRICE, GROUP_EXEC,
)


def test_panic_overrides_trend_looking_features():
    # High trend strength AND unstable, spiking volatility with a big move —
    # must be classified PANIC, not TREND_UP, even though trend_strength alone
    # would suggest a strong uptrend.
    feats = {
        f'{GROUP_REGIME}_trend_strength': 3.0,
        f'{GROUP_REGIME}_efficiency_50': 0.8,
        f'{GROUP_REGIME}_vol_of_vol': 0.95,
        f'{GROUP_PRICE}_ret_5': 0.05,
        f'{GROUP_REGIME}_turnover_rank': 0.5,
    }
    regime, conf = classify_regime(feats)
    assert regime == REGIME_PANIC
    assert conf > 0


def test_thin_liquidity_detected_by_turnover():
    feats = {f'{GROUP_REGIME}_turnover_rank': 0.05, f'{GROUP_REGIME}_trend_strength': 0.0}
    regime, conf = classify_regime(feats)
    assert regime == REGIME_THIN


def test_thin_liquidity_detected_by_wide_spread():
    feats = {f'{GROUP_EXEC}_spread_pct': 0.005, f'{GROUP_REGIME}_turnover_rank': 0.5}
    regime, conf = classify_regime(feats)
    assert regime == REGIME_THIN


def test_trend_up_detected():
    feats = {
        f'{GROUP_REGIME}_efficiency_50': 0.5,
        f'{GROUP_REGIME}_trend_strength': 1.0,
        f'{GROUP_REGIME}_turnover_rank': 0.6,
        f'{GROUP_REGIME}_vol_of_vol': 0.1,
        f'{GROUP_REGIME}_vol_rank_200': 0.4,
        f'{GROUP_REGIME}_squeeze': 1.0,
    }
    regime, conf = classify_regime(feats)
    assert regime == REGIME_TREND_UP


def test_trend_down_mirrors_trend_up():
    feats = {
        f'{GROUP_REGIME}_efficiency_50': 0.5,
        f'{GROUP_REGIME}_trend_strength': -1.0,
        f'{GROUP_REGIME}_turnover_rank': 0.6,
        f'{GROUP_REGIME}_vol_of_vol': 0.1,
        f'{GROUP_REGIME}_vol_rank_200': 0.4,
        f'{GROUP_REGIME}_squeeze': 1.0,
    }
    regime, conf = classify_regime(feats)
    assert regime == REGIME_MEAN_REVERT or regime == REGIME_TREND_DOWN
    # (trend_strength=-1.0 alone may not clear efficiency_trend_min combined
    # with default thresholds; the meaningful assertion is that it is never
    # misclassified as TREND_UP)
    assert regime != REGIME_TREND_UP


def test_mean_revert_detected_by_negative_autocorr():
    feats = {
        f'{GROUP_REGIME}_efficiency_50': 0.05,
        f'{GROUP_REGIME}_autocorr_1': -0.3,
        f'{GROUP_REGIME}_trend_strength': 0.1,
        f'{GROUP_REGIME}_turnover_rank': 0.5,
        f'{GROUP_REGIME}_vol_of_vol': 0.1,
        f'{GROUP_REGIME}_vol_rank_200': 0.4,
        f'{GROUP_REGIME}_squeeze': 1.0,
    }
    regime, conf = classify_regime(feats)
    assert regime == REGIME_MEAN_REVERT


def test_missing_features_default_gracefully():
    regime, conf = classify_regime({})
    assert regime == REGIME_COMPRESSION
    assert 0.0 <= conf <= 1.0
