"""
Regime model — answers "what kind of market is this right now?" as a discrete
label with a confidence, derived from the continuous regime-group features the
feature engine already computed (trend strength, efficiency, autocorrelation,
volatility rank/of-vol, squeeze, turnover).

Deliberately rule-based rather than learned. Two reasons:

  1. Interpretability matters more here than anywhere else in the system. The
     regime label gates whether the edge model's forecast is even allowed to
     produce a trade (see engine.py). If the gate itself were an opaque model,
     a bad day is undebuggable — "the model says compression" isn't an answer
     to "why aren't we trading" the way "efficiency 0.04, squeeze 1.6×, ADX 11"
     is.
  2. A rule-based regime classifier can be evaluated and tuned against realized
     forward volatility/return dispersion by regime — see `evaluate_regimes` —
     without the chicken-and-egg problem of needing regime labels to train a
     regime labeler.

The thresholds are exposed as a dataclass so the optimizer (or a human) can
retune them against realized data without touching this module's logic.
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional
import math

import numpy as np
import pandas as pd

from .state import (
    REGIME_TREND_UP, REGIME_TREND_DOWN, REGIME_MEAN_REVERT, REGIME_COMPRESSION,
    REGIME_EXPANSION, REGIME_PANIC, REGIME_THIN, GROUP_REGIME, GROUP_PRICE,
    GROUP_EXEC,
)


@dataclass
class RegimeThresholds:
    trend_strength_min: float = 0.55     # |ema_fast - ema_slow| / ATR to call trend
    efficiency_trend_min: float = 0.35   # net/path ratio for genuine trend
    efficiency_revert_max: float = 0.12  # below this, price is going nowhere net
    autocorr_revert_max: float = -0.05   # negative autocorrelation → mean reversion
    squeeze_compression_max: float = 0.75  # bb-width vs its own 50-bar average
    vol_rank_expansion_min: float = 0.85   # realized vol in its own top 15%
    vol_of_vol_panic_min: float = 0.90     # vol-of-vol spike: instability itself
    ret_5_panic_abs_min: float = 0.03      # 5-bar move ≥3% alongside vol_of_vol spike
    turnover_thin_max: float = 0.15        # turnover in bottom 15% of its own history
    spread_thin_bps: float = 15.0          # wide spread also marks a thin/untradeable book


DEFAULT_THRESHOLDS = RegimeThresholds()


def _get(features: Dict[str, float], key: str, default: float = float('nan')) -> float:
    v = features.get(key, default)
    return default if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)


def classify_regime(features: Dict[str, float],
                    thresholds: RegimeThresholds = DEFAULT_THRESHOLDS) -> tuple:
    """
    Classify one observation's features into (regime, confidence).

    Order of checks matters: panic and thin-liquidity are checked first because
    they are override conditions — a "trending" market that is actually a
    liquidation cascade should be flagged PANIC, not TREND_UP, because the
    tradeable character of the two is completely different even when the price
    is moving in the same direction.
    """
    t = thresholds
    # A completely absent regime vector (cold start: not enough bars yet for the
    # warmup windows these features need) is a different situation from a
    # regime vector that was actually computed and happens to read near zero.
    # Treating "never observed" as "observed and flat" would let a classifier
    # issue a confident regime call from data it never saw — the same failure
    # mode the coverage gate exists to prevent everywhere else in this engine.
    _core_keys = (f'{GROUP_REGIME}_trend_strength', f'{GROUP_REGIME}_efficiency_50',
                 f'{GROUP_REGIME}_squeeze', f'{GROUP_REGIME}_vol_rank_200',
                 f'{GROUP_REGIME}_turnover_rank', f'{GROUP_REGIME}_vol_of_vol',
                 f'{GROUP_REGIME}_autocorr_1', f'{GROUP_EXEC}_spread_pct')
    if not any(k in features for k in _core_keys):
        return REGIME_COMPRESSION, 0.0

    trend_strength = _get(features, f'{GROUP_REGIME}_trend_strength', 0.0)
    efficiency = _get(features, f'{GROUP_REGIME}_efficiency_50', 0.0)
    autocorr = _get(features, f'{GROUP_REGIME}_autocorr_1', 0.0)
    squeeze = _get(features, f'{GROUP_REGIME}_squeeze', 1.0)
    vol_rank = _get(features, f'{GROUP_REGIME}_vol_rank_200', 0.5)
    vol_of_vol = _get(features, f'{GROUP_REGIME}_vol_of_vol', 0.0)
    turnover_rank = _get(features, f'{GROUP_REGIME}_turnover_rank', 0.5)
    ret_5 = abs(_get(features, f'{GROUP_PRICE}_ret_5', 0.0))
    spread_bps = _get(features, f'{GROUP_EXEC}_spread_pct', float('nan')) * 10_000.0 \
        if not math.isnan(_get(features, f'{GROUP_EXEC}_spread_pct', float('nan'))) else float('nan')

    # ── Panic: volatility itself is unstable AND price is moving hard. This
    # catches liquidation cascades that a plain trend-strength check would
    # misfile as a strong trend — the distinguishing signal is vol-of-vol, not
    # direction.
    if vol_of_vol >= t.vol_of_vol_panic_min and ret_5 >= t.ret_5_panic_abs_min:
        conf = min(1.0, 0.5 + (vol_of_vol - t.vol_of_vol_panic_min) * 2 + ret_5)
        return REGIME_PANIC, round(conf, 3)

    # ── Thin: little real turnover, or a spread wide enough that nothing here
    # survives round-trip costs. Checked before trend/reversion because a
    # trend detected on a dead book is not a trend you can actually trade.
    thin_by_turnover = turnover_rank <= t.turnover_thin_max and not math.isnan(turnover_rank)
    thin_by_spread = (not math.isnan(spread_bps)) and spread_bps >= t.spread_thin_bps
    if thin_by_turnover or thin_by_spread:
        conf = 0.55 if (thin_by_turnover and thin_by_spread) else 0.4
        return REGIME_THIN, round(conf, 3)

    # ── Expansion: volatility breaking out of its own recent range. This is
    # about MAGNITUDE of movement, independent of direction or persistence —
    # early innings of a move that hasn't shown its character yet.
    if vol_rank >= t.vol_rank_expansion_min and squeeze > 1.0:
        conf = min(1.0, (vol_rank - t.vol_rank_expansion_min) / (1 - t.vol_rank_expansion_min + 1e-9))
        return REGIME_EXPANSION, round(0.5 + 0.5 * conf, 3)

    # ── Compression: the coil before expansion. Low relative bb-width, weak
    # trend strength — nothing is happening yet, which is itself the state.
    if squeeze <= t.squeeze_compression_max and abs(trend_strength) < t.trend_strength_min:
        conf = min(1.0, (t.squeeze_compression_max - squeeze) / t.squeeze_compression_max + 0.3)
        return REGIME_COMPRESSION, round(conf, 3)

    # ── Trend: efficient, sustained, directional movement.
    if efficiency >= t.efficiency_trend_min and abs(trend_strength) >= t.trend_strength_min:
        conf = min(1.0, 0.4 + efficiency + min(abs(trend_strength), 2.0) * 0.15)
        return (REGIME_TREND_UP if trend_strength > 0 else REGIME_TREND_DOWN), round(conf, 3)

    # ── Mean reversion: inefficient path (a lot of back-and-forth for little
    # net displacement) with negative autocorrelation — moves tend to give back.
    if efficiency <= t.efficiency_revert_max or autocorr <= t.autocorr_revert_max:
        conf = min(1.0, 0.4 + max(0.0, t.efficiency_revert_max - efficiency) * 3
                        + max(0.0, -autocorr) * 4)
        return REGIME_MEAN_REVERT, round(conf, 3)

    # ── Nothing decisive fired — the honest answer is low-confidence
    # compression, not a forced guess at a more exciting label.
    return REGIME_COMPRESSION, 0.30


def evaluate_regimes(labeled_df: pd.DataFrame, return_col: str = 'ret_5m') -> pd.DataFrame:
    """
    Sanity-check the regime taxonomy against realized data: does forward return
    dispersion actually differ across the labels we're assigning? A regime model
    earns its keep only if grouping by it changes the distribution of what
    happens next — group by `regime`, report mean/std of `return_col` and count.

    Expects `labeled_df` to have a 'regime' column (e.g. from load_training_frame
    after classify_regime has been applied) and the named outcome column.
    """
    if 'regime' not in labeled_df.columns or return_col not in labeled_df.columns:
        return pd.DataFrame()
    g = labeled_df.dropna(subset=[return_col]).groupby('regime')[return_col]
    return g.agg(['count', 'mean', 'std']).sort_values('count', ascending=False)


class RegimeModel:
    """Thin stateful wrapper so callers don't have to pass thresholds around."""

    def __init__(self, thresholds: Optional[RegimeThresholds] = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def classify(self, features: Dict[str, float]) -> tuple:
        return classify_regime(features, self.thresholds)

    def retune(self, **overrides) -> None:
        self.thresholds = replace(self.thresholds, **overrides)
