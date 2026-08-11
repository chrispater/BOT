"""
Market Intelligence Engine.

A research-first replacement for "does the model say buy or sell": records
market state continuously without a label, waits for the future to actually
happen, attaches real forward return / MFE / MAE outcomes, and only lets a
model influence a live decision after it has survived purged walk-forward
validation on data it never trained on.

    MARKET DATA → FeatureEngine → RegimeModel → EdgeModel → cost model
                → MarketIntelligenceEngine.decide() → TradeDecision
                → ObservationStore (record + eventually resolve outcomes)
                → analysis.run_post_mortem() once trades close

DO_NOTHING is the default TradeDecision — see state.py — not a fallback wired
in after the fact. See engine.py for the orchestration and README.md (next to
this file) for the full design rationale.
"""

from .state import (
    MarketState, TradeDecision, EdgeForecast, CostEstimate,
    ACTION_LONG, ACTION_SHORT, ACTION_NOTHING,
    HORIZONS_SEC, horizon_label,
    REGIME_TREND_UP, REGIME_TREND_DOWN, REGIME_MEAN_REVERT, REGIME_COMPRESSION,
    REGIME_EXPANSION, REGIME_PANIC, REGIME_THIN, HOSTILE_REGIMES,
)
from .feature_engine import FeatureEngine
from .regime import RegimeModel, RegimeThresholds
from .edge_model import EdgeModel
from .costs import estimate_costs_from_book, estimate_costs_fallback, FeeTier
from .store import ObservationStore
from .engine import MarketIntelligenceEngine, EngineConfig
from .analysis import run_post_mortem, PostMortemReport

__all__ = [
    'MarketState', 'TradeDecision', 'EdgeForecast', 'CostEstimate',
    'ACTION_LONG', 'ACTION_SHORT', 'ACTION_NOTHING',
    'HORIZONS_SEC', 'horizon_label',
    'REGIME_TREND_UP', 'REGIME_TREND_DOWN', 'REGIME_MEAN_REVERT', 'REGIME_COMPRESSION',
    'REGIME_EXPANSION', 'REGIME_PANIC', 'REGIME_THIN', 'HOSTILE_REGIMES',
    'FeatureEngine', 'RegimeModel', 'RegimeThresholds', 'EdgeModel',
    'estimate_costs_from_book', 'estimate_costs_fallback', 'FeeTier',
    'ObservationStore', 'MarketIntelligenceEngine', 'EngineConfig',
    'run_post_mortem', 'PostMortemReport',
]
