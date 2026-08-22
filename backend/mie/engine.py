"""
The Market Intelligence Engine — the orchestrator.

    MARKET DATA → FEATURE ENGINE → REGIME MODEL → EDGE MODEL → COST MODEL
                → RISK/EV GATE → TRADE DECISION → (later) AI POST-MORTEM

The single output of this module is a TradeDecision, and DO_NOTHING is the
default value of that object — not a special case bolted on afterward. Every
gate in `decide()` is a reason a candidate trade gets vetoed on the way OUT of
DO_NOTHING; nothing in this file has to affirmatively justify staying flat.
That asymmetry is the entire point: guessing wrong about "should I trade"
costs money on every wrong guess, so the burden of proof sits on the trade,
never on the abstention.

What makes it to a LONG/SHORT action has to clear, simultaneously:
  • a regime that isn't hostile to trading at all (panic / thin liquidity)
  • an edge model that PASSED purged walk-forward validation for this horizon
  • positive expected return net of realistic, currently-observed costs
  • a minimum count of comparable historical observations behind the forecast
  • (when the store has enough history) nearest-neighbor corroboration that
    doesn't flatly contradict the model

Nothing here predicts candles. It reports, with the receipts attached, whether
an edge exists right now — and how it would size a bet on it if it does.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from .state import (
    MarketState, TradeDecision, EdgeForecast, HORIZONS_SEC, HOSTILE_REGIMES,
    ACTION_LONG, ACTION_SHORT, ACTION_NOTHING, GROUP_PRICE, GROUP_EXEC,
    horizon_label,
)
from .feature_engine import FeatureEngine, DEFAULT_PROBE_NOTIONAL
from .regime import RegimeModel
from .edge_model import EdgeModel, select_trainable_features
from .costs import estimate_costs_from_book, estimate_costs_fallback, DEFAULT_FEE_TIER, FeeTier
from .store import ObservationStore
from .outcomes import resolve_distinct_horizons

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    horizons: List[int] = field(default_factory=lambda: list(HORIZONS_SEC))
    min_quality: int = 60                  # 0-100 — below this, DO_NOTHING regardless of EV sign
    min_pooled_trades: int = 30            # OOS trades behind the model before it's trusted at all
    min_model_confidence: float = 0.30     # EdgeModel._confidence() floor
    min_regime_confidence: float = 0.35    # a regime call this uncertain isn't a basis for sizing up
    max_cost_to_return_ratio: float = 0.65 # costs must not eat more than this fraction of the gross move
    min_neighbor_k: int = 20               # neighbors needed before their vote counts at all
    stop_floor: float = 0.0015             # never treat the stop as tighter than 15bps of price
    risk_per_trade_floor: float = 0.0015
    risk_per_trade_cap: float = 0.02
    probe_notional: float = DEFAULT_PROBE_NOTIONAL
    fee_tier: FeeTier = field(default_factory=lambda: DEFAULT_FEE_TIER)


class MarketIntelligenceEngine:
    """
    Owns one FeatureEngine, one RegimeModel, and one EdgeModel per horizon.
    `fit()` trains/validates every horizon's model from the store's accumulated
    (or backtest-supplied) history; `decide()` produces one TradeDecision per
    call for one symbol's current state.
    """

    def __init__(self, config: Optional[EngineConfig] = None,
                store: Optional[ObservationStore] = None):
        self.config = config or EngineConfig()
        self.feature_engine = FeatureEngine(probe_notional=self.config.probe_notional)
        self.regime_model = RegimeModel()
        self.store = store
        self.edge_models: Dict[int, EdgeModel] = {}
        self._candidate_features: List[str] = []

    # ── training ───────────────────────────────────────────────────────

    def fit(self, training_df: pd.DataFrame, bar_seconds: float,
           candidate_features: Optional[List[str]] = None) -> Dict[int, str]:
        """
        Fit + validate one EdgeModel per DISTINCT configured horizon against
        `training_df` (typically store.load_training_frame(...) or a backtest's
        feature+outcome frame). Returns {horizon_sec: validation summary line}
        for logging/inspection; the models themselves are held internally.

        "Distinct" matters: `bars_for_horizon` floors every horizon to a whole
        number of bars, so on a timeframe coarser than several configured
        horizons (e.g. 30s/1m/3m/5m all round to "1 bar ahead" on a 5-minute
        bot), fitting one model per nominal horizon would silently fit the
        same model several times over — same features, same forward-return
        column, byte-identical validation stats — under different name tags.
        `resolve_distinct_horizons` collapses those to one model per bar
        count before any fitting happens, so `edge_models` never holds
        duplicates and callers never see confusingly-identical reports.
        """
        candidate_features = candidate_features or [
            c for c in training_df.columns
            if c not in ('symbol', 'as_of', 'price', 'regime', 'regime_confidence')
            and not c.startswith(('ret_', 'mfe_', 'mae_'))
        ]
        self._candidate_features = candidate_features

        distinct_horizons = resolve_distinct_horizons(self.config.horizons, bar_seconds)
        if len(distinct_horizons) < len(self.config.horizons):
            logger.info(
                f"MIE fit: {len(self.config.horizons)} configured horizons collapse to "
                f"{len(distinct_horizons)} distinct at {bar_seconds}s bars — "
                f"{self.config.horizons} -> {distinct_horizons}")

        # Drop any previously-fit model whose horizon isn't in the current
        # distinct set — e.g. after a timeframe change, a horizon that used to
        # be distinct may now alias another, and a stale model must not linger
        # and be consulted as if it still reflected the current bar interval.
        for stale in set(self.edge_models) - set(distinct_horizons):
            del self.edge_models[stale]

        summaries = {}
        for h in distinct_horizons:
            model = EdgeModel(horizon_sec=h)
            model.fit(training_df, candidate_features, bar_seconds)
            self.edge_models[h] = model
            summaries[h] = model.report.reason if model.report else 'not fit'
        return summaries

    @property
    def any_validated(self) -> bool:
        return any(m.validated for m in self.edge_models.values())

    # ── live decision ──────────────────────────────────────────────────

    def decide(self, symbol: str, df: pd.DataFrame,
              references: Optional[Dict[str, pd.DataFrame]] = None,
              book: Optional[Dict[str, Any]] = None,
              trades: Optional[List[Dict[str, Any]]] = None,
              derivs: Optional[Dict[str, Any]] = None,
              user_id: Optional[int] = None) -> TradeDecision:
        state = self.feature_engine.build_state(symbol, df, references, book, trades, derivs)
        regime, regime_conf = self.regime_model.classify(state.features)
        state.regime, state.regime_confidence = regime, regime_conf

        if self.store is not None:
            self.store.record(user_id or 0, state)

        decision = TradeDecision(symbol=symbol, as_of=state.as_of, regime=regime, regime_confidence=regime_conf)

        if regime in HOSTILE_REGIMES:
            decision.blockers.append(f"hostile regime for trading: {regime}")
            return decision

        if regime_conf < self.config.min_regime_confidence:
            decision.blockers.append(
                f"regime confidence too low ({regime_conf:.0%} < {self.config.min_regime_confidence:.0%})")
            return decision

        atr_pct = state.features.get(f'{GROUP_PRICE}_atr_pct')
        cost_est = (estimate_costs_from_book(book, self.config.probe_notional, self.config.fee_tier)
                   if book else estimate_costs_fallback(atr_pct=atr_pct, fee_tier=self.config.fee_tier))

        if not self.edge_models:
            decision.reasons.append("no edge models trained yet")
            return decision

        best = self._best_candidate(state, cost_est)
        if best is None:
            decision.reasons.append("no validated model currently shows positive net expectancy here")
            decision.cost_detail = cost_est
            return decision

        horizon_sec, direction, fc, net_return, stop_distance, expectancy_r = best
        target_distance = max(abs(fc.expected_mfe), stop_distance * 1.2)

        blockers: List[str] = []
        if fc.sample_size < self.config.min_pooled_trades:
            blockers.append(f"only {fc.sample_size} OOS observations behind this model (need "
                            f"{self.config.min_pooled_trades})")
        if fc.model_confidence < self.config.min_model_confidence:
            blockers.append(f"model reliability {fc.model_confidence:.2f} below floor "
                            f"{self.config.min_model_confidence:.2f}")
        if not cost_est.fillable:
            blockers.append("insufficient visible depth to fill this size")
        gross = abs(fc.expected_return)
        cost_ratio = cost_est.total / gross if gross > 1e-9 else 1.0
        if cost_ratio > self.config.max_cost_to_return_ratio:
            blockers.append(f"costs consume {cost_ratio:.0%} of the expected move "
                            f"(cap {self.config.max_cost_to_return_ratio:.0%})")

        historical_sample = fc.sample_size
        neighbor_conf = 1.0  # neutral multiplier unless neighbors actively disagree
        if self.store is not None:
            model = self.edge_models[horizon_sec]
            neighbor_stats = self._neighbor_corroboration(state, model, direction, horizon_sec)
            if neighbor_stats is not None:
                historical_sample += neighbor_stats['k']
                if neighbor_stats['k'] >= self.config.min_neighbor_k:
                    if neighbor_stats['mean_ret'] * direction < 0 and neighbor_stats['win_rate'] < 0.40:
                        blockers.append(
                            f"nearest-neighbor history disagrees "
                            f"({neighbor_stats['k']} similar states, {neighbor_stats['win_rate']:.0%} favorable)")
                        neighbor_conf = 0.5
                    elif neighbor_stats['mean_ret'] * direction > 0:
                        neighbor_conf = 1.1

        quality = self._quality_score(fc, regime_conf, cost_ratio, neighbor_conf)

        decision.horizon_sec = horizon_sec
        decision.expected_return = net_return
        decision.expected_adverse = fc.expected_mae   # already sign-flipped per direction by EdgeModel.predict
        decision.prob_positive = fc.prob_positive
        decision.costs = cost_est.total
        decision.historical_sample = historical_sample
        decision.expectancy_r = expectancy_r
        decision.quality = quality
        decision.stop_distance = stop_distance
        decision.target_distance = target_distance
        decision.forecast = fc
        decision.cost_detail = cost_est
        decision.blockers = blockers

        if blockers or quality < self.config.min_quality:
            if not blockers:
                blockers.append(f"quality {quality}/100 below threshold {self.config.min_quality}")
                decision.blockers = blockers
            decision.action = ACTION_NOTHING
            return decision

        decision.action = ACTION_LONG if direction == 1 else ACTION_SHORT
        decision.size_fraction = self._size_fraction(fc.prob_positive, stop_distance, target_distance)
        decision.reasons.append(
            f"validated {horizon_label(horizon_sec)} model, {fc.sample_size} OOS trades, "
            f"{historical_sample - fc.sample_size} corroborating neighbors")
        return decision

    # ── internals ──────────────────────────────────────────────────────

    def _best_candidate(self, state: MarketState, cost_est) -> Optional[tuple]:
        """
        Across every validated horizon and both directions, pick the candidate
        with the highest expectancy-per-unit-risk (R-multiple) — NOT the highest
        raw expected return. A trade with a smaller edge but a tighter, more
        reliable stop can beat a flashier one, exactly the "casino, not gambler"
        framing: consistency of edge per unit of risk compounds, a lucky big
        number does not.
        """
        best = None
        for h, model in self.edge_models.items():
            if not model.validated:
                continue
            forecasts = model.predict(state)
            for direction, fc in forecasts.items():
                net_return = fc.expected_return - cost_est.total
                if net_return <= 0:
                    continue
                stop = max(abs(fc.expected_mae), self.config.stop_floor)
                expectancy_r = net_return / stop
                if best is None or expectancy_r > best[5]:
                    best = (h, direction, fc, net_return, stop, expectancy_r)
        return best

    def _neighbor_corroboration(self, state: MarketState, model: EdgeModel,
                                direction: int, horizon_sec: int) -> Optional[Dict[str, float]]:
        if not model.feature_columns:
            return None
        try:
            neighbors, dists = self.store.similar_observations(
                state, model.feature_columns, k=max(50, self.config.min_neighbor_k))
        except Exception as e:
            logger.debug(f"neighbor lookup failed: {e}")
            return None
        if not neighbors:
            return None
        lab = horizon_label(horizon_sec)
        col = f'ret_{lab}'
        rets = [n[col] * direction for n in neighbors if n.get(col) is not None]
        if not rets:
            return None
        return {'k': len(rets), 'mean_ret': float(np.mean(rets)),
               'win_rate': float(np.mean([r > 0 for r in rets]))}

    def _quality_score(self, fc: EdgeForecast, regime_conf: float,
                       cost_ratio: float, neighbor_conf: float) -> int:
        """
        0-100 composite. Every component is itself already a 0-1 measure of
        trustworthiness (not of raw signal strength) — quality is meant to
        answer "how much should we believe this", not "how big is the number".
        """
        sample_component = min(1.0, fc.sample_size / 200.0)
        cost_component = max(0.0, 1.0 - cost_ratio / 0.65)
        prob_edge_component = min(1.0, abs(fc.prob_positive - 0.5) * 4)
        score = (0.35 * fc.model_confidence + 0.20 * regime_conf +
                0.20 * sample_component + 0.15 * cost_component +
                0.10 * prob_edge_component) * neighbor_conf
        return int(round(max(0.0, min(1.0, score)) * 100))

    def _size_fraction(self, prob_positive: float, stop_distance: float,
                       target_distance: float) -> float:
        """
        Half-Kelly on the R-multiple payoff (target/stop), not on the raw
        percentage return — sizing follows the risk taken, not the price of the
        asset. Bounded well inside account-blowup range regardless of what the
        formula outputs, because a formula is a starting point, not a promise.
        """
        p = max(0.01, min(0.99, prob_positive))
        b = max(0.1, target_distance / max(stop_distance, 1e-9))
        kelly = (p * b - (1 - p)) / b
        half = max(0.0, kelly) * 0.5
        return round(max(self.config.risk_per_trade_floor,
                         min(self.config.risk_per_trade_cap, half)), 5)
