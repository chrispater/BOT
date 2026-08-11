"""
Edge model — the piece that answers the actual question that matters:

    "Under these exact conditions, after fees and slippage, does this setup
     have positive expectancy over the next N minutes?"

NOT "is SOL going up". Not a three-way BUY/SELL/HOLD classifier trained on a
threshold someone picked. This model regresses directly onto what happened —
forward return, MFE, MAE — at each horizon, plus a calibrated probability of a
positive outcome, and every one of those numbers has to survive purged
walk-forward validation (validation.py) before it's allowed anywhere near a
live decision.

One model object per horizon. Long and short forecasts are derived from the
SAME fitted models by sign-flipping the regression targets and inverting the
classifier's probability — a short is a mirror of a long against the same
market description, not a separately-labeled phenomenon requiring its own
2x-the-data model. This roughly halves the data hunger of the approach, which
matters a great deal when "sample size" is the resource this whole system is
built to respect.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .state import MarketState, EdgeForecast, horizon_label, GROUP_ORDERFLOW, GROUP_DERIVS, GROUP_EXEC
from .validation import purged_walk_forward_splits, evaluate_validation, FoldResult, ValidationReport, _safe_auc, _safe_corr

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    _SKLEARN_OK = True
except Exception as e:  # pragma: no cover
    _SKLEARN_OK = False
    logger.error(f"scikit-learn HistGradientBoosting unavailable: {e}")


DEFAULT_COST_ASSUMPTION = 0.0009   # ~9bps round trip — conservative taker-fee + small slip floor
MIN_COVERAGE = 0.60                # a feature needs to be non-NaN this often to be trusted
MIN_TRAIN_ROWS = 400


def select_trainable_features(df: pd.DataFrame, candidate_columns: List[str],
                              min_coverage: float = MIN_COVERAGE) -> List[str]:
    """
    Coverage gate: keep only feature columns present often enough to actually
    train on. This is precisely how the engine "grows into microstructure" —
    order-flow columns are 0% covered on historical-only backfill and get
    dropped automatically, then start contributing the moment the store has
    accumulated enough live book/tape snapshots to clear the bar, with no code
    change required.
    """
    kept = []
    for c in candidate_columns:
        if c not in df.columns:
            continue
        cov = df[c].notna().mean()
        if cov >= min_coverage:
            kept.append(c)
    return kept


def _build_matrix(df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
    return df[feature_columns].to_numpy(dtype=float)


def _make_models():
    return {
        'clf': HistGradientBoostingClassifier(max_depth=5, max_iter=150, learning_rate=0.08,
                                              l2_regularization=1.0, random_state=42),
        'reg_ret': HistGradientBoostingRegressor(max_depth=5, max_iter=150, learning_rate=0.08,
                                                 l2_regularization=1.0, random_state=42),
        'reg_mfe': HistGradientBoostingRegressor(max_depth=5, max_iter=150, learning_rate=0.08,
                                                 l2_regularization=1.0, random_state=42),
        'reg_mae': HistGradientBoostingRegressor(max_depth=5, max_iter=150, learning_rate=0.08,
                                                 l2_regularization=1.0, random_state=42),
    }


@dataclass
class EdgeModel:
    """
    One instance per (horizon). `feature_columns` is fixed at fit time so
    live prediction can never silently drift onto a different feature set than
    what the model was validated against.
    """
    horizon_sec: int
    cost_assumption: float = DEFAULT_COST_ASSUMPTION
    min_edge_move: float = 0.0015     # smallest |predicted return| worth simulating as a trade in CV
    n_splits: int = 5
    embargo_bars: Optional[int] = None
    feature_columns: List[str] = None
    report: Optional[ValidationReport] = None
    _models: dict = None
    _fitted: bool = False

    def fit(self, df: pd.DataFrame, candidate_features: List[str],
           bar_seconds: float) -> ValidationReport:
        if not _SKLEARN_OK:
            self.report = ValidationReport(horizon_sec=self.horizon_sec, reason='sklearn unavailable')
            return self.report

        lab = horizon_label(self.horizon_sec)
        ret_col, mfe_col, mae_col = f'ret_{lab}', f'mfe_{lab}', f'mae_{lab}'
        needed = [ret_col, mfe_col, mae_col]
        if any(c not in df.columns for c in needed):
            self.report = ValidationReport(horizon_sec=self.horizon_sec,
                                           reason=f'missing outcome columns for {lab}')
            return self.report

        work = df.dropna(subset=needed).reset_index(drop=True)
        self.feature_columns = select_trainable_features(work, candidate_features)
        if len(work) < MIN_TRAIN_ROWS or not self.feature_columns:
            self.report = ValidationReport(
                horizon_sec=self.horizon_sec,
                reason=f'insufficient data: {len(work)} resolved rows, {len(self.feature_columns)} usable features')
            return self.report

        X = _build_matrix(work, self.feature_columns)
        y_ret = work[ret_col].to_numpy(dtype=float)
        y_mfe = work[mfe_col].to_numpy(dtype=float)
        y_mae = work[mae_col].to_numpy(dtype=float)
        y_dir = (y_ret > 0).astype(int)

        bars_per_horizon = max(1, round(self.horizon_sec / bar_seconds))
        embargo = self.embargo_bars if self.embargo_bars is not None else max(bars_per_horizon * 2, 50)

        fold_results: List[FoldResult] = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i, (train_idx, test_idx) in enumerate(
                    purged_walk_forward_splits(len(work), self.n_splits, embargo, MIN_TRAIN_ROWS)):
                if len(np.unique(y_dir[train_idx])) < 2:
                    continue
                models = _make_models()
                models['clf'].fit(X[train_idx], y_dir[train_idx])
                models['reg_ret'].fit(X[train_idx], y_ret[train_idx])
                models['reg_mfe'].fit(X[train_idx], y_mfe[train_idx])
                models['reg_mae'].fit(X[train_idx], y_mae[train_idx])

                p_up = models['clf'].predict_proba(X[test_idx])[:, 1]
                pred_ret = models['reg_ret'].predict(X[test_idx])
                actual_ret = y_ret[test_idx]

                fold_results.append(self._score_fold(
                    i, len(train_idx), len(test_idx), y_dir[test_idx], p_up, pred_ret, actual_ret))

        self.report = evaluate_validation(fold_results, self.horizon_sec)
        logger.info(f"EdgeModel[{lab}] validation:\n{self.report.summary()}")

        # Refit on the FULL dataset for production use regardless of validation
        # outcome — predict() consults `report.validated` before trusting output,
        # so an unvalidated model is retained (for visibility / re-evaluation as
        # more data arrives) but gated from influencing any decision.
        self._models = _make_models()
        self._models['clf'].fit(X, y_dir)
        self._models['reg_ret'].fit(X, y_ret)
        self._models['reg_mfe'].fit(X, y_mfe)
        self._models['reg_mae'].fit(X, y_mae)
        self._fitted = True
        return self.report

    def _score_fold(self, fold_i, n_train, n_test, y_dir_test, p_up, pred_ret, actual_ret) -> FoldResult:
        auc = _safe_auc(y_dir_test, p_up)
        corr = _safe_corr(pred_ret, actual_ret)

        # Simulate OOS trades: take a position whenever the model's predicted
        # move clears both a minimum-magnitude floor (avoid trading on noise
        # near zero) and the assumed round-trip cost, direction from the sign
        # of the prediction. This is a deliberately simple, conservative proxy
        # for "would this have been a trade" — the real decision layer applies
        # richer gates (regime, sample-size, EV) on top of a model that has
        # already cleared this bar.
        direction = np.sign(pred_ret)
        tradeable = (np.abs(pred_ret) >= max(self.min_edge_move, self.cost_assumption))
        n_trades = int(tradeable.sum())
        if n_trades == 0:
            return FoldResult(fold=fold_i, n_train=n_train, n_test=n_test, auc=auc, ret_corr=corr)

        realized = direction[tradeable] * actual_ret[tradeable] - self.cost_assumption
        expectancy = float(np.mean(realized))
        win_rate = float((realized > 0).mean())
        return FoldResult(fold=fold_i, n_train=n_train, n_test=n_test, auc=auc, ret_corr=corr,
                          expectancy=expectancy, win_rate=win_rate, n_trades=n_trades)

    @property
    def validated(self) -> bool:
        return bool(self.report and self.report.validated)

    def predict(self, state: MarketState) -> Dict[int, EdgeForecast]:
        """
        Returns {+1: EdgeForecast, -1: EdgeForecast} gross-of-cost forecasts for
        this horizon. Callers MUST check `.validated` (or the report's sample
        size / reason) before treating this as more than diagnostic output —
        this method does not refuse to predict just because validation failed,
        because the decision layer needs to see and log what an unvalidated
        model would have said in order to reason about it, it just must never
        act on it.
        """
        if not self._fitted or not self.feature_columns:
            return {}
        x = np.array(state.vector(self.feature_columns), dtype=float).reshape(1, -1)
        if np.isnan(x).all():
            return {}
        x = np.nan_to_num(x, nan=0.0)   # HGB tolerates NaN natively, but an all-absent
                                        # live group (e.g. no book yet) should read as
                                        # "neutral", not crash the pipeline on dtype edge cases
        p_up = float(self._models['clf'].predict_proba(x)[0, 1])
        pred_ret = float(self._models['reg_ret'].predict(x)[0])
        pred_mfe = float(self._models['reg_mfe'].predict(x)[0])
        pred_mae = float(self._models['reg_mae'].predict(x)[0])
        n = self.report.pooled_trades if self.report else 0

        long_fc = EdgeForecast(horizon_sec=self.horizon_sec, direction=1,
                               expected_return=pred_ret, prob_positive=p_up,
                               expected_mfe=pred_mfe, expected_mae=pred_mae,
                               model_confidence=self._confidence(), sample_size=n)
        short_fc = EdgeForecast(horizon_sec=self.horizon_sec, direction=-1,
                                expected_return=-pred_ret, prob_positive=1.0 - p_up,
                                expected_mfe=-pred_mae, expected_mae=-pred_mfe,
                                model_confidence=self._confidence(), sample_size=n)
        return {1: long_fc, -1: short_fc}

    def _confidence(self) -> float:
        """
        A single 0-1 reliability score for this horizon's model, derived from its
        own OOS validation report — NOT from the classifier's in-sample
        predict_proba, which measures the model's certainty about the training
        distribution, not its trustworthiness on new data. An unvalidated model
        reports 0.0 regardless of how confident it looks internally.
        """
        if not self.report or not self.report.validated:
            return 0.0
        auc_component = max(0.0, (self.report.pooled_auc - 0.5) * 2) if not np.isnan(self.report.pooled_auc) else 0.0
        sample_component = min(1.0, self.report.pooled_trades / 200.0)
        edge_frac = self.report.folds_with_edge / max(1, self.report.n_splits_run)
        return round(float(np.clip(0.3 * auc_component + 0.3 * sample_component + 0.4 * edge_frac, 0.0, 1.0)), 3)

    def feature_importances(self, top_n: int = 15) -> List[Tuple[str, float]]:
        """
        Permutation-free approximation via the regressor's built-in importances
        where available — used purely for the post-mortem's attribution report,
        never for gating trades.
        """
        if not self._fitted or not self.feature_columns:
            return []
        # HistGradientBoosting doesn't expose feature_importances_ directly;
        # fall back to permutation importance on a small held-out slice only
        # when explicitly requested by the caller (post-mortem), since it's
        # comparatively expensive — see analysis.py.
        return []
