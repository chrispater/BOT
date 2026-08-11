"""
Out-of-sample validation — the anti-overfitting core of the whole engine.

Financial ML dies from one failure mode more than any other: a relationship
that looks great on the data it was fit to and evaporates (or reverses) the
moment it meets data it hasn't seen. This module exists to make that failure
visible before a model reaches production, using purged walk-forward
cross-validation:

  • WALK-FORWARD — folds are chronological. We never train on the future and
    test on the past; every test fold comes strictly after its training data,
    the way live trading actually works.
  • PURGED — a training sample near the train/test boundary has a label built
    from a forward window that can reach INTO the test period, which would let
    the model see test-period price action during training. We drop
    (`purge`) any training row whose label window overlaps the test window, by
    embargoing a gap of at least `embargo_bars` immediately before each test
    block.

A model is not "validated" because it looks good in-sample. It is validated
because summing up its predictions across held-out folds it has never
influenced produces positive expectancy, with enough trades to mean something,
and — critically — that expectancy isn't manufactured by a single lucky fold.
"""

from dataclasses import dataclass, field
from typing import Iterator, Tuple, List, Optional
import numpy as np


def purged_walk_forward_splits(n: int, n_splits: int = 5, embargo: int = 60,
                               min_train: int = 200) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, test_idx) over a chronologically ordered array of length n.

    Each fold's test block is a contiguous, non-overlapping future segment; the
    train set is an EXPANDING window of everything before it, truncated by
    `embargo` bars immediately preceding the test block. `embargo` must be at
    least as large as the longest label horizon (in bars) used anywhere in the
    training target — otherwise labels straddling the boundary leak test-period
    information into training.
    """
    if n <= min_train + embargo + 1:
        return
    usable = n - min_train
    fold_size = max(1, usable // n_splits)
    for i in range(n_splits):
        test_start = min_train + i * fold_size
        test_end = n if i == n_splits - 1 else min(n, min_train + (i + 1) * fold_size)
        if test_start >= test_end:
            continue
        train_end = max(0, test_start - embargo)
        if train_end < min_train // 2:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    auc: float = float('nan')
    ret_corr: float = float('nan')
    expectancy: float = float('nan')      # mean net return per trade taken, this fold
    win_rate: float = float('nan')
    n_trades: int = 0


@dataclass
class ValidationReport:
    horizon_sec: int
    n_splits_run: int = 0
    folds: List[FoldResult] = field(default_factory=list)
    pooled_auc: float = float('nan')
    pooled_ret_corr: float = float('nan')
    pooled_expectancy: float = float('nan')
    pooled_win_rate: float = float('nan')
    pooled_trades: int = 0
    folds_with_edge: int = 0
    validated: bool = False
    reason: str = ''

    def summary(self) -> str:
        lines = [f"horizon={self.horizon_sec}s  folds_run={self.n_splits_run}  "
                 f"validated={self.validated}  ({self.reason})"]
        lines.append(f"  pooled OOS: auc={self.pooled_auc:.3f} ret_corr={self.pooled_ret_corr:.3f} "
                     f"expectancy={self.pooled_expectancy:+.4%} win_rate={self.pooled_win_rate:.1%} "
                     f"n_trades={self.pooled_trades} folds_with_edge={self.folds_with_edge}/{self.n_splits_run}")
        for f in self.folds:
            lines.append(f"    fold {f.fold}: n_train={f.n_train} n_test={f.n_test} "
                         f"auc={f.auc:.3f} corr={f.ret_corr:.3f} "
                         f"expectancy={f.expectancy:+.4%} trades={f.n_trades}")
        return "\n".join(lines)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float('nan')
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float('nan')


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float('nan')
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float('nan')
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def evaluate_validation(reports_per_fold: List[FoldResult], horizon_sec: int,
                        min_total_trades: int = 30,
                        min_expectancy: float = 0.0,
                        min_fold_edge_frac: float = 0.5,
                        min_folds: int = 3) -> ValidationReport:
    """
    Roll individual fold results up into a pass/fail verdict.

    Requires ALL of:
      • at least `min_folds` folds actually ran (enough distinct time periods —
        "different months and different volatility regimes" per the underlying
        research concern, not one lucky window)
      • pooled OOS expectancy (net of the cost assumption baked into each fold's
        trade simulation) is strictly positive and exceeds `min_expectancy`
      • pooled trade count clears `min_total_trades` — an edge measured on 12
        trades is a coin flip with a story attached
      • at least `min_fold_edge_frac` of individual folds show non-negative
        expectancy on their own — guards against one anomalous fold carrying
        an otherwise-losing model
    """
    folds = [f for f in reports_per_fold if f.n_trades > 0]
    report = ValidationReport(horizon_sec=horizon_sec, n_splits_run=len(reports_per_fold), folds=reports_per_fold)
    if not folds:
        report.reason = 'no fold produced any trades'
        return report

    total_trades = sum(f.n_trades for f in folds)
    pooled_expectancy = sum(f.expectancy * f.n_trades for f in folds) / max(1, total_trades)
    pooled_wins = sum(f.win_rate * f.n_trades for f in folds) / max(1, total_trades)
    aucs = [f.auc for f in folds if not np.isnan(f.auc)]
    corrs = [f.ret_corr for f in folds if not np.isnan(f.ret_corr)]

    report.pooled_expectancy = pooled_expectancy
    report.pooled_win_rate = pooled_wins
    report.pooled_trades = total_trades
    report.pooled_auc = float(np.mean(aucs)) if aucs else float('nan')
    report.pooled_ret_corr = float(np.mean(corrs)) if corrs else float('nan')
    report.folds_with_edge = sum(1 for f in folds if f.expectancy >= 0)

    if len(reports_per_fold) < min_folds:
        report.reason = f'only {len(reports_per_fold)} folds ran (need {min_folds})'
        return report
    if total_trades < min_total_trades:
        report.reason = f'only {total_trades} OOS trades (need {min_total_trades})'
        return report
    if pooled_expectancy <= min_expectancy:
        report.reason = f'pooled OOS expectancy {pooled_expectancy:+.4%} is not positive'
        return report
    edge_frac = report.folds_with_edge / len(folds)
    if edge_frac < min_fold_edge_frac:
        report.reason = (f'edge concentrated in too few folds '
                         f'({report.folds_with_edge}/{len(folds)} folds ≥0)')
        return report

    report.validated = True
    report.reason = 'passed OOS validation'
    return report
