"""
Post-mortem / attribution — turning a day's (or week's) closed trades into the
answer to "why did this actually work", not just "did it work".

This is the module that produces the report described as the transformative
use of AI here: not predicting the next candle, but explaining, with numbers,
which CONDITIONS carried the edge — so tomorrow's engine can trade only the
subset of situations that actually have one. Every finding below is a plain
groupby over closed decisions joined to the exact feature state that produced
them; there is deliberately no LLM or opaque scoring in this file. A daily
narrative report can be handed to an actual LLM to phrase in prose (obviously
easy to layer on), but the numbers it reports have to be traceable to a
specific SQL-groupby-shaped computation, or the post-mortem becomes exactly the
kind of unfalsifiable "AI insight" this whole system is trying to replace.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd

EXCLUDED_COLS = {
    'symbol', 'as_of', 'action', 'horizon_sec', 'expected_return', 'expected_adverse',
    'prob_positive', 'costs', 'regime', 'regime_confidence', 'historical_sample',
    'expectancy_r', 'quality', 'size_fraction', 'realized_r', 'realized_pnl', 'closed_at',
}


@dataclass
class SliceStat:
    label: str
    n: int
    total_pnl: float
    pnl_share: float          # this slice's share of TOTAL pnl across all slices of the same cut
    win_rate: float
    mean_r: float


@dataclass
class FeatureAttribution:
    feature: str
    best_quartile: str        # e.g. 'Q4 (top 25%)'
    best_quartile_pnl_share: float
    best_quartile_mean_r: float
    n_in_quartile: int
    concentrated: bool        # True when one quartile is carrying a disproportionate share
    note: str


@dataclass
class PostMortemReport:
    n_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    expectancy_r: float = 0.0
    by_regime: List[SliceStat] = field(default_factory=list)
    by_quality_bucket: List[SliceStat] = field(default_factory=list)
    feature_attribution: List[FeatureAttribution] = field(default_factory=list)
    narrative: List[str] = field(default_factory=list)
    pruning_suggestion: Optional[Dict[str, Any]] = None

    def render(self) -> str:
        lines = [f"Post-mortem — {self.n_trades} closed trades, net P&L ${self.total_pnl:+.2f}, "
                f"win rate {self.win_rate:.0%}, expectancy {self.expectancy_r:+.3f}R"]
        lines.append("")
        for n in self.narrative:
            lines.append(f"  • {n}")
        if self.pruning_suggestion:
            p = self.pruning_suggestion
            lines.append("")
            lines.append(f"  Removing trades where {p['condition']} would have cut trade count by "
                        f"{p['trade_reduction_pct']:.0%} and moved expectancy from "
                        f"{p['baseline_expectancy']:+.3f}R to {p['pruned_expectancy']:+.3f}R.")
        return "\n".join(lines)


def _slice_stats(df: pd.DataFrame, group_col: str) -> List[SliceStat]:
    if group_col not in df.columns or df.empty:
        return []
    total_abs_pnl = df['realized_pnl'].abs().sum() or 1.0
    out = []
    for label, g in df.groupby(group_col):
        pnl = float(g['realized_pnl'].sum())
        out.append(SliceStat(
            label=str(label), n=len(g), total_pnl=pnl,
            pnl_share=float(g['realized_pnl'].sum() / total_abs_pnl) if total_abs_pnl else 0.0,
            win_rate=float((g['realized_r'] > 0).mean()),
            mean_r=float(g['realized_r'].mean()),
        ))
    return sorted(out, key=lambda s: s.total_pnl, reverse=True)


def _feature_quartile_attribution(df: pd.DataFrame, feature: str,
                                  min_n: int = 5, concentration_threshold: float = 0.60) -> Optional[FeatureAttribution]:
    """
    Bucket trades into quartiles of `feature`'s value and check whether one
    quartile is carrying a disproportionate share of total P&L — the exact shape
    of the letter's example ("91% of today's profit occurred when 5-minute
    realized volatility was above the 70th percentile").
    """
    col = df[feature].astype(float)
    if col.notna().sum() < min_n * 4:
        return None
    try:
        quartile = pd.qcut(col, 4, labels=['Q1 (bottom 25%)', 'Q2', 'Q3', 'Q4 (top 25%)'], duplicates='drop')
    except ValueError:
        return None  # not enough distinct values to form quartiles

    total_pnl = df['realized_pnl'].sum()
    grouped = df.groupby(quartile, observed=True)['realized_pnl'].sum()
    if total_pnl == 0 or grouped.empty:
        return None
    shares = grouped / (df['realized_pnl'].abs().sum() or 1.0)
    best_q = shares.idxmax()
    best_share = float(shares.max())
    n_in_q = int((quartile == best_q).sum())
    if n_in_q < min_n:
        return None
    mean_r_q = float(df.loc[quartile == best_q, 'realized_r'].mean())
    concentrated = best_share >= concentration_threshold
    note = (f"{best_share:.0%} of net P&L came from trades in {best_q} of {feature} "
           f"(n={n_in_q}, mean {mean_r_q:+.3f}R)")
    return FeatureAttribution(feature=feature, best_quartile=str(best_q),
                              best_quartile_pnl_share=best_share, best_quartile_mean_r=mean_r_q,
                              n_in_quartile=n_in_q, concentrated=concentrated, note=note)


def _pruning_suggestion(df: pd.DataFrame, attributions: List[FeatureAttribution]) -> Optional[Dict[str, Any]]:
    """
    For the single most concentrated feature finding, simulate what expectancy
    would look like with the WORST-performing quartile removed — the "removing
    those conditions would have reduced trades by 38% but increased expectancy
    from 0.11R to 0.34R" idea. Purely descriptive of history; the caller decides
    whether to actually act on it (and should require this to reproduce across
    multiple post-mortems before hard-coding it into the live gate).
    """
    concentrated = [a for a in attributions if a.concentrated]
    if not concentrated:
        return None
    top = max(concentrated, key=lambda a: a.best_quartile_pnl_share)
    col = df[top.feature].astype(float)
    try:
        quartile = pd.qcut(col, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    except ValueError:
        return None
    worst_q = df.groupby(quartile, observed=True)['realized_pnl'].sum().idxmin()
    kept = df[quartile != worst_q]
    if kept.empty or len(kept) == len(df):
        return None
    baseline_exp = float(df['realized_r'].mean())
    pruned_exp = float(kept['realized_r'].mean())
    return {
        'condition': f"{top.feature} in its worst quartile ({worst_q})",
        'trade_reduction_pct': 1.0 - len(kept) / len(df),
        'baseline_expectancy': baseline_exp,
        'pruned_expectancy': pruned_exp,
    }


def run_post_mortem(decisions_with_features: pd.DataFrame,
                    feature_columns: Optional[List[str]] = None,
                    min_trades: int = 10) -> PostMortemReport:
    """
    `decisions_with_features` is store.load_decisions_with_features(...) output:
    one row per CLOSED decision, with realized_r / realized_pnl filled in and the
    full feature vector that produced the decision alongside it.
    """
    df = decisions_with_features.dropna(subset=['realized_r', 'realized_pnl']).copy()
    report = PostMortemReport(n_trades=len(df))
    if len(df) < min_trades:
        report.narrative.append(
            f"Only {len(df)} closed trades on record — need at least {min_trades} before "
            f"attribution is meaningful. Keep trading; this report will fill in.")
        return report

    report.total_pnl = float(df['realized_pnl'].sum())
    report.win_rate = float((df['realized_r'] > 0).mean())
    report.expectancy_r = float(df['realized_r'].mean())

    report.by_regime = _slice_stats(df, 'regime')
    if report.by_regime:
        best = report.by_regime[0]
        report.narrative.append(
            f"Regime '{best.label}' contributed the most: {best.pnl_share:.0%} of net P&L "
            f"across {best.n} trades ({best.win_rate:.0%} win rate, {best.mean_r:+.3f}R avg).")
        losers = [s for s in report.by_regime if s.total_pnl < 0]
        if losers:
            worst = min(losers, key=lambda s: s.total_pnl)
            report.narrative.append(
                f"Regime '{worst.label}' lost ${abs(worst.total_pnl):.2f} across {worst.n} trades "
                f"— consider whether it should gate out entirely rather than size down.")

    if 'quality' in df.columns:
        df['_quality_bucket'] = pd.cut(df['quality'], bins=[0, 60, 75, 90, 100],
                                       labels=['<60', '60-75', '75-90', '90+'], include_lowest=True)
        report.by_quality_bucket = _slice_stats(df, '_quality_bucket')
        if len(report.by_quality_bucket) >= 2:
            hi = max(report.by_quality_bucket, key=lambda s: {'90+': 4, '75-90': 3, '60-75': 2, '<60': 1}.get(s.label, 0))
            lo = min(report.by_quality_bucket, key=lambda s: {'90+': 4, '75-90': 3, '60-75': 2, '<60': 1}.get(s.label, 0))
            if hi.mean_r > lo.mean_r:
                report.narrative.append(
                    f"Quality score is doing its job: bucket {hi.label} averaged {hi.mean_r:+.3f}R "
                    f"vs {lo.mean_r:+.3f}R in {lo.label}.")
            else:
                report.narrative.append(
                    f"Quality score is NOT well calibrated right now: bucket {lo.label} outperformed "
                    f"{hi.label} ({lo.mean_r:+.3f}R vs {hi.mean_r:+.3f}R) — worth re-fitting the model.")

    candidate_features = feature_columns or [c for c in df.columns if c not in EXCLUDED_COLS
                                             and c != '_quality_bucket' and df[c].dtype != object]
    attributions = []
    for feat in candidate_features:
        try:
            a = _feature_quartile_attribution(df, feat)
        except Exception:
            a = None
        if a is not None:
            attributions.append(a)
    attributions.sort(key=lambda a: a.best_quartile_pnl_share, reverse=True)
    report.feature_attribution = attributions[:10]
    for a in [a for a in attributions if a.concentrated][:3]:
        report.narrative.append(a.note)

    report.pruning_suggestion = _pruning_suggestion(df, attributions)
    if not report.narrative:
        report.narrative.append("No single regime, quality bucket or feature quartile dominates "
                                "P&L yet — expectancy looks broadly distributed across conditions.")
    return report
