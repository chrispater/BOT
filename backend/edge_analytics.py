"""
Edge analytics — the post-mortem layer of the Market Intelligence Engine.

This module does not predict anything. It answers one question, honestly:

    "Under which conditions has this system actually demonstrated an
     advantage, and is the sample large enough to believe it?"

Two data sources feed it:

  • trades       — realised outcomes in R-multiples, with the market context
                   captured at entry. Ground truth, but slow to accumulate.
  • observations — market state recorded every candle whether or not we traded,
                   with forward returns and MFE/MAE backfilled. Accumulates
                   ~1,440 rows/day at 5 coins on 5m bars, so it can characterise
                   conditions long before trade counts are meaningful.

Two statistical guards run throughout, because slicing a small sample many ways
is the single easiest way to fool yourself in this domain:

  1. MINIMUM SAMPLE. A bucket below `min_sample` is reported but never gates
     trading and is never called an edge.

  2. LOWER CONFIDENCE BOUND, not the point estimate. Expectancy is reported as
     mean - t * standard_error. Slice trades by regime x volatility x setup x
     hour and some bucket will look brilliant by luck; a bound scaled to that
     bucket's own sample size is what luck fails to survive. Every gating
     decision uses the bound, never the mean.

R is the unit throughout: one R = the configured stop distance in margin terms.
Expectancy in R is comparable across position sizes, leverage and coins, which
percent-return is not.
"""

import math
from collections import defaultdict

# t-multipliers for a one-sided ~95% bound, by degrees of freedom. Small samples
# get a materially wider haircut than the normal approximation would allow.
_T95 = [(5, 2.132), (10, 1.833), (15, 1.753), (20, 1.729), (30, 1.699),
        (60, 1.671), (120, 1.658)]


def _t_crit(n: int) -> float:
    if n < 2:
        return 6.314
    df = n - 1
    for cutoff, t in _T95:
        if df <= cutoff:
            return t
    return 1.645  # normal limit


def _stats(values: list) -> dict:
    """Mean, standard error and one-sided 95% lower bound for a sample."""
    n = len(values)
    if n == 0:
        return {'n': 0, 'mean': 0.0, 'lb': 0.0, 'sd': 0.0}
    mean = sum(values) / n
    if n == 1:
        return {'n': 1, 'mean': round(mean, 4), 'lb': round(min(0.0, mean), 4), 'sd': 0.0}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    return {
        'n': n,
        'mean': round(mean, 4),
        'sd': round(sd, 4),
        'lb': round(mean - _t_crit(n) * se, 4),
    }


def _f(row, key, default=None):
    """Tolerant field read — rows may be dicts or DB row objects, and Decimal
    values need coercing to float before arithmetic."""
    try:
        v = row[key] if key in row else row.get(key, default)
    except Exception:
        try:
            v = row.get(key, default)
        except Exception:
            return default
    if v is None:
        return default
    return v


def _num(row, key, default=None):
    v = _f(row, key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _vol_band(vol_pct) -> str:
    if vol_pct is None:
        return 'unknown'
    if vol_pct >= 0.6:
        return 'high'
    if vol_pct <= 0.4:
        return 'low'
    return 'mid'


def _session(hour) -> str:
    """UTC hour to trading session — liquidity and behaviour differ sharply."""
    if hour is None:
        return 'unknown'
    h = int(hour)
    if 0 <= h < 8:
        return 'asia'
    if 8 <= h < 13:
        return 'europe'
    if 13 <= h < 21:
        return 'us'
    return 'late'


def _bucket_keys(row) -> list:
    """Condition buckets a single trade belongs to. Deliberately a small, fixed
    set of interpretable slices — an unbounded search over feature combinations
    would generate false discoveries faster than any bound could suppress."""
    regime = _f(row, 'regime') or 'unknown'
    setup = _f(row, 'setup_name') or 'ml'
    vol = _vol_band(_num(row, 'vol_pct'))
    keys = [
        f'setup={setup}',
        f'regime={regime}',
        f'vol={vol}',
        f'regime={regime}|vol={vol}',
        f'session={_session(_f(row, "hour_utc"))}',
        f'symbol={_f(row, "symbol") or "unknown"}',
        f'side={_f(row, "side") or "unknown"}',
    ]
    ctx = _f(row, 'context') or {}
    if isinstance(ctx, dict):
        agree = ctx.get('btc_agree')
        if agree is not None:
            keys.append(f'btc_agree={bool(agree)}')
        bi = ctx.get('book_imbalance')
        if bi is not None:
            try:
                keys.append('book=bid_heavy' if float(bi) >= 0.55 else
                            ('book=ask_heavy' if float(bi) <= 0.45 else 'book=balanced'))
            except (TypeError, ValueError):
                pass
    return keys


def analyze_trades(trades: list, min_sample: int = 25) -> dict:
    """Conditional expectancy profile in R-multiples.

    Returns overall stats plus a per-bucket map consumed by the live trade
    quality gate. `expectancy_lb` is the gating number; `expectancy` (the mean)
    is for human reading only.
    """
    rows = [t for t in trades if _num(t, 'r_multiple') is not None]
    if not rows:
        return {'ok': False, 'reason': 'no trades with R-multiple recorded',
                'n': 0, 'buckets': {}, 'min_sample': min_sample}

    all_r = [_num(t, 'r_multiple') for t in rows]
    overall = _stats(all_r)
    wins = [r for r in all_r if r > 0]
    losses = [r for r in all_r if r <= 0]

    grouped = defaultdict(list)
    for t in rows:
        r = _num(t, 'r_multiple')
        for k in _bucket_keys(t):
            grouped[k].append(r)

    buckets = {}
    for k, vals in grouped.items():
        s = _stats(vals)
        w = [v for v in vals if v > 0]
        buckets[k] = {
            'n': s['n'],
            'expectancy': s['mean'],
            'expectancy_lb': s['lb'],
            'win_rate': round(len(w) / len(vals), 3) if vals else 0.0,
            'total_r': round(sum(vals), 2),
            'qualified': s['n'] >= min_sample,
        }

    return {
        'ok': True,
        'n': overall['n'],
        'expectancy': overall['mean'],
        'expectancy_lb': overall['lb'],
        'win_rate': round(len(wins) / len(all_r), 3),
        'avg_win_r': round(sum(wins) / len(wins), 3) if wins else 0.0,
        'avg_loss_r': round(sum(losses) / len(losses), 3) if losses else 0.0,
        'total_r': round(sum(all_r), 2),
        'buckets': buckets,
        'min_sample': min_sample,
    }


def analyze_observations(observations: list, horizon: str = 'ret_5',
                         min_sample: int = 100) -> dict:
    """Characterise conditions from observations rather than trades.

    Observations accumulate orders of magnitude faster than trades, so this is
    what makes the engine useful in week one instead of month six. It measures
    raw forward move by condition — no position sizing, no exit rule — which is
    the cleanest read on whether a condition carries information at all.

    Reported as directional edge: for candles where the engines leaned long, the
    forward return as-is; where they leaned short, its negation. A condition with
    no directional information averages ~0 and is correctly judged worthless.
    """
    rows = [o for o in observations
            if _num(o, horizon) is not None and _f(o, 'ml_signal') is not None]
    if not rows:
        return {'ok': False, 'reason': 'no completed observations', 'n': 0, 'buckets': {}}

    grouped = defaultdict(list)
    mfe_by = defaultdict(list)
    mae_by = defaultdict(list)
    for o in rows:
        sig = int(_f(o, 'ml_signal', 0) or 0)
        if sig == 0:
            continue
        fwd = _num(o, horizon, 0.0) * (1 if sig > 0 else -1)
        mfe = _num(o, 'mfe', 0.0)
        mae = _num(o, 'mae', 0.0)
        if sig < 0:
            mfe, mae = (-mae if mae is not None else None), (-mfe if mfe is not None else None)
        for k in _bucket_keys(o):
            grouped[k].append(fwd)
            if mfe is not None:
                mfe_by[k].append(mfe)
            if mae is not None:
                mae_by[k].append(mae)

    buckets = {}
    for k, vals in grouped.items():
        s = _stats(vals)
        buckets[k] = {
            'n': s['n'],
            'fwd_mean': s['mean'],
            'fwd_lb': s['lb'],
            'hit_rate': round(sum(1 for v in vals if v > 0) / len(vals), 3),
            'avg_mfe': round(sum(mfe_by[k]) / len(mfe_by[k]), 5) if mfe_by.get(k) else None,
            'avg_mae': round(sum(mae_by[k]) / len(mae_by[k]), 5) if mae_by.get(k) else None,
            'qualified': s['n'] >= min_sample,
        }
    return {'ok': True, 'n': len(rows), 'horizon': horizon,
            'buckets': buckets, 'min_sample': min_sample}


def cost_per_trade_r(stop_loss_pct: float, leverage: int,
                     maker_fee: float = 0.0002, taker_fee: float = 0.0006,
                     slippage: float = 0.0005) -> float:
    """Round-trip cost expressed in R — the hurdle every edge must clear.

    Reporting cost in the same unit as expectancy makes the comparison direct:
    an edge of +0.10R against a cost of 0.12R is a losing strategy, however good
    the win rate looks.
    """
    fee_margin = (maker_fee + taker_fee + slippage) * max(1, leverage)
    return round(fee_margin / max(stop_loss_pct, 1e-6), 4)


def build_edge_profile(trades: list, observations: list = None,
                       min_sample: int = 25) -> dict:
    """Assemble the profile consumed by the live quality gate.

    Trade-derived buckets take precedence — they are ground truth including
    costs and slippage. Observation-derived buckets fill gaps for conditions not
    yet traded enough, converted to a conservative R-equivalent and marked as
    provisional so their weaker provenance stays visible.
    """
    prof = analyze_trades(trades or [], min_sample=min_sample)
    buckets = dict(prof.get('buckets') or {})

    if observations:
        obs = analyze_observations(observations, min_sample=max(100, min_sample * 4))
        for k, b in (obs.get('buckets') or {}).items():
            if k in buckets and buckets[k].get('qualified'):
                continue  # real trades already speak for this condition
            if not b.get('qualified'):
                continue
            # Forward return is a fraction of PRICE; R is margin-relative. Without
            # this trade's leverage we cannot convert exactly, so scale by a
            # deliberately conservative factor and flag it provisional.
            buckets[k] = {
                'n': b['n'],
                'expectancy': round(b['fwd_mean'] * 10, 4),
                'expectancy_lb': round(b['fwd_lb'] * 10, 4),
                'win_rate': b['hit_rate'],
                'provisional': True,
                'qualified': True,
            }
        prof['observation_n'] = obs.get('n', 0)

    prof['buckets'] = buckets
    prof['min_sample'] = min_sample
    return prof


def format_postmortem(profile: dict, cost_r: float = None, top: int = 6) -> str:
    """Plain-language summary: where the money came from, and what to stop doing."""
    if not profile.get('ok'):
        return (f"Not enough data yet — {profile.get('reason', 'unknown')}. "
                f"The engine records every candle, so this fills in on its own.")

    lines = []
    exp, lb, n = profile['expectancy'], profile['expectancy_lb'], profile['n']
    lines.append(
        f"Overall: {n} trades · expectancy {exp:+.3f}R "
        f"(95% lower bound {lb:+.3f}R) · win rate {profile['win_rate']:.0%} · "
        f"avg win {profile['avg_win_r']:+.2f}R / avg loss {profile['avg_loss_r']:+.2f}R"
    )
    if cost_r is not None:
        verdict = "clears costs" if lb > cost_r else "does NOT clear costs"
        lines.append(f"Round-trip cost ≈ {cost_r:.3f}R — edge {verdict} on the lower bound.")
    if n < profile.get('min_sample', 25):
        lines.append(
            f"Sample below {profile['min_sample']} trades: treat all of this as "
            f"indicative only. No gating decisions should rest on it yet."
        )

    qualified = [(k, b) for k, b in profile['buckets'].items() if b.get('qualified')]
    if not qualified:
        lines.append(
            f"No condition bucket has reached {profile['min_sample']} trades yet, "
            f"so no bucket-level conclusions are available."
        )
        return "\n".join(lines)

    best = sorted(qualified, key=lambda kv: kv[1]['expectancy_lb'], reverse=True)[:top]
    worst = sorted(qualified, key=lambda kv: kv[1]['expectancy_lb'])[:top]

    lines.append("\nConditions carrying the edge (ranked by lower bound):")
    for k, b in best:
        tag = ' [provisional]' if b.get('provisional') else ''
        lines.append(f"  {k:<28} n={b['n']:<5} exp {b['expectancy']:+.3f}R "
                     f"(lb {b['expectancy_lb']:+.3f}R) win {b['win_rate']:.0%}{tag}")

    bleeding = [(k, b) for k, b in worst if b['expectancy_lb'] < 0]
    if bleeding:
        lines.append("\nConditions to stop trading (negative lower bound):")
        for k, b in bleeding:
            lines.append(f"  {k:<28} n={b['n']:<5} exp {b['expectancy']:+.3f}R "
                         f"(lb {b['expectancy_lb']:+.3f}R) total {b['total_r']:+.1f}R")
    return "\n".join(lines)
