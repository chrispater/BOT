"""
Core vocabulary of the Market Intelligence Engine.

Everything downstream — feature engine, store, models, decision layer — agrees on
the definitions here: what a market state is, which horizons we measure outcomes
over, and what a decision looks like when it comes out the far end.

The single most important convention in this file: an observation is recorded
WITHOUT a label. We do not decide at record time whether a moment was a BUY, a
SELL or a HOLD. We record what the market looked like, and later — once the
future has actually happened — we attach what followed. Labels are opinions;
outcomes are facts.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


# ── Outcome horizons ─────────────────────────────────────────────────────────
# Measured in seconds. Every observation eventually carries a forward return, a
# maximum favorable excursion (MFE) and a maximum adverse excursion (MAE) at each
# of these horizons. Short horizons are only resolvable when observations are
# recorded densely (live tick/second sampling); candle-derived backfill resolves
# whichever horizons its bar interval can support.
HORIZONS_SEC: List[int] = [30, 60, 180, 300, 900]


def horizon_label(seconds: int) -> str:
    """30 → '30s', 180 → '3m', 900 → '15m'. Used to name outcome columns."""
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def outcome_columns(horizons: Optional[List[int]] = None) -> List[str]:
    """Canonical outcome column names, e.g. ret_3m / mfe_3m / mae_3m."""
    cols = []
    for h in (horizons or HORIZONS_SEC):
        lab = horizon_label(h)
        cols += [f"ret_{lab}", f"mfe_{lab}", f"mae_{lab}"]
    return cols


# ── Feature groups ───────────────────────────────────────────────────────────
# Every feature name is prefixed with its group. This is not cosmetic: the
# attribution layer reports edge contribution per group, the coverage gate
# reasons about groups that are only available live, and the decision layer can
# explain itself in terms a human recognises ("order flow disagreed").
GROUP_PRICE      = 'ps'   # price structure: returns, vol, VWAP distance, ranges
GROUP_ORDERFLOW  = 'of'   # order book & tape: imbalance, depth, spread, aggression
GROUP_DERIVS     = 'dv'   # positioning: open interest, funding, basis, liquidations
GROUP_CROSS      = 'xm'   # cross-market: BTC/ETH confirmation or contradiction
GROUP_REGIME     = 'rg'   # regime descriptors (continuous, not the label itself)
GROUP_EXEC       = 'ex'   # execution conditions: spread, depth, expected slippage
GROUP_TIME       = 'tm'   # session / time-of-day

ALL_GROUPS = [GROUP_PRICE, GROUP_ORDERFLOW, GROUP_DERIVS, GROUP_CROSS,
              GROUP_REGIME, GROUP_EXEC, GROUP_TIME]

# Groups that only exist when the bot is watching a live venue. Historical OHLCV
# backfill cannot reconstruct them, so they are expected to be missing in the
# early life of the store and must never be treated as "broken data".
LIVE_ONLY_GROUPS = [GROUP_ORDERFLOW, GROUP_DERIVS, GROUP_EXEC]

GROUP_NAMES = {
    GROUP_PRICE:     'price structure',
    GROUP_ORDERFLOW: 'order flow',
    GROUP_DERIVS:    'derivatives positioning',
    GROUP_CROSS:     'cross-market',
    GROUP_REGIME:    'regime',
    GROUP_EXEC:      'execution conditions',
    GROUP_TIME:      'session',
}


def feature_group(name: str) -> str:
    """'of_book_imbalance' → 'of'. Unprefixed names fall into price structure."""
    head = name.split('_', 1)[0]
    return head if head in ALL_GROUPS else GROUP_PRICE


# ── Regime taxonomy ──────────────────────────────────────────────────────────
# Deliberately not 'bull/bear/sideways'. What matters for a 3-to-10-minute trade
# is not the macro direction but the character of the tape: does it continue, does
# it revert, is volatility expanding, is liquidity there at all.
REGIME_TREND_UP      = 'trend_up'
REGIME_TREND_DOWN    = 'trend_down'
REGIME_MEAN_REVERT   = 'mean_revert'
REGIME_COMPRESSION   = 'compression'      # low vol, coiled, pre-expansion
REGIME_EXPANSION     = 'expansion'        # vol breaking out of its own range
REGIME_PANIC         = 'panic'            # liquidation cascade / disorderly
REGIME_THIN          = 'thin'             # illiquid chop — untradeable after costs

ALL_REGIMES = [REGIME_TREND_UP, REGIME_TREND_DOWN, REGIME_MEAN_REVERT,
               REGIME_COMPRESSION, REGIME_EXPANSION, REGIME_PANIC, REGIME_THIN]

# Regimes in which the engine refuses to size up regardless of model output.
# Being right about direction does not help if the exit is unfillable.
HOSTILE_REGIMES = [REGIME_PANIC, REGIME_THIN]


# ── Decision vocabulary ──────────────────────────────────────────────────────
ACTION_LONG    = 'LONG'
ACTION_SHORT   = 'SHORT'
ACTION_NOTHING = 'DO_NOTHING'


@dataclass
class MarketState:
    """
    One point-in-time observation of a market. `features` is the full state
    vector; nothing in here describes what we intend to do about it.

    `as_of` is the timestamp of the most recent CLOSED information used to build
    the state. Feature builders must never look past it — the entire value of
    this system rests on that being true.
    """
    symbol: str
    as_of: Any                      # datetime (tz-aware UTC by convention)
    price: float                    # reference mid/close at as_of
    features: Dict[str, float] = field(default_factory=dict)
    regime: str = REGIME_COMPRESSION
    regime_confidence: float = 0.0
    coverage: Dict[str, float] = field(default_factory=dict)  # group → frac present

    def vector(self, columns: List[str]) -> List[float]:
        """Extract features in a fixed column order, NaN-filling absentees."""
        return [self.features.get(c, float('nan')) for c in columns]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['as_of'] = self.as_of.isoformat() if hasattr(self.as_of, 'isoformat') else self.as_of
        return d


@dataclass
class CostEstimate:
    """What it costs to get in and back out, in fractions of notional."""
    spread_cost: float = 0.0        # half-spread paid on entry (taker)
    slippage: float = 0.0           # depth-walk beyond touch, entry + exit
    fees: float = 0.0               # exchange fees, both sides, at our tier
    total: float = 0.0              # everything above, round trip
    use_maker_entry: bool = False   # is a passive entry economically better?
    fillable: bool = True           # is there enough depth for our size at all?
    note: str = ''


@dataclass
class EdgeForecast:
    """
    The edge model's read on a single direction at a single horizon. All returns
    are fractions of price, gross of costs — the decision layer subtracts costs
    so that cost assumptions stay visible and auditable in one place.
    """
    horizon_sec: int
    direction: int                  # +1 long, -1 short
    expected_return: float = 0.0
    prob_positive: float = 0.5
    expected_mfe: float = 0.0
    expected_mae: float = 0.0
    model_confidence: float = 0.0   # OOS-validated reliability of this forecast
    sample_size: int = 0            # comparable historical observations behind it


@dataclass
class TradeDecision:
    """
    The engine's output. Note what is NOT here: a bare LONG/SHORT with a
    confidence percentage. A decision that cannot state its expected adverse
    excursion, its costs and how many comparable observations it rests on is not
    a decision, it is a guess with a number attached.
    """
    symbol: str
    as_of: Any = None                   # the MarketState.as_of this decision was made from —
                                         # ties a decision back to its exact observation for
                                         # attribution, so this must never be wall-clock "now"
    action: str = ACTION_NOTHING
    horizon_sec: int = 300
    expected_return: float = 0.0        # net of costs
    expected_adverse: float = 0.0       # expected MAE, the risk we're accepting
    prob_positive: float = 0.5
    costs: float = 0.0
    regime: str = REGIME_COMPRESSION
    regime_confidence: float = 0.0
    historical_sample: int = 0
    expectancy_r: float = 0.0           # expected value per unit of risk
    quality: int = 0                    # 0-100 composite
    stop_distance: float = 0.0          # fraction of price to the stop = 1R
    target_distance: float = 0.0        # fraction of price to the target
    size_fraction: float = 0.0          # fraction of equity to risk (not notional)
    reasons: List[str] = field(default_factory=list)   # why we are/aren't trading
    blockers: List[str] = field(default_factory=list)  # gates that vetoed a trade
    forecast: Optional[EdgeForecast] = None
    cost_detail: Optional[CostEstimate] = None

    @property
    def is_trade(self) -> bool:
        return self.action in (ACTION_LONG, ACTION_SHORT)

    def explain(self) -> str:
        """Human-readable summary — the format the operator actually reads."""
        lines = [f"{self.symbol}  {self.action}   quality {self.quality}/100"]
        if self.is_trade or self.forecast is not None:
            lines += [
                f"  expected return    {self.expected_return*100:+.3f}%  (net of costs)",
                f"  expected adverse   {self.expected_adverse*100:+.3f}%",
                f"  probability favorable  {self.prob_positive*100:.1f}%",
                f"  estimated costs    {self.costs*100:.3f}%",
                f"  expectancy         {self.expectancy_r:+.3f}R",
                f"  regime             {self.regime} ({self.regime_confidence*100:.0f}% conf)",
                f"  historical sample  {self.historical_sample} comparable observations",
            ]
        if self.is_trade:
            lines += [
                f"  stop / target      {self.stop_distance*100:.3f}% / {self.target_distance*100:.3f}%",
                f"  risk sizing        {self.size_fraction*100:.2f}% of equity at 1R",
            ]
        for b in self.blockers:
            lines.append(f"  BLOCKED: {b}")
        for r in self.reasons:
            lines.append(f"  · {r}")
        return "\n".join(lines)
