"""
Cost model — what it actually costs to get in and back out.

This exists as its own module, separate from the edge model, on purpose: edge
forecasts should be gross of costs so they're comparable across venues and fee
tiers, and costs should be estimated from what we can actually observe about
current market conditions. The decision layer then combines them in exactly one
place (engine.py), so "did we account for fees here" is never a per-call
judgment call scattered across the codebase — one of the sharper mistakes the
predecessor system made when EV formulas hard-coded fee assumptions in three
different functions that could drift out of sync.

Two cost regimes are supported:
  • live, with an order book  → walk the book, know the real cost
  • no book (historical / cold start) → fall back to venue fee-tier assumptions
    plus a volatility-scaled slippage estimate, clearly flagged as an estimate
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from .state import CostEstimate
from .feature_engine import walk_book, DEFAULT_PROBE_NOTIONAL


@dataclass
class FeeTier:
    """Round-trip-relevant fee schedule. Defaults match a typical crypto perp
    venue's mid tier; override from account-specific data when known."""
    maker_fee: float = 0.0002    # 2 bps
    taker_fee: float = 0.0005    # 5 bps


DEFAULT_FEE_TIER = FeeTier()


def estimate_costs_from_book(book: Optional[Dict[str, Any]], notional: float,
                             fee_tier: FeeTier = DEFAULT_FEE_TIER) -> CostEstimate:
    """
    Real cost estimate from a live order book: walk both sides at the size we'd
    actually trade, compare a maker-style entry (post at/near the touch, pay
    nothing extra but risk not filling) against a taker-style entry (cross the
    spread immediately, guaranteed fill).
    """
    if not book or not book.get('bids') or not book.get('asks'):
        return estimate_costs_fallback(notional=notional, fee_tier=fee_tier)

    bids, asks = book['bids'], book['asks']
    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return estimate_costs_fallback(notional=notional, fee_tier=fee_tier)

    buy_px = walk_book(asks, notional)
    sell_px = walk_book(bids, notional)
    fillable = bool(buy_px and sell_px)

    if not fillable:
        # Book can't absorb this size without walking off the visible depth —
        # report the venue's displayed depth as insufficient rather than
        # silently underestimating slippage with a partial fill average.
        return CostEstimate(
            spread_cost=(best_ask - best_bid) / mid / 2.0,
            slippage=float('inf'), fees=fee_tier.taker_fee * 2, total=float('inf'),
            fillable=False, note='order size exceeds visible book depth')

    taker_slip = ((buy_px - mid) + (mid - sell_px)) / mid   # round-trip, both legs taker
    taker_total = taker_slip + fee_tier.taker_fee * 2

    # Maker-style: assume fill at the touch (best bid to buy, best ask to sell)
    # since a resting order that gets filled pays no slippage by definition —
    # the risk it carries (non-fill, adverse selection) is priced as a note for
    # the decision layer to weigh, not folded into this number.
    maker_total = (best_ask - best_bid) / mid + fee_tier.maker_fee * 2

    use_maker = maker_total < taker_total * 0.8   # only prefer maker if materially cheaper
    chosen_total = maker_total if use_maker else taker_total

    return CostEstimate(
        spread_cost=(best_ask - best_bid) / mid,
        slippage=taker_slip,
        fees=(fee_tier.maker_fee if use_maker else fee_tier.taker_fee) * 2,
        total=chosen_total,
        use_maker_entry=use_maker,
        fillable=True,
        note='maker entry preferred' if use_maker else 'taker entry (spread crossed)',
    )


def estimate_costs_fallback(atr_pct: Optional[float] = None, notional: float = DEFAULT_PROBE_NOTIONAL,
                            fee_tier: FeeTier = DEFAULT_FEE_TIER) -> CostEstimate:
    """
    No live book available (historical backtest, or a symbol we haven't
    subscribed order-book data for). Estimate slippage as a small, volatility-
    scaled fraction rather than assuming a free fill — a backtest that assumes
    zero slippage is the single most common way profitable-looking research dies
    on contact with real fills.
    """
    # Heuristic: round-trip slippage of roughly 3% of one bar's typical range on
    # a liquid venue, floored at 2bps so illiquid/low-vol assets aren't credited
    # implausibly cheap execution.
    slip = max(0.0002, (atr_pct or 0.003) * 0.03)
    total = slip + fee_tier.taker_fee * 2
    return CostEstimate(
        spread_cost=fee_tier.taker_fee, slippage=slip, fees=fee_tier.taker_fee * 2,
        total=total, use_maker_entry=False, fillable=True,
        note='no live book — cost is a volatility-scaled estimate, not measured',
    )
