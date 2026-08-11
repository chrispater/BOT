"""
Feature engine — turns raw market data into a market STATE.

Two design commitments worth stating up front, because they are what separate
this from the indicator pile it replaces:

1. No lookahead, ever. Every column is built from rolling/shift operations over
   past data only, and the last row of an exchange OHLCV frame is treated as a
   still-forming candle and dropped by default. A feature that peeks is worse
   than no feature: it produces a backtest that cannot be traded.

2. Missing is a first-class value, not an error. Order-book, tape and
   derivatives features simply do not exist for historical candles. Rather than
   fabricate them or refuse to train, we emit NaN and let the coverage gate
   decide which features have enough real support to be used. The system starts
   life reasoning about price structure alone and grows into microstructure as it
   accumulates live observations.

Nothing here depends on TA-Lib. That is deliberate — the research path has to run
anywhere (a laptop, CI, a cheap VPS) without a C toolchain.
"""

import math
import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from .state import (
    MarketState, GROUP_PRICE, GROUP_ORDERFLOW, GROUP_DERIVS, GROUP_CROSS,
    GROUP_REGIME, GROUP_EXEC, GROUP_TIME, ALL_GROUPS, feature_group,
)

logger = logging.getLogger(__name__)

EPS = 1e-12

# A reference notional used when computing "what would it cost to trade this"
# features. The state should describe tradeability at a realistic clip size, not
# at an infinitesimal one where every book looks deep.
DEFAULT_PROBE_NOTIONAL = 5_000.0


# ── small numeric helpers ────────────────────────────────────────────────────

def _safe_div(a, b):
    return a / (b + EPS)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    return pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR without TA-Lib."""
    return _true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _rolling_rank(s: pd.Series, window: int) -> pd.Series:
    """
    Percentile rank of the current value within its own trailing window, in
    [0, 1]. This is how the engine expresses ideas like "volatility expanding out
    of its bottom quartile" without hard-coding a volatility number that is only
    meaningful for one asset in one month.
    """
    return s.rolling(window, min_periods=max(10, window // 4)).apply(
        lambda w: (w[-1] > w[:-1]).mean() if len(w) > 1 else 0.5, raw=True
    )


def _efficiency(close: pd.Series, window: int) -> pd.Series:
    """
    Fractal efficiency: net displacement divided by total path length. Near 1 the
    market is travelling in a straight line (trend continuation is cheap); near 0
    it is grinding back and forth (every entry pays the spread twice for nothing).
    """
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return _safe_div(net, path)


def _wilder_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Directional-movement trend strength, Wilder's ADX, pure pandas."""
    up = df['high'].diff()
    dn = -df['low'].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * _safe_div(
        pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean(), atr)
    minus_di = 100 * _safe_div(
        pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean(), atr)
    dx = 100 * _safe_div((plus_di - minus_di).abs(), (plus_di + minus_di))
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ── price structure, regime descriptors, session ─────────────────────────────

def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized price-structure / regime / session features for every row of an
    OHLCV frame. Index must be a DatetimeIndex; columns open/high/low/close/volume.

    Returns a frame with the same index containing only prefixed feature columns,
    so callers can concatenate feature blocks without worrying about collisions
    with raw price columns.
    """
    out = pd.DataFrame(index=df.index)
    close, high, low, vol = df['close'], df['high'], df['low'], df['volume']
    open_ = df['open']

    # ── Multi-horizon returns: where price has been heading, at several scales.
    ret1 = close.pct_change()
    out[f'{GROUP_PRICE}_ret_1'] = ret1
    for n in (3, 5, 10, 20):
        out[f'{GROUP_PRICE}_ret_{n}'] = close.pct_change(n)
    out[f'{GROUP_PRICE}_ret_accel'] = ret1 - ret1.shift(1)

    # ── Realized volatility and its own trend. Absolute vol is not comparable
    # across assets; the ratio and the percentile rank are.
    vol20 = ret1.rolling(20, min_periods=10).std()
    vol50 = ret1.rolling(50, min_periods=20).std()
    out[f'{GROUP_PRICE}_rvol_20'] = vol20
    out[f'{GROUP_PRICE}_rvol_ratio'] = _safe_div(vol20, vol50)
    atr = _atr(df, 14)
    atr_pct = _safe_div(atr, close)
    out[f'{GROUP_PRICE}_atr_pct'] = atr_pct

    # ── Distance from fair value. A 20-bar rolling VWAP, not a cumulative one:
    # cumulative VWAP drifts into irrelevance over a long frame.
    tp = (high + low + close) / 3.0
    vwap = _safe_div((tp * vol).rolling(20, min_periods=5).sum(),
                     vol.rolling(20, min_periods=5).sum())
    out[f'{GROUP_PRICE}_vwap_dist'] = _safe_div(close - vwap, vwap)
    out[f'{GROUP_PRICE}_vwap_slope'] = vwap.pct_change(5)
    # Normalising the same distance by ATR answers a different question: not "how
    # far in percent" but "how far relative to how far this market usually moves".
    out[f'{GROUP_PRICE}_vwap_dist_atr'] = _safe_div(close - vwap, atr)

    # ── Position within recent ranges, and proximity to breakout levels.
    for n in (5, 20):
        hi_n = high.rolling(n, min_periods=2).max()
        lo_n = low.rolling(n, min_periods=2).min()
        out[f'{GROUP_PRICE}_range_pos_{n}'] = _safe_div(close - lo_n, hi_n - lo_n)
    hi20_prev = high.rolling(20, min_periods=5).max().shift(1)
    lo20_prev = low.rolling(20, min_periods=5).min().shift(1)
    out[f'{GROUP_PRICE}_dist_high_20'] = _safe_div(close - hi20_prev, atr)
    out[f'{GROUP_PRICE}_dist_low_20'] = _safe_div(close - lo20_prev, atr)
    out[f'{GROUP_PRICE}_new_high_20'] = (close >= hi20_prev).astype(float)
    out[f'{GROUP_PRICE}_new_low_20'] = (close <= lo20_prev).astype(float)

    # ── Candle anatomy: conviction and rejection. A big range that closes mid is
    # a fight; a big range that closes on its high is one side winning.
    rng = (high - low)
    out[f'{GROUP_PRICE}_body_frac'] = _safe_div(close - open_, rng)
    out[f'{GROUP_PRICE}_upper_wick'] = _safe_div(high - np.maximum(close, open_), rng)
    out[f'{GROUP_PRICE}_lower_wick'] = _safe_div(np.minimum(close, open_) - low, rng)
    out[f'{GROUP_PRICE}_gap'] = _safe_div(open_ - close.shift(1), close.shift(1))

    # ── Persistence and participation.
    direction = np.sign(ret1.fillna(0.0))
    run_group = (direction != direction.shift()).cumsum()
    run_len = direction.groupby(run_group).cumcount() + 1
    out[f'{GROUP_PRICE}_streak'] = (run_len * direction).clip(-10, 10)
    out[f'{GROUP_PRICE}_green_frac_10'] = (ret1 > 0).rolling(10, min_periods=5).mean()

    vol_sma = vol.rolling(20, min_periods=5).mean()
    out[f'{GROUP_PRICE}_vol_ratio'] = _safe_div(vol, vol_sma)
    up_vol = vol.where(close > open_, 0.0)
    dn_vol = vol.where(close < open_, 0.0)
    out[f'{GROUP_PRICE}_dir_volume'] = _safe_div(
        up_vol.rolling(10, min_periods=3).sum() - dn_vol.rolling(10, min_periods=3).sum(),
        vol.rolling(10, min_periods=3).sum())

    # ── Mean-reversion pressure: how stretched is price from its own recent mean,
    # measured in standard deviations of itself.
    m20 = close.rolling(20, min_periods=10).mean()
    s20 = close.rolling(20, min_periods=10).std()
    out[f'{GROUP_PRICE}_zscore_20'] = _safe_div(close - m20, s20)
    out[f'{GROUP_PRICE}_ret_skew_20'] = ret1.rolling(20, min_periods=15).skew()

    # ── Regime descriptors. These are the continuous inputs the regime model
    # consumes; the discrete regime label is derived from them elsewhere.
    out[f'{GROUP_REGIME}_adx'] = _wilder_adx(df, 14)
    out[f'{GROUP_REGIME}_efficiency_10'] = _efficiency(close, 10)
    out[f'{GROUP_REGIME}_efficiency_50'] = _efficiency(close, 50)
    # Lag-1 return autocorrelation: the cleanest single discriminator between a
    # market that continues (positive) and one that reverts (negative).
    out[f'{GROUP_REGIME}_autocorr_1'] = ret1.rolling(50, min_periods=30).apply(
        lambda w: pd.Series(w).autocorr(lag=1) if np.nanstd(w) > 0 else 0.0, raw=True)
    out[f'{GROUP_REGIME}_vol_rank_200'] = _rolling_rank(vol20, 200)
    out[f'{GROUP_REGIME}_vol_of_vol'] = _safe_div(
        vol20.rolling(50, min_periods=20).std(), vol20.rolling(50, min_periods=20).mean())
    # Bollinger-style compression, expressed relative to its own history so
    # "squeezed" means squeezed for this market, not below a magic number.
    bb_width = _safe_div(4 * close.rolling(20, min_periods=10).std(), m20)
    out[f'{GROUP_REGIME}_squeeze'] = _safe_div(bb_width, bb_width.rolling(50, min_periods=20).mean())
    ema_f = close.ewm(span=12, adjust=False).mean()
    ema_s = close.ewm(span=48, adjust=False).mean()
    out[f'{GROUP_REGIME}_trend_strength'] = _safe_div(ema_f - ema_s, atr)
    out[f'{GROUP_REGIME}_ema_slope'] = ema_s.pct_change(10)
    # Turnover in quote currency — the raw material of a thin-liquidity check.
    turnover = (close * vol).rolling(20, min_periods=5).median()
    out[f'{GROUP_REGIME}_turnover_rank'] = _rolling_rank(turnover, 200)

    # ── Session. Crypto trades continuously but its liquidity does not: the
    # character of the tape differs materially by hour.
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        hour = idx.hour + idx.minute / 60.0
        out[f'{GROUP_TIME}_hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        out[f'{GROUP_TIME}_hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        out[f'{GROUP_TIME}_dow'] = idx.dayofweek.astype(float)
        out[f'{GROUP_TIME}_is_weekend'] = (idx.dayofweek >= 5).astype(float)
        out[f'{GROUP_TIME}_asia'] = ((hour >= 0) & (hour < 8)).astype(float)
        out[f'{GROUP_TIME}_europe'] = ((hour >= 7) & (hour < 15)).astype(float)
        out[f'{GROUP_TIME}_us'] = ((hour >= 13) & (hour < 21)).astype(float)

    return out.replace([np.inf, -np.inf], np.nan)


# ── cross-market confirmation ────────────────────────────────────────────────

def build_cross_features(df: pd.DataFrame,
                         references: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Does the rest of the market agree with this move?

    `references` maps a short tag ('btc', 'eth') to that asset's OHLCV frame.
    Reference frames are reindexed onto our own index with forward-fill, which is
    safe in one direction only: forward-filling uses the last KNOWN reference
    value, never a future one.
    """
    out = pd.DataFrame(index=df.index)
    own_ret1 = df['close'].pct_change()
    own_ret5 = df['close'].pct_change(5)

    for tag, ref in (references or {}).items():
        if ref is None or len(ref) < 20 or 'close' not in ref:
            continue
        tag = tag.lower()
        ref_close = ref['close'].reindex(df.index.union(ref.index)).ffill().reindex(df.index)
        r_ret1 = ref_close.pct_change()

        for n in (1, 5, 15):
            out[f'{GROUP_CROSS}_{tag}_ret_{n}'] = ref_close.pct_change(n)

        # Rolling correlation and beta: is this asset currently tethered to the
        # reference, and by how much? Both matter — a 0.9 correlation makes BTC's
        # move informative, a 0.1 correlation makes it noise.
        cov = own_ret1.rolling(50, min_periods=30).cov(r_ret1)
        ref_var = r_ret1.rolling(50, min_periods=30).var()
        out[f'{GROUP_CROSS}_{tag}_corr_50'] = own_ret1.rolling(50, min_periods=30).corr(r_ret1)
        beta = _safe_div(cov, ref_var)
        out[f'{GROUP_CROSS}_{tag}_beta_50'] = beta

        # Agreement, and the residual move that the reference does NOT explain.
        # A rally the reference explains entirely is a different animal from one
        # the asset is making on its own.
        ref_ret5 = ref_close.pct_change(5)
        out[f'{GROUP_CROSS}_{tag}_agree'] = np.sign(own_ret5) * np.sign(ref_ret5)
        out[f'{GROUP_CROSS}_{tag}_resid_5'] = own_ret5 - beta * ref_ret5
        out[f'{GROUP_CROSS}_{tag}_vol_ratio'] = _safe_div(
            own_ret1.rolling(20, min_periods=10).std(),
            r_ret1.rolling(20, min_periods=10).std())

    return out.replace([np.inf, -np.inf], np.nan)


# ── order book & tape (live only) ────────────────────────────────────────────

def _side_depth(levels: List[List[float]], mid: float, bps: float) -> float:
    """Quote-currency depth resting within `bps` basis points of the mid."""
    if not levels or mid <= 0:
        return 0.0
    limit = mid * (bps / 10_000.0)
    total = 0.0
    for price, qty in levels:
        if abs(float(price) - mid) > limit:
            break
        total += float(price) * float(qty)
    return total


def walk_book(levels: List[List[float]], notional: float) -> Optional[float]:
    """
    Average fill price for a market order of `notional` quote currency walking
    this side of the book. Returns None when the visible book cannot absorb the
    order — which is itself a decision-relevant answer, not a failure.
    """
    if not levels or notional <= 0:
        return None
    remaining, cost, filled = notional, 0.0, 0.0
    for price, qty in levels:
        price, qty = float(price), float(qty)
        level_notional = price * qty
        take = min(remaining, level_notional)
        if take <= 0:
            continue
        cost += take
        filled += take / price
        remaining -= take
        if remaining <= EPS:
            break
    if remaining > EPS or filled <= 0:
        return None
    return cost / filled


def orderbook_features(book: Dict[str, Any],
                       prev: Optional[Dict[str, Any]] = None,
                       probe_notional: float = DEFAULT_PROBE_NOTIONAL,
                       atr_pct: Optional[float] = None) -> Dict[str, float]:
    """
    Microstructure state from one order-book snapshot.

    `book` is ccxt-shaped: {'bids': [[price, qty], ...], 'asks': [...]} sorted
    best-first. `prev` is an earlier snapshot for the same symbol, which is what
    makes the interesting features possible: a static book tells you where
    liquidity is, but comparing two tells you that liquidity is LEAVING — the
    "ask depth just dropped 18%" observation that precedes a lot of real moves.

    The crucial framing: order-book imbalance is not another oscillator to
    threshold. It is one dimension of state whose predictive value the edge model
    has to establish conditionally, alongside everything else.
    """
    f: Dict[str, float] = {}
    bids = book.get('bids') or []
    asks = book.get('asks') or []
    if not bids or not asks:
        return f

    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    bid_qty, ask_qty = float(bids[0][1]), float(asks[0][1])
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return f

    spread = best_ask - best_bid
    f[f'{GROUP_ORDERFLOW}_spread_bps'] = (spread / mid) * 10_000.0

    # Microprice: the size-weighted touch. Its deviation from the mid is a
    # short-horizon lean that a plain mid hides entirely.
    micro = (best_bid * ask_qty + best_ask * bid_qty) / (bid_qty + ask_qty + EPS)
    f[f'{GROUP_ORDERFLOW}_microprice_dev'] = (micro - mid) / mid
    f[f'{GROUP_ORDERFLOW}_imbalance_l1'] = (bid_qty - ask_qty) / (bid_qty + ask_qty + EPS)

    # Depth imbalance at widening distances. Imbalance at the touch is noisy and
    # easily spoofed; imbalance 25bps deep is harder to fake and says more about
    # where real inventory sits.
    depths = {}
    for bps in (5, 25, 100):
        b = _side_depth(bids, mid, bps)
        a = _side_depth(asks, mid, bps)
        depths[bps] = (b, a)
        f[f'{GROUP_ORDERFLOW}_imbalance_{bps}bps'] = (b - a) / (b + a + EPS)
    bid_25, ask_25 = depths[25]
    f[f'{GROUP_ORDERFLOW}_depth_bid_usd'] = bid_25
    f[f'{GROUP_ORDERFLOW}_depth_ask_usd'] = ask_25
    f[f'{GROUP_ORDERFLOW}_depth_total_usd'] = bid_25 + ask_25

    # Book shape: how quickly does size accumulate as you move away from the
    # touch? A steep book resists; a flat one lets price travel.
    b5, a5 = depths[5]
    f[f'{GROUP_ORDERFLOW}_bid_slope'] = _safe_div(bid_25 - b5, b5) if b5 > 0 else np.nan
    f[f'{GROUP_ORDERFLOW}_ask_slope'] = _safe_div(ask_25 - a5, a5) if a5 > 0 else np.nan

    # Change versus the previous snapshot — liquidity appearing or vanishing.
    if prev:
        p_bids, p_asks = prev.get('bids') or [], prev.get('asks') or []
        if p_bids and p_asks:
            p_mid = (float(p_bids[0][0]) + float(p_asks[0][0])) / 2.0
            pb, pa = _side_depth(p_bids, p_mid, 25), _side_depth(p_asks, p_mid, 25)
            if pb > 0:
                f[f'{GROUP_ORDERFLOW}_bid_depth_change'] = (bid_25 - pb) / pb
            if pa > 0:
                f[f'{GROUP_ORDERFLOW}_ask_depth_change'] = (ask_25 - pa) / pa
            prev_imb = (pb - pa) / (pb + pa + EPS)
            cur_imb = f[f'{GROUP_ORDERFLOW}_imbalance_25bps']
            f[f'{GROUP_ORDERFLOW}_imbalance_velocity'] = cur_imb - prev_imb
            if p_mid > 0:
                f[f'{GROUP_ORDERFLOW}_mid_drift'] = (mid - p_mid) / p_mid

    # Cost of crossing, both sides, at a realistic clip. This is where a pretty
    # signal meets the actual bill.
    buy_px = walk_book(asks, probe_notional)
    sell_px = walk_book(bids, probe_notional)
    if buy_px:
        f[f'{GROUP_EXEC}_slip_buy'] = (buy_px - mid) / mid
    if sell_px:
        f[f'{GROUP_EXEC}_slip_sell'] = (mid - sell_px) / mid
    f[f'{GROUP_EXEC}_fillable'] = 1.0 if (buy_px and sell_px) else 0.0
    f[f'{GROUP_EXEC}_spread_pct'] = spread / mid

    # The single most useful tradeability number in the whole vector: round-trip
    # frictional cost as a fraction of the move this market typically makes. Above
    # roughly 0.5 there is no short-horizon trade here at any win rate.
    if atr_pct and atr_pct > 0:
        rt = (spread / mid) + abs(f.get(f'{GROUP_EXEC}_slip_buy', 0.0)) \
                            + abs(f.get(f'{GROUP_EXEC}_slip_sell', 0.0))
        f[f'{GROUP_EXEC}_cost_to_atr'] = rt / atr_pct

    return f


def tape_features(trades: List[Dict[str, Any]],
                  window_sec: float = 60.0,
                  baseline_volume: Optional[float] = None,
                  price_move: Optional[float] = None) -> Dict[str, float]:
    """
    Aggression from the trade tape: who is crossing the spread, how hard, and is
    price actually moving as a result.

    `trades` is ccxt-shaped ({'side': 'buy'|'sell', 'amount', 'price',
    'timestamp'}), most recent last. On most venues 'side' is the taker's side,
    which is exactly what we want — resting orders are intent, crossing orders are
    commitment.
    """
    f: Dict[str, float] = {}
    if not trades:
        return f

    latest_ts = max(float(t.get('timestamp') or 0) for t in trades)
    cutoff = latest_ts - window_sec * 1000.0
    recent = [t for t in trades if float(t.get('timestamp') or 0) >= cutoff]
    if not recent:
        return f

    buy_notional = sum(float(t.get('amount', 0)) * float(t.get('price', 0))
                       for t in recent if str(t.get('side', '')).lower() == 'buy')
    sell_notional = sum(float(t.get('amount', 0)) * float(t.get('price', 0))
                        for t in recent if str(t.get('side', '')).lower() == 'sell')
    total = buy_notional + sell_notional
    if total <= 0:
        return f

    f[f'{GROUP_ORDERFLOW}_aggr_imbalance'] = (buy_notional - sell_notional) / total
    f[f'{GROUP_ORDERFLOW}_aggr_notional'] = total
    f[f'{GROUP_ORDERFLOW}_trade_count'] = float(len(recent))
    f[f'{GROUP_ORDERFLOW}_avg_trade_size'] = total / len(recent)

    if baseline_volume and baseline_volume > 0:
        f[f'{GROUP_ORDERFLOW}_aggr_volume_norm'] = total / baseline_volume

        # Absorption: heavy aggressive volume that fails to move price means the
        # other side is soaking it up — often the opposite of what the volume
        # spike naively suggests.
        if price_move is not None:
            f[f'{GROUP_ORDERFLOW}_absorption'] = (total / baseline_volume) / (abs(price_move) * 100 + 0.01)

    return f


# ── derivatives positioning (live only) ──────────────────────────────────────

def derivative_features(funding_rate: Optional[float] = None,
                        open_interest: Optional[float] = None,
                        oi_history: Optional[List[float]] = None,
                        mark_price: Optional[float] = None,
                        index_price: Optional[float] = None,
                        price_change: Optional[float] = None,
                        liquidation_notional: Optional[float] = None,
                        baseline_volume: Optional[float] = None) -> Dict[str, float]:
    """
    Positioning state. Price tells you what happened; open interest and funding
    tell you who is on the hook for it, which is what determines whether a move
    continues or unwinds violently.

    `oi_history` is a series of recent open-interest readings, oldest first,
    including the current one.
    """
    f: Dict[str, float] = {}

    if funding_rate is not None:
        f[f'{GROUP_DERIVS}_funding'] = float(funding_rate)
        # Funding scaled to an annualised figure is easier to reason about across
        # venues with different settlement intervals (assumes 8h settlement).
        f[f'{GROUP_DERIVS}_funding_annual'] = float(funding_rate) * 3 * 365

    if oi_history and len(oi_history) >= 2:
        hist = [float(x) for x in oi_history if x is not None and float(x) > 0]
        if len(hist) >= 2:
            cur = hist[-1]
            f[f'{GROUP_DERIVS}_oi_change_1'] = (cur - hist[-2]) / (hist[-2] + EPS)
            if len(hist) >= 5:
                f[f'{GROUP_DERIVS}_oi_change_5'] = (cur - hist[-5]) / (hist[-5] + EPS)
            if len(hist) >= 15:
                f[f'{GROUP_DERIVS}_oi_change_15'] = (cur - hist[-15]) / (hist[-15] + EPS)
            mean_oi = float(np.mean(hist))
            std_oi = float(np.std(hist))
            if std_oi > 0:
                f[f'{GROUP_DERIVS}_oi_z'] = (cur - mean_oi) / std_oi

            # The OI/price interaction is the informative part, not either alone:
            #   price up + OI up   → new longs, fuel for continuation
            #   price up + OI down → shorts covering, a rally with no new backing
            #   price down + OI up → new shorts pressing
            #   price down + OI down→ longs capitulating, exhaustion
            if price_change is not None:
                oi_chg = f.get(f'{GROUP_DERIVS}_oi_change_5', f.get(f'{GROUP_DERIVS}_oi_change_1', 0.0))
                f[f'{GROUP_DERIVS}_oi_price_agree'] = float(np.sign(oi_chg) * np.sign(price_change))
                f[f'{GROUP_DERIVS}_oi_price_interaction'] = float(oi_chg * price_change * 1_000)

    if open_interest is not None:
        f[f'{GROUP_DERIVS}_oi'] = float(open_interest)

    # Perp/spot basis: how far the derivative has run from the underlying.
    if mark_price and index_price and index_price > 0:
        f[f'{GROUP_DERIVS}_basis'] = (float(mark_price) - float(index_price)) / float(index_price)

    if liquidation_notional is not None and baseline_volume:
        f[f'{GROUP_DERIVS}_liq_intensity'] = float(liquidation_notional) / (baseline_volume + EPS)

    return f


# ── the engine ───────────────────────────────────────────────────────────────

class FeatureEngine:
    """
    Assembles market states. Two entry points, sharing one definition of every
    feature so that what the model trains on and what it sees live cannot drift
    apart — the failure mode that silently invalidates most trading research.

      build_frame()  → historical feature matrix (price/regime/session/cross)
      build_state()  → one live MarketState (the above plus microstructure)
    """

    def __init__(self, probe_notional: float = DEFAULT_PROBE_NOTIONAL,
                 closed_only: bool = True):
        self.probe_notional = probe_notional
        # Drop the final row of incoming OHLCV: on a live venue it is the candle
        # currently forming, whose high/low/close are not yet real. Training on it
        # is a subtle lookahead; trading on it is worse.
        self.closed_only = closed_only
        self._prev_books: Dict[str, Dict[str, Any]] = {}

    # -- historical -------------------------------------------------------

    def build_frame(self, df: pd.DataFrame,
                    references: Optional[Dict[str, pd.DataFrame]] = None,
                    keep_prices: bool = True) -> pd.DataFrame:
        """
        Feature matrix for a full OHLCV history. Live-only groups are absent
        entirely rather than zero-filled: absent means "unknown", and a model that
        can represent unknown will not mistake it for "balanced book".
        """
        if df is None or len(df) < 60:
            raise ValueError(f"need at least 60 bars to build features, got {0 if df is None else len(df)}")
        df = df.sort_index()
        if self.closed_only and len(df) > 1:
            df = df.iloc[:-1]

        blocks = [build_price_features(df)]
        if references:
            blocks.append(build_cross_features(df, references))
        feats = pd.concat(blocks, axis=1)

        if keep_prices:
            # Outcome resolution needs the price path; carry it alongside rather
            # than making callers re-align two frames by hand.
            for col in ('open', 'high', 'low', 'close', 'volume'):
                if col in df.columns:
                    feats[col] = df[col]
        return feats

    # -- live -------------------------------------------------------------

    def build_state(self, symbol: str, df: pd.DataFrame,
                    references: Optional[Dict[str, pd.DataFrame]] = None,
                    book: Optional[Dict[str, Any]] = None,
                    trades: Optional[List[Dict[str, Any]]] = None,
                    derivs: Optional[Dict[str, Any]] = None) -> MarketState:
        """
        Build the current state for one symbol from everything available. Any
        argument may be None; the resulting state simply carries lower coverage,
        which the decision layer takes into account instead of pretending.
        """
        feats_df = self.build_frame(df, references, keep_prices=False)
        row = feats_df.iloc[-1]
        as_of = feats_df.index[-1]
        features: Dict[str, float] = {
            k: (float(v) if pd.notna(v) else float('nan')) for k, v in row.items()
        }

        closed = df.iloc[:-1] if (self.closed_only and len(df) > 1) else df
        price = float(closed['close'].iloc[-1])
        atr_pct = features.get(f'{GROUP_PRICE}_atr_pct')
        baseline_volume = float((closed['close'] * closed['volume']).rolling(20, min_periods=5)
                                .mean().iloc[-1]) if len(closed) >= 5 else None

        if book:
            features.update(orderbook_features(
                book, prev=self._prev_books.get(symbol),
                probe_notional=self.probe_notional,
                atr_pct=atr_pct if (atr_pct and not math.isnan(atr_pct)) else None))
            self._prev_books[symbol] = {'bids': (book.get('bids') or [])[:50],
                                        'asks': (book.get('asks') or [])[:50]}

        if trades:
            features.update(tape_features(
                trades, baseline_volume=baseline_volume,
                price_move=features.get(f'{GROUP_PRICE}_ret_1')))

        if derivs:
            features.update(derivative_features(
                funding_rate=derivs.get('funding_rate'),
                open_interest=derivs.get('open_interest'),
                oi_history=derivs.get('oi_history'),
                mark_price=derivs.get('mark_price'),
                index_price=derivs.get('index_price'),
                price_change=features.get(f'{GROUP_PRICE}_ret_5'),
                liquidation_notional=derivs.get('liquidation_notional'),
                baseline_volume=baseline_volume))

        return MarketState(symbol=symbol, as_of=as_of, price=price,
                           features=features, coverage=group_coverage(features))


def group_coverage(features: Dict[str, float]) -> Dict[str, float]:
    """
    Fraction of each feature group that is actually present (non-NaN).

    Coverage is how the engine knows what it is flying on. A decision made with
    zero order-flow and zero derivatives coverage is not the same decision as one
    made with full coverage, even when the numbers coming out happen to match.
    """
    counts = {g: [0, 0] for g in ALL_GROUPS}
    for name, val in features.items():
        g = feature_group(name)
        counts[g][1] += 1
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            counts[g][0] += 1
    return {g: (c[0] / c[1] if c[1] else 0.0) for g, c in counts.items()}
