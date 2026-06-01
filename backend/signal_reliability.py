"""
Signal Reliability Scorer
─────────────────────────
Measures, per discrete technical-signal type, how profitable it has actually
been on a given coin — using the bot's real leverage, stop-loss, take-profit
and taker fees. A signal counts as "reliable" only if its historical win rate
clears a bar AND its average per-trade return (after fees, in margin terms) is
positive.

This is a confluence gate: the ML model decides direction, but an entry is only
taken when a reliable technical signal agrees. Double-confirmation cuts the loss
rate (which protects the compound curve) and concentrates capital into the
highest-conviction setups.

Adapted from a standalone reliability scorecard (RSI/Stoch/MACD/Bollinger), but
rebuilt for leveraged crypto futures: it never uses signal-flip exits (which have
no stop loss); every occurrence is scored under the bot's actual SL/TP rules.

Indicator periods and over/oversold levels are TUNABLE (defaults are the
hand-validated GRONKAI levels) and computed internally from OHLCV so the optimizer
can search them per coin.
"""

import numpy as np
import pandas as pd

try:
    import talib
    _TALIB = True
except Exception:
    _TALIB = False

TAKER_FEE = 0.0006  # matches trading_service.TAKER_FEE

# Hand-validated GRONKAI defaults — proven on the standalone crypto scanner.
DEFAULT_PARAMS = {
    'rsi_period': 25,
    'rsi_oversold': 30,
    'rsi_overbought': 72,
    'stoch_k': 15,
    'stoch_d': 15,
    'stoch_oversold': 24,
    'stoch_overbought': 90,
    'bb_period': 180,
    'bb_stddev': 5,
}


class SignalReliability:
    def __init__(self, leverage=10, stop_loss_pct=0.15, take_profit_pct=0.30,
                 horizon=24, min_winrate=0.60, min_samples=8, params=None):
        self.leverage = max(1, leverage)
        self.stop_loss_pct = stop_loss_pct      # % of margin
        self.take_profit_pct = take_profit_pct  # % of margin
        self.horizon = horizon                  # candles to let a signal play out
        self.min_winrate = min_winrate
        self.min_samples = min_samples
        self.params = {**DEFAULT_PARAMS, **(params or {})}

    # ── Indicator computation (own periods, independent of the ML pipeline) ────
    def _indicators(self, df):
        """Compute RSI / Stoch / Bollinger with the configured periods. Returns
        a dict of numpy arrays, or None if talib is unavailable / data too short."""
        if not _TALIB or len(df) < max(self.params['bb_period'], self.params['rsi_period']) + 5:
            return None
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float) if 'high' in df else close
        low = df['low'].values.astype(float) if 'low' in df else close
        p = self.params
        try:
            rsi = talib.RSI(close, timeperiod=int(p['rsi_period']))
            sk, sd = talib.STOCH(high, low, close,
                                 fastk_period=int(p['stoch_k']),
                                 slowk_period=int(p['stoch_d']),
                                 slowd_period=int(p['stoch_d']))
            bb_up, _, bb_lo = talib.BBANDS(close, timeperiod=int(p['bb_period']),
                                           nbdevup=p['bb_stddev'], nbdevdn=p['bb_stddev'])
        except Exception:
            return None
        return {'close': close, 'rsi': rsi, 'stoch_k': sk, 'bb_upper': bb_up, 'bb_lower': bb_lo}

    # ── Discrete signal detection ────────────────────────────────────────────
    def _detect(self, df):
        """
        Return {signal_name: {'dir': +1/-1, 'idx': np.array of candle indices}}.
        Each named signal fires at specific candles in one direction.
        """
        out = {}
        n = len(df)
        if n < 30:
            return out
        ind = self._indicators(df)
        if ind is None:
            return out

        rsi, sk, close = ind['rsi'], ind['stoch_k'], ind['close']
        bb_lo, bb_hi = ind['bb_lower'], ind['bb_upper']
        p = self.params

        def add(name, direction, mask):
            mask = np.asarray(mask)
            idx = np.where(mask)[0]
            idx = idx[idx < n - 1]  # need at least one forward candle
            if len(idx):
                out[name] = {'dir': direction, 'idx': idx}

        with np.errstate(invalid='ignore'):
            add('RSI Oversold (long)', 1, rsi <= p['rsi_oversold'])
            add('RSI Overbought (short)', -1, rsi >= p['rsi_overbought'])
            add('Stoch Oversold (long)', 1, sk <= p['stoch_oversold'])
            add('Stoch Overbought (short)', -1, sk >= p['stoch_overbought'])
            add('BB Lower Break (long)', 1, close < bb_lo)
            add('BB Upper Break (short)', -1, close > bb_hi)

        return out

    # ── Per-occurrence outcome under the bot's real exit rules ────────────────
    def _outcome(self, df, i, direction):
        """
        Margin-% return of taking `direction` at candle i, managed with the
        bot's SL/TP over up to `horizon` candles, after round-trip taker fees.
        Uses intrabar high/low so SL/TP can trigger within a candle.
        """
        close = df['close'].values
        high = df['high'].values if 'high' in df else close
        low = df['low'].values if 'low' in df else close
        entry = close[i]
        if entry <= 0:
            return None
        fee = 2 * TAKER_FEE * self.leverage  # round-trip, in margin terms
        end = min(i + self.horizon, len(df) - 1)

        for j in range(i + 1, end + 1):
            if direction == 1:
                worst = (low[j] - entry) / entry
                best = (high[j] - entry) / entry
            else:
                worst = (entry - high[j]) / entry
                best = (entry - low[j]) / entry
            if worst * self.leverage <= -self.stop_loss_pct:
                return -self.stop_loss_pct - fee
            if best * self.leverage >= self.take_profit_pct:
                return self.take_profit_pct - fee

        final = (close[end] - entry) / entry if direction == 1 else (entry - close[end]) / entry
        return final * self.leverage - fee

    # ── Build the scorecard ───────────────────────────────────────────────────
    def score(self, df):
        """
        Return {signal_name: {dir, count, win_rate, avg_return, reliable}}.
        avg_return is mean margin-% return per occurrence after fees.
        """
        card = {}
        detected = self._detect(df)
        for name, info in detected.items():
            direction = info['dir']
            outcomes = []
            for i in info['idx']:
                o = self._outcome(df, int(i), direction)
                if o is not None:
                    outcomes.append(o)
            if not outcomes:
                continue
            wins = sum(1 for o in outcomes if o > 0)
            count = len(outcomes)
            win_rate = wins / count
            avg_return = float(np.mean(outcomes))
            reliable = (
                count >= self.min_samples
                and win_rate >= self.min_winrate
                and avg_return > 0
            )
            card[name] = {
                'dir': direction, 'count': count,
                'win_rate': round(win_rate, 4),
                'avg_return': round(avg_return, 4),
                'reliable': reliable,
            }
        return card

    # ── Live confluence check ─────────────────────────────────────────────────
    def confluence(self, df, card, ml_signal):
        """
        Does a *reliable* signal fire on the latest candle in the same direction
        as ml_signal? Returns (agrees, boost, best_name, best_winrate).
        boost is always 1.0 here (gate-only design); kept for interface stability.
        """
        if not card or ml_signal == 0:
            return False, 1.0, None, 0.0

        latest = self._detect(df)
        best_wr = 0.0
        best_name = None
        for name, info in latest.items():
            if info['dir'] != ml_signal:
                continue
            if (len(df) - 1) not in set(info['idx'].tolist()):
                continue
            meta = card.get(name)
            if meta and meta['reliable'] and meta['win_rate'] > best_wr:
                best_wr = meta['win_rate']
                best_name = name

        if best_name is None:
            return False, 1.0, None, 0.0
        return True, 1.0, best_name, best_wr
