"""
Signal Reliability Scorer
─────────────────────────
Measures, per discrete technical-signal type, how profitable it has actually
been on a given coin — using the bot's real leverage, stop-loss, take-profit
and taker fees. A signal counts as "reliable" only if its historical win rate
clears a bar AND its average per-trade return (after fees, in margin terms) is
positive.

This is a confluence layer: the ML model still decides direction, but an entry
is only taken / up-sized when a reliable technical signal agrees with it. That
double-confirmation cuts the loss rate (which protects the compound curve) and
concentrates capital into the highest-conviction setups.

Adapted from a standalone Lock/Float reliability scorecard, but rebuilt for
leveraged crypto futures: it never uses signal-flip exits (which have no stop
loss); every occurrence is scored under the bot's actual SL/TP exit rules.
"""

import numpy as np
import pandas as pd

TAKER_FEE = 0.0006  # matches trading_service.TAKER_FEE


class SignalReliability:
    def __init__(self, leverage=10, stop_loss_pct=0.15, take_profit_pct=0.30,
                 horizon=24, min_winrate=0.60, min_samples=8):
        self.leverage = max(1, leverage)
        self.stop_loss_pct = stop_loss_pct      # % of margin
        self.take_profit_pct = take_profit_pct  # % of margin
        self.horizon = horizon                  # candles to let a signal play out
        self.min_winrate = min_winrate
        self.min_samples = min_samples

    # ── Discrete signal detection ────────────────────────────────────────────
    def _detect(self, df):
        """
        Return a dict {signal_name: {'dir': +1/-1, 'idx': np.array of candle indices}}.
        Each named signal fires at specific candles in one direction.
        """
        out = {}
        n = len(df)
        if n < 30:
            return out

        rsi = df['rsi'].values if 'rsi' in df else None
        sk  = df['stoch_k'].values if 'stoch_k' in df else None
        macd = df['macd'].values if 'macd' in df else None
        macd_sig = df['macd_signal'].values if 'macd_signal' in df else None
        close = df['close'].values
        bb_lo = df['bb_lower'].values if 'bb_lower' in df else None
        bb_hi = df['bb_upper'].values if 'bb_upper' in df else None

        def add(name, direction, mask):
            idx = np.where(mask)[0]
            idx = idx[idx < n - 1]  # need at least one forward candle
            if len(idx):
                out[name] = {'dir': direction, 'idx': idx}

        if rsi is not None:
            add('RSI Oversold (long)', 1, rsi <= 30)
            add('RSI Overbought (short)', -1, rsi >= 70)
        if sk is not None:
            add('Stoch Oversold (long)', 1, sk <= 20)
            add('Stoch Overbought (short)', -1, sk >= 80)
        if macd is not None and macd_sig is not None:
            cross_up = (np.r_[np.nan, macd[:-1]] < np.r_[np.nan, macd_sig[:-1]]) & (macd > macd_sig)
            cross_dn = (np.r_[np.nan, macd[:-1]] > np.r_[np.nan, macd_sig[:-1]]) & (macd < macd_sig)
            add('MACD Cross Up (long)', 1, cross_up)
            add('MACD Cross Down (short)', -1, cross_dn)
        if bb_lo is not None and bb_hi is not None:
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

        # Horizon reached — mark to close
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
        as ml_signal? Returns (agrees: bool, boost: float, best_name, best_winrate).

        boost scales with how far the best agreeing signal's win rate clears the
        bar, capped at 1.5×. No agreement → boost 1.0.
        """
        if not card or ml_signal == 0:
            return False, 1.0, None, 0.0

        latest = self._detect(df)
        best_wr = 0.0
        best_name = None
        for name, info in latest.items():
            if info['dir'] != ml_signal:
                continue
            # did this signal fire on the most recent candle?
            if (len(df) - 1) not in set(info['idx'].tolist()):
                continue
            meta = card.get(name)
            if meta and meta['reliable'] and meta['win_rate'] > best_wr:
                best_wr = meta['win_rate']
                best_name = name

        if best_name is None:
            return False, 1.0, None, 0.0

        # win rate above the bar → up to +0.5x size, linearly to a 90% ceiling
        span = max(0.01, 0.90 - self.min_winrate)
        boost = 1.0 + 0.5 * min(1.0, (best_wr - self.min_winrate) / span)
        return True, round(boost, 3), best_name, best_wr
