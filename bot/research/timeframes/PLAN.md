# Timeframe study — pre-registered plan (committed 2026-09-23, before any results)

Owner's goal: $1MM. The question is which bar timeframe gives the engine
the most edge after costs, for stocks and for crypto, possibly different
per symbol. The same question applies to lead-lag (cross-asset)
relationships.

The danger is picking the best of many symbol × timeframe combinations in
hindsight. With 80 symbols and 3 timeframes, some combination will always
look great by luck. So every choice below is fixed here, before results
exist, and every selection is either made point-in-time or checked on data
it never saw.

## 1. Engine timeframe study

**What we can test, set by the data that exists:**

| Asset | Timeframes | Data | Out-of-sample window |
|---|---|---|---|
| Equities (65) | 1h (existing replay), 2h, 1d | Broker hourly Dec-2025→ (real intraday history starts there; 2024 hourly bars are all filler); broker daily 2021→ | Common window where all three have signals |
| Crypto (15) | 1h (existing, 5 coins), 4h, 1d | Coinbase hourly 2y, daily 2021→ | Last 365 days (1h: its 240 days) |

- Crypto universe: BTC ETH SOL XRP DOGE LTC BCH LINK AVAX ADA DOT UNI
  XLM SHIB ATOM. All are Robinhood-tradable and not halted.
- Equity 2h is built from the broker hourly feed, the same source the live
  loop reads: bars 10:00-12:00, 12:00-14:00 and 14:00-16:00 ET.
- Sub-hourly timeframes are excluded. The loop can run at most hourly, and
  orders fill 5–25 minutes after the snapshot.

**The engine is unchanged.** Only these parameters vary with the timeframe:

- **Forward window:** 5 bars at every timeframe, as live.
- **Label threshold:** 0.4% × √(bar hours). Bar hours are 1 for equity 1h,
  2 for 2h and 6.5 for 1d; 4 for crypto 4h and 24 for 1d. That gives equity
  2h 0.57%, equity 1d 1.02%, crypto 4h 0.80% and crypto 1d 1.96%.
- **Training window:**
  - 500 bars for crypto 4h and 1d and for equity 1d.
  - 300 bars for equity 2h, the most the data allows.
- **Retraining cadence** (live retrains every cycle; research retrains less
  often to fit the compute):
  - Every 5 bars for daily.
  - Every 12 bars for crypto 4h.
  - Every 6 bars for equity 2h.
  - The existing 1h replays retrain daily.
- **Costs:** 0.10% per side for equities and 0.20% for crypto. The
  validation gate charges 0.20% / 0.40% round trip.
- **Everything else is live config:** stops, take-profit, trailing stop,
  breakeven, the validated-only entry gate and the 3-day expectancy
  cooldown. Hold and hysteresis counters count bars.
- **Known approximation:** exits are evaluated only at bar closes. Live
  would check a daily-timeframe position hourly against quotes, so the
  research understates stop protection on the longer timeframes.

**Selection methods compared (per asset class, one portfolio each):**

- **(a) Fixed timeframe for all symbols:** each timeframe on its own.
- **(b) Adaptive, point-in-time:** at every retrain, each symbol trades
  the timeframe whose latest model is validated net of cost with the
  highest net out-of-sample expectancy per hour of holding horizon:
  (gross − round-trip cost) / (5 × bar hours). If no timeframe validates,
  the symbol trades nothing. This uses only data before the decision, so
  its backtest is honest.
- **(c) Hindsight-best timeframe per symbol:** reported only as an upper
  bound on what per-symbol selection could ever add. It uses the future
  and can never be adopted.

**Adoption bar (per asset class), all required:**

1. Beats the live setup on net return over the common window. For
   equities that is fixed 1h. For crypto, the lane is off, so it must beat
   zero.
2. Positive in both halves of the window.
3. Positive at double cost.
4. Max drawdown no worse than 1.5× the fixed-1h equity drawdown on the
   same window.
5. Block-bootstrap probability of a loss over 70 trading days below 25%.

## 2. Multi-timeframe lead-lag study

This extends the 1h/overnight study (section 7 of REPORT.md, nothing
survived) to other timeframes and lags. It uses two independent agents: a
discovery agent and a verifier.

- **Timeframes:** 1h, 4h and 1d. Lags are 1–3 bars. The lead is known at
  bar close; entry is at a price after the signal (next bar open, plus the
  5-minute latency floor intraday). Long-only on tradable symbols.
- **Pair families:**
  - Crypto → crypto-linked equities.
  - Crypto → crypto.
  - Equities → crypto.
  - Sector leader → followers, e.g. NVDA → semis and SPY/QQQ → high-beta
    names.
- **Discovery / holdout split:**
  - Daily data: discovery 2021-01-01 → 2024-12-31, holdout 2025-01-01 →
    2026-09-23.
  - Intraday: discovery up to 2026-06-30, holdout from 2026-07-01, still
    unused.
- **Discovery rules:**
  - Benjamini–Hochberg at 10% across ALL tests.
  - Positive in both halves of discovery, ≥ 30 trades.
  - Beats the SPY/QQQ control and the target's unconditional drift.
  - At most 10 candidates, each fully specified as a frozen rule.
- **Verifier (a separate agent):** runs the frozen rules once on the
  holdout. A rule passes if its holdout return is net-positive at p < 0.05
  (one-sided), positive at double cost, and has ≥ 20 holdout trades.

## 3. Rollout

Nothing goes live on a backtest alone:

1. Anything that passes ships dormant behind a config switch, with tests.
2. The owner approves.
3. It goes live, with its live record tracked separately so it can be
   switched off on evidence.

Research compute runs only outside market hours (9:30–16:00 ET).
