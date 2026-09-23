# Lead-lag discovery: result is NULL (no candidates)

**Bottom line:** 648 pre-registered tests, Benjamini–Hochberg at FDR 10%. **None survived.** The smallest q-value was 1.0. Only 6 tests had p < 0.05. Pure noise would give about 32. The candidate list is empty. There is nothing for the holdout agent to test.

## Data findings (read first)
- **`bars5.pkl` is not 5-minute data.** Every symbol has exactly 7 bars per session, at 09:30, 10:30, …, 15:30 ET. These are session-aligned 60-minute bars, and the 15:30 bar is a half hour. They are internally consistent: the gap from one bar's close to the next bar's open is about 1 bp for SPY.
- **`bars.pkl` hourly bars are badly defective.** There are big gaps from one bar's close to the next bar's open: COIN median 21 bp and 90th percentile 117 bp; CONL 90th percentile 227 bp. On 10% of days the 16:00 close differs from bars5 by more than 0.8% for COIN and more than 1.1% for CONL. The 10:00 bars often carry a small fraction of normal volume (for example SPY 2026-03-10 10:00 has 87k shares).
- So **all equity prices come from bars5**, which has 89 sessions (2026-02-23 to 2026-06-30). The first discovery half has only about 27 sessions. There were no zero-volume bars in either equity file.
- The finest entry grid is 10:30, 11:30 and so on. A 9:50 entry cannot be priced from this data. Crypto hourly data has one 6-hour gap (2026-05-08).
- The illiquid XRP ETFs (TOXR, GXRP, XRPR) trade a few hundred to a few thousand shares an hour. A 0.20% round-trip cost is optimistic for them, so they were excluded as targets. XRP (the Bitwise ETF) and XXRP were used instead.

## Tests run (all specified before looking at results; no extra family added)
Price definitions: P_eq(T) is the open of the bars5 bar starting at T, and P_eq(16:00) is the close of the 15:30 bar. P_c(T) is the open of the Coinbase hourly bar starting at T (ET). Every rule was tested at threshold 0 and at the 70th and 80th percentile of |signal|, in both directions (buy the target when s > thr, or when s < −thr). Every trade buys the target (long only). Costs are 0.20% per equity round trip and 0.40% per crypto round trip.
- **A** (216 tests): s = P_c(10:00)/P_c(16:00 of the previous session) − 1. Buy at P_eq(10:30), exit at 11:30, 12:30 or 16:00. Pairs: BTC→IBIT, SBIT, COIN, MSTR, MARA, RIOT, HOOD, CONL; ETH→ETHA, COIN; XRP→XRP, XXRP.
- **B** (144 tests): for H = 10..14, s = P_c(H:00)/P_c(H−k:00) − 1 with k = 1 or 3. Buy at P_eq(H:30), exit at P_eq(H+1:30). Same 12 pairs, pooled across the 5 hourly slots.
- **C** (120 tests): s = the session return P_eq(16:00)/P_eq(09:30) − 1 of COIN, MSTR, IBIT, SPY or QQQ. Buy BTC or ETH at P_c(17:00), exit at P_c(10:00) the next day (17 hours) or 17:00 the next day (24 hours).
- **D** (72 tests): same as A with a 16:00 exit, restricted to the first session after a weekend or holiday. There are only about 17 such sessions, so this family can never reach 30 trades.
- **E** (96 tests): s is the lead's return over k = 1 or 3 hours ending at T. Buy at P_c(T+1h), exit at P_c(T+2h). Pairs: BTC→ETH, SOL, XRP, DOGE and ETH, SOL, XRP, DOGE→BTC.

## What looked promising but failed
- **A, BTC down overnight → buy MARA to the close** (51 trades, +0.68% net per trade, t = 1.33). This loses in the first half (−1.17%). It is also no better than the placebo that uses SPY or QQQ's own overnight-plus-first-hour drop, which gives +0.47% and +0.75%. It is generic buy-the-dip market beta, not a crypto lead.
- **Nine rules passed the "≥30 trades and positive in both halves" gate.** All of them are family A threshold-0 rules on HOOD, CONL, COIN and SBIT. Each has p > 0.14 and beats its unconditional same-window return by a trivial amount or not at all. SPY/QQQ controls match them.
- **XRP → XRP ETFs, family A continuation.** This had the strongest Pearson correlation of the whole study: IC 0.17–0.25 at n = 88. Once thresholded, the trades are too few (7 at p80) to survive, and the unthresholded version is not significant after costs. A plausible explanation is stale ETF prints at 10:30 in thin XRP ETFs. That would be an artifact, not an edge.
- **The weekend family D produced the smallest raw p-values** (for example BTC down weekend → MARA, 5 of 5 winners). These come from 3–6 trades, all in the second half, and are meaningless.
- **E is hopeless at a 0.40% crypto cost.** The average absolute next-hour move is 0.32–0.48%, so every E rule loses 0.3–0.43% per trade (t around −30). Gross ICs are within ±0.03.
- **C: the equity session positively predicts BTC and ETH over the next 24 hours** (IC 0.10–0.13 for IBIT, SPY and QQQ leads). Thresholded rules have 12–14 trades and flip sign between the two halves.

## Caveats
- The sample is short: 89 equity sessions, with only 27 in the first half. A real effect of a realistic size (IC around 0.05) could not be detected here. So "nothing survived" means "no evidence", not "proof of no effect".
- The entry timing is conservative (10:30 ET) because of the data grid. An effect that lives in the 9:45–10:30 window would be missed.
- One-sided p-values; BH assumes positive dependence, which is reasonable here.

## Files
- `leadlag_discovery.py` builds every test and writes `all_tests.csv`, one row per test with q-values and half-period stats.
- `controls.py` runs the SPY/QQQ placebo controls and writes `controls.csv`. It also hand-checks one trade.
- `ic_diagnostics.csv` has the gross Pearson correlations. `discovery.json` is the machine-readable result (empty candidates).
- To rerun: `OMP_NUM_THREADS=1 python3 leadlag_discovery.py && python3 controls.py` (about 10 seconds).

---

## Addendum: rerun of families A, B and D on genuine 5-minute bars (`bars5min_raw.pkl`)
The coordinator later supplied real 5-minute broker bars (filler removed; 78 bars per session for liquid names, down to 64 for the thin XRP ETF). They match bars5's hourly opens exactly. Families A, B and D were rerun with **the same pairs, thresholds, directions, costs, evaluation and promotion rule**. Only the timing changed:
- **A and D:** s = P_c(09:00 ET)/P_c(16:00 of the previous session) − 1. This is the latest hourly crypto price known before 9:50. **Entry** is the open of the first 5-minute bar starting at or after 09:50. **Exit** is the open of the first bar at or after 10:50 (+1h) or 11:50 (+2h), or the close of the last bar of the session (16:00). If the needed bar does not start within 15 minutes of the target time, the trade is skipped.
- **B:** s = P_c(H:00)/P_c(H−k:00) − 1 for H = 10..14. **Entry** is the first 5-minute bar open at or after H:05. **Exit** is the first 5-minute bar open at or after (H+1):05.
- **Controls:** SPY or QQQ's own return from the prior close to 09:45 (A and D), or over the same k hours (B). These are now fully causal.
- Code: `leadlag_rerun_5m.py`. Outputs: `rerun5m_tests.csv`, `rerun5m_controls.csv`, `rerun5m_ic.csv`, and `combined_tests.csv` (all tests together, with `q_all`).

**Test count:** the rerun added 432 tests (A 216, B 144, D 72). **The new total is 1,080 tests.** Benjamini–Hochberg at FDR 10% was applied over all 1,080.

**Result: still no survivors.**
- The smallest q-value is 1.0, both over all 1,080 tests and in a sensitivity view where A, B and D are replaced by their reruns (648 tests).
- Among the reruns, 10 of 432 tests had p < 0.05, against about 22 expected by chance. The smallest p is 0.015.
- **The smallest p-values all come from family D**, with 3–5 trades each. Examples: XRP weekend up → XXRP (3 trades), and BTC weekend down → MARA, MSTR, CONL, RIOT or COIN (5 trades). The SPY/QQQ weekend-gap controls do about as well (+1–3%, t 1.5–2.8), so these are market-gap effects on a handful of Mondays.
- **Four rules passed "30+ trades and positive in both halves":**
  - XRP up overnight → XXRP to the close: 34 trades, +0.36% net, p = 0.30.
  - The same signal → XRP ETF: +0.13% net, p = 0.35.
  - XRP up → XXRP, +2h exit: +0.12% net, p = 0.38.
  - B, BTC 3h up (p80) → HOOD next hour: 44 trades, +0.02% net, p = 0.45.
  
  None comes close to significance.
- **The one relationship that persists is XRP overnight → XRP ETFs** (continuation). With the causal 9:50 entry, its correlation with the next-period return is 0.18–0.21 (n = 88), close to the earlier 0.17–0.25. So it is not only a stale-10:30-print artifact. However, it is weak in absolute terms: at threshold 0 the edge is ≤0.36% net per trade with t ≤ 0.5, and at p80 there are only 8 trades. It fails the correction by a wide margin.
- **BTC overnight → crypto-equity to the close still leans reversal** (correlation −0.07 to −0.13), and SPY/QQQ controls match it. This is generic dip-buying.

`discovery.json` is unchanged: the candidate list stays empty. The original total was 648 tests; with the rerun it is 1,080.
