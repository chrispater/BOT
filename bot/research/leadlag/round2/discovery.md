# Lead-lag discovery, round 2 (1h / 4h / 1d)

## Part A: implementation choices, written BEFORE any test result was computed

This part was written after checking data quality (listed in part C) and before the grid was run.
It fixes every choice that section 2 of PLAN.md and the task brief leave open. Nothing here was
changed after results were seen.

### A1. Grid (as pre-registered)
- Timeframes are 1h, 4h and 1d.
- (L, H) pairs are (1,1), (2,1), (2,2), (3,1) and (3,3). L is the number of lead bars in the signal. H is the number of bars held, either 1 or L. For L = 1 the two holding options are the same, so there are 5 combinations, not 6.
- Directions: `cont` buys the target when s > thr. `rev` buys the target when s < -thr. Every trade is a long trade in the target.
- Thresholds are 0, and the 70th and 80th percentiles of |s| over the event set's own discovery sample. They are stored as fixed numbers in `all_tests.csv`.
- That gives 5 × 2 × 3 = **30 tests per (pair, timeframe)**.
- Pairs:
  - **F1, crypto → crypto-linked equities.**
    - BTC → COIN, MSTR, MARA, RIOT, HOOD, IBIT, CONL, SBIT.
    - ETH → ETHA, COIN.
    - XRP → the XRP ETFs `XRP` (Bitwise) and `XXRP`, intraday only, because no XRP ETF is in the daily file.
    - The other XRP ETFs are excluded before results (GXRP, TOXR, XRPR, XRPC, XRPT, XRPI, XRPZ, UXRP). Each has fewer than 82% of its 5-minute bars populated (TOXR 14%, XRPR 26%), so a 0.20% round-trip cost is not realistic for them. This is the same exclusion as round 1.
    - Daily pairs: 10. Intraday pairs: 12.
  - **F2, crypto → crypto:** all 210 ordered pairs of the 15 coins.
  - **F3, equities → crypto:** SPY, QQQ, COIN, MSTR and NVDA → each of the 15 coins, 75 pairs.
  - **F4, sector leader → followers:** 16 pairs.
    - NVDA → AMD, MU, AVGO, ARM, SMH, SOXL, NVDL.
    - TSLA → TSLL.
    - SPY → TQQQ, SOXL, IWM, ARKK.
    - QQQ → TQQQ, SOXL, IWM, ARKK.
  - **F5, own momentum/reversal:** the lead is the target itself, for every F1–F4 target. It is a control and can never be promoted. Its tests are still logged and **included in the BH denominator**, so the correction is conservative.
- A pair is tested only where data exists. Some pairs cannot reach both halves, for example IBIT, ETHA, SBIT and ARM on daily bars, or XRP-USD daily with its 2021–23 gap. Their tests are still counted and logged. They fail the both-halves gate automatically.

### A2. Prices and timing (all times ET unless marked UTC)
- **Crypto intraday.** The mark C(T) is the close of the Coinbase hourly bar that ends at T. Entry at T is the open of the hourly bar that starts at T. 4h crypto bars are anchored at 00:00 UTC. Daily crypto bars are UTC days.
- **Equity intraday.** Built only from `equity_5min.pkl`.
  - `open_at(T)` is the open of the first 5-minute bar that starts in [T, T+15m).
  - `close_by(M)` is the close of the last 5-minute bar that starts in [M-15m, M).
  - The 16:00 price is `close_by(16:00)`.
- **Equity 1h bars are clock-aligned:** [9:30–10:00) (a half bar), then [10–11), …, [15–16). This is the broker/live-loop grid, and it lines up with the crypto hourly marks.
  - Equity 1h marks are `close_by` at 10:00, 11:00, …, 16:00.
- **Equity 4h bars:** [9:30–13:30) and [13:30–16:00). Their marks are `close_by` at 13:30 and 16:00.
- **Equity daily bars:** broker daily open and close, 2021–2024.
- **Equity lead signal.** s = mark_k / mark_(k-L) - 1 on the lead's own bar grid. mark_k is the latest mark at or before the decision time τ. Marks are close-to-close across sessions, so a window that crosses the open includes the overnight gap.
- **Crypto lead signal.** s = C(T) / C(T - L·b) - 1, where T is the latest hourly mark at or before τ and b is the bar length (1h or 4h). For equity targets on the 4h grid, the crypto window is therefore a rolling L×4h window that ends at the latest full hour. On daily bars it is the latest complete UTC day(s).
- **Equity targets, 1h.**
  - τ = 10:00, 11:00, …, 15:00.
  - Entry is `open_at(τ+5m)`.
  - Exit is at τ + H hours: `open_at(exit+5m)`, or the 16:00 close when the exit time is 16:00.
  - Only events with τ + H ≤ 16:00 are used. **No overnight 1h trades.** The overnight crypto → equity case was round 1's family A.
- **Equity targets, 4h.** Two decisions per session:
  - τ = 13:30: entry 13:35.
  - τ = 09:30 of the next session, for the bar that closed at 16:00: entry 09:35.
  - For this 09:30 decision, a crypto lead window ends at 09:00, which includes overnight crypto moves. An equity lead window ends at the prior 16:00 close.
  - Exit at the end of the H-th bar: `open_at(13:35)` for a morning bar, or the 16:00 close for an afternoon bar. Multi-bar holds cross overnight.
- **Equity targets, 1d.**
  - Entry is the session open. Exit is the close of the H-th session.
  - A crypto lead uses the UTC days that end at or before 00:00 UTC of the entry date, i.e. up to 20:00 ET the prior evening.
  - An equity lead uses closes up to the prior session.
- **Crypto targets with a crypto lead (native grid).**
  - τ is the bar close, which is the next bar's open.
  - Entry is the next bar's open (1h, 4h or UTC day), as the brief specifies.
  - Exit is the close of the H-th bar.
  - **Known limitation:** hourly data cannot show a price 5 minutes after τ, so crypto-target intraday entries have 0 minutes of latency. This is flagged, not fixed.
- **Crypto targets with an equity lead (F3).**
  - 1h: τ = 10:00, …, 16:00 ET, and entry is the crypto open at τ.
  - 4h: τ = 13:30 or 16:00, and entry is the crypto open at the next full hour (14:00 or 16:00).
  - 1d: τ = 16:00 ET, and entry is the open of the next UTC day.
  - Hold H × (1h, 4h or 1d).
  - **The same-day close-to-close daily variant is NOT tested.** It needs a crypto price at 16:05 ET, and hourly crypto data overlaps the 2021–2024 daily equity data only in Sep–Dec 2024.
- **No daily bars are ever rebuilt from intraday data.** 2025–26 intraday data lies inside the daily holdout window, so rebuilding daily bars from it would leak.

### A3. Trades and statistics
- **One position per rule; trades do not overlap.** Events are scanned in time order. A signal is taken only if its entry time is at or after the previous trade's exit time. This matters only when H > 1, and for F3 4h, where the 13:30 and 16:00 events can overlap.
- Net = gross − cost. Cost is 0.20% round trip for equity targets and 0.40% for crypto targets.
- **p-value** (as in round 1): one-sided t-test of H1: mean net > 0. p = max(per-trade p, day-clustered p). The day-clustered p sums net return per date and t-tests the daily sums. The date is the ET session date for anything that involves equities, otherwise the UTC date.
- **Halves.**
  - Daily data is split at 2023-01-01.
  - Crypto-only intraday data (2024-09-01 → 2026-06-30) is split at 2025-08-01, the date midpoint, which gives 334 and 334 days.
  - Intraday data involving equities (2026-02-23 → 2026-06-30) is split at 2026-04-27, the first session after the date midpoint, which gives 44 and 45 sessions.
- **Correction:** Benjamini–Hochberg at FDR 10% across **all** tests, including F5 and tests that cannot reach 30 trades.
- **Candidate gate.** A test is a candidate only if all of the following hold:
  1. q ≤ 0.10.
  2. n ≥ 30 trades.
  3. Mean net > 0 in each half.
  4. **Control (i):** the rule's mean net is strictly above the target's unconditional mean net over the same windows (all events with a valid signal and return, cost subtracted).
  5. **Control (ii):** the rule's mean net is strictly above the same rule with the lead swapped out. The swapped rule uses the same τ grid, L, H, direction, the same percentile threshold on the control's own |s|, and the same trade-selection rule.
     - For equity targets the swap is SPY; for a SPY lead it is QQQ.
     - For crypto targets it is BTC. When the lead is BTC, BTC cannot serve as a separate market control, so the swap is the **target's own lagged return (F5 logic)**.
     - For crypto targets, the crypto control signal uses crypto marks up to the entry time. This is the freshest crypto information, so it is the harder control to beat.
     - If the swapped rule has 0 trades, this gate passes.
  6. The test is not in F5.
- **Promotion if more than 10 pass:** sort by p. Keep only the best test per (lead, target, timeframe). Take the first 10.
- **Descriptive only (never used for selection):**
  - The gross information coefficient, i.e. the Pearson correlation of s with the gross window return over all events.
  - The per-pair best timeframe, meaning the smallest-p cell per pair and timeframe.
  - The own-momentum control for every rule.
- **Sensitivity checks on candidates and top near-misses (not counted as tests, and not gates):**
  - Mean net at double cost.
  - For crypto targets, entry delayed by one bar.

---

## Part B: results (written after the run)

### Bottom line
**31,380 tests. No survivors. The smallest q-value is 1.0, and the candidate list is empty.**
- The smallest p is 0.00082. With 31,380 tests, BH at 10% would need p ≤ 3.2×10⁻⁶ for even one discovery. No test comes within three orders of magnitude of that.
- Nothing survives even when BH is run within a single family and timeframe. The closest is F4 4h, with a within-cell q of 0.39.
- The verifier has nothing to run.

| Family | 1h | 4h | 1d | Total |
|---|---|---|---|---|
| F1 crypto → crypto-linked equities | 360 | 360 | 300 | 1,020 |
| F2 crypto → crypto | 6,300 | 6,300 | 6,300 | 18,900 |
| F3 equities → crypto | 2,250 | 2,250 | 2,250 | 6,750 |
| F4 sector leader → followers | 480 | 480 | 480 | 1,440 |
| F5 own momentum/reversal (control) | 1,110 | 1,110 | 1,050 | 3,270 |
| **All** | 10,500 | 10,500 | 10,380 | **31,380** |

### Is there anything beyond chance? No.
There was one descriptive check, and it was not a test. For every non-F5 cell, the lead signal was shifted in time by a random 25–75% of the sample. This breaks the lead timing but keeps the signal distribution and the market regime. The same 28,110 tests were then rerun. The real grid produced no more small p-values than the shifted ones:

| Run | Tests with p < 0.05 | p < 0.01 | p < 0.001 | Smallest p | "Robust" (n ≥ 30, both halves +) | Robust with p < 0.01 |
|---|---|---|---|---|---|---|
| **Real** | 160 | 17 | 1 | 0.00082 | 1,510 | 14 |
| Placebo, seed 11 | 193 | 20 | 0 | 0.00139 | 1,422 | 18 |
| Placebo, seed 22 | 176 | 24 | 1 | 0.00037 | 1,368 | 20 |
| Placebo, seed 33 | 191 | 23 | 3 | 0.00047 | 1,439 | 19 |

The gates short of BH are also not selective. 903 non-F5 tests pass n ≥ 30, both halves positive, and both controls. The placebo grids produce about as many "robust" rules. This is why selection has to come from the corrected p-value and not from the robustness filter.

### What each family showed
- **Crypto → crypto (F2) at 1h and 4h: costs kill everything.**
  - 0 of 6,300 1h tests and 25 of 6,300 4h tests have a positive net mean.
  - The best gross edge per trade is 0.23% at 1h and 0.74% at 4h. The median is about 0.01%. The round-trip cost is 0.40%.
  - There is real gross structure at 1h. 56 of 630 H = 1 cells have |IC·√n| > 3, against about 2 expected by chance. Most of it is slight negative cross-autocorrelation. It is one to two orders of magnitude too small to pay the spread.
- **Crypto → crypto (F2) at 1d.**
  - 38% of tests are positive net. Every pair's smallest p falls at 1d.
  - The best are XLM 3-day up (p80) → buy LINK for 3 days (72 trades, +3.2% net, p = 0.0033) and BCH 1-day drop (p80) → ATOM next day (141 trades, +1.85%, p = 0.0053).
  - Both look like ordinary alt-coin co-movement, and both fail BH by a wide margin.
- **Crypto → crypto-linked equities (F1).**
  - Nothing reached p < 0.005.
  - BTC 2-day up (p80) → MSTR for 2 days: 79 trades, +2.2%, p = 0.006. It is just MSTR's own momentum: the same rule on MSTR's own return gives +2.2%.
  - The intraday version (1h/4h, 89 sessions, 2026) repeats round 1's null. The XRP → XRP-ETF link still has the wrong sign, or too few trades, once thresholded.
- **Equities → crypto (F3).** No rule had p < 0.01. See the regime note below about the 4h cells.
- **Sector leader → followers (F4).** This is the only family with a cluster of small p-values, and it is still at chance level overall.
  - "NVDA fell over the last two 4h bars → buy AMD, ARM, SOXL or SMH for two 4h bars" gives 49 trades, 69% wins, +1.8% net and p = 0.0008 for AMD.
  - That is one pattern repeated across correlated followers, in 89 sessions of a strong semiconductor rally. Unconditional AMD returned +0.64% per window.
  - QQQ/SPY → SOXL/TQQQ at 1h is the targets' own momentum (see the table below).

### Near-misses (the 10 smallest non-F5 p-values) and why they fail
Every one fails BH (q = 1.0). The other gates are shown for completeness. "Own" is the same rule using the target's own lagged return (F5 logic). "Ctrl" is the SPY/BTC-lead control.

| # | Rule | Trades | Win | Mean net | p | Halves (net) | Uncond | Ctrl | Own | Other reason it is not a lead |
|---|---|---|---|---|---|---|---|---|---|---|
| 19423 | F4 4h NVDA 2-bar return < 0 → buy AMD 2 bars | 49 | 69% | +1.76% | 0.0008 | +2.05 / +1.55 | +0.64% | +1.42% | +1.06% | Semis rally in 89 sessions. The SPY-lead control earns most of the edge. |
| 9326 | F4 1h QQQ 3h > +1.05% → buy SOXL 3h | 32 | 72% | +1.44% | 0.0015 | +1.33 / +1.53 | +0.23% | +0.83% | **+1.45%** | Identical to SOXL's own momentum. |
| 26578 | F2 1d XLM 3d > +8.2% → buy LINK 3d | 72 | 64% | +3.17% | 0.0033 | +3.15 / +3.22 | +0.07% | +0.80% | −0.42% | No structural reason. Alt co-movement. |
| 19435 | F4 4h NVDA 3-bar < 0 → buy AMD 3 bars | 34 | 62% | +2.62% | 0.0035 | +1.43 / +3.56 | +1.28% | +1.70% | +1.56% | Same as #19423. |
| 29935 | F4 1d NVDA 3d < 0 → buy AVGO 3d | 181 | 59% | +0.82% | 0.0043 | +0.45 / +1.23 | +0.25% | +0.50% | +0.38% | Small margin over the SPY control. |
| 9296 | F4 1h QQQ 3h > +1.05% → buy TQQQ 3h | 32 | 63% | +0.71% | 0.0045 | +0.81 / +0.62 | −0.10% | +0.26% | **+0.68%** | Own momentum. |
| 19573 | F4 4h NVDA 2-bar < 0 → buy SOXL 2 bars | 49 | 71% | +3.44% | 0.0050 | +2.84 / +3.90 | +1.12% | +2.70% | +2.39% | Semis beta. |
| 24215 | F2 1d BCH 1d < −5.1% → buy ATOM 1d | 141 | 58% | +1.85% | 0.0053 | +2.44 / +0.64 | −0.22% | +1.01% | +1.18% | Fades in the second half. Generic dip-buying. |
| 19585 | F4 4h NVDA 3-bar < 0 → buy SOXL 3 bars | 34 | 56% | +4.64% | 0.0060 | +1.61 / +7.04 | +2.07% | +3.27% | +2.25% | Driven by the second half. |
| 21046 | F1 1d BTC 2d > +5.3% → buy MSTR 2d | 79 | 61% | +2.17% | 0.0063 | +1.58 / +3.13 | +0.29% | −0.03% | **+2.21%** | MSTR's own momentum. |

The full top 25 is in `near_misses.csv`. All of these are also present in the placebo-level count of p < 0.01 tests.

### A regime observation, not a candidate: US-afternoon crypto reversal, Feb–Jun 2026
- In F3 at 4h, every one of the 225 cells has a negative gross IC. The mean is −0.20, and 79 cells have |z| > 3. The pattern: when US equities rose over the past day, crypto fell over the next 4h, from 14:00 or 16:00 ET.
- Checks:
  - The timestamps are aligned: IBIT's clock-hour return and BTC's hourly bar correlate at 0.999 at lag 0, and at −0.07 and −0.03 at ±1h.
  - It is robust to outliers: Spearman −0.38 for COIN → DOGE, and −0.33 after dropping the 5 largest moves.
- **It is not a lead-lag effect.** On the same grid, the crypto's own 8h return (IC −0.47 for DOGE) and BTC's return (−0.39) predict just as well as COIN (−0.45) or SPY (−0.31). The pre-registered controls exist to reject exactly this.
- **It is specific to this regime.** The same own-return statistic at 14:00/16:00 ET is about 0 in Sep 2024–Jul 2025 (−0.09 to +0.18 across six coins) and in Aug 2025–Feb 2026 (−0.18 to +0.09). It is strongly negative (−0.15 to −0.54) only in Feb–Jun 2026, the one window that has equity intraday data.
- It cannot be traded under the grid's rules either. The thresholded 4h rules have 9–18 trades, and the thr-0 rules do not beat the controls.
- If the owner wants to pursue it, it needs its own pre-registered test ("crypto own 8h return at 14:00/16:00 ET → reverse next 4h") on data it has not seen. The obvious candidate is the Jul–Sep 2026 intraday holdout, but only if it is used for this one question.

### Per-pair best timeframe (descriptive only, never used for selection)
This is the smallest-p cell per pair and timeframe. With this much testing, these are mostly noise rankings.
- **F2 (210 pairs):** the best timeframe is 1d for all 210, because 1h and 4h cannot beat 0.40% cost. The median smallest p is 0.11 at 1d, 0.83 at 4h and 1.0 at 1h.
- **F3 (75 pairs):**
  - SPY and QQQ leads: mostly 4h (26 of 30), because of the regime pattern above.
  - MSTR: mostly 1d (14 of 15).
  - COIN and NVDA: split.

| Lead → target | Min p 1h | Min p 4h | Min p 1d | Best tf | Best cell |
|---|---|---|---|---|---|
| BTC → COIN | 0.327 | 0.256 | 0.084 | 1d | L2 H2 rev p70, n=83, +1.14% |
| BTC → CONL | 0.250 | 0.142 | 0.007 | 1d | L2 H2 rev p80, n=31, +6.00% |
| BTC → HOOD | 0.166 | 0.093 | 0.057 | 1d | L3 H3 rev p80, n=41, +2.48% |
| BTC → IBIT | 0.811 | 0.414 | 0.033 | 1d | L3 H3 rev p80, n=11, +3.16% |
| BTC → MARA | 0.481 | 0.037 | 0.021 | 1d | L2 H2 rev p70, n=87, +2.04% |
| BTC → MSTR | 0.857 | 0.586 | 0.006 | 1d | L2 H2 cont p80, n=79, +2.17% |
| BTC → RIOT | 0.628 | 0.016 | 0.059 | 4h | L2 H2 cont 0, n=54, +1.24% |
| BTC → SBIT | 0.148 | 0.029 | 0.300 | 4h | L3 H3 cont p80, n=10, +2.47% |
| ETH → COIN | 0.214 | 0.066 | 0.062 | 1d | L2 H2 cont p70, n=112, +1.08% |
| ETH → ETHA | 0.683 | 0.350 | 0.114 | 1d | L3 H3 rev p70, n=11, +1.80% |
| XRP → XRP (ETF) | 0.282 | 0.251 | n/a | 4h | L2 H1 cont p80, n=17, +0.25% |
| XRP → XXRP | 0.149 | 0.170 | n/a | 1h | L3 H3 rev p80, n=21, +0.49% |
| NVDA → AMD | 0.037 | 0.001 | 0.091 | 4h | L2 H2 rev 0, n=49, +1.76% |
| NVDA → ARM | 0.096 | 0.006 | 0.132 | 4h | L2 H2 rev 0, n=49, +2.00% |
| NVDA → AVGO | 0.356 | 0.097 | 0.004 | 1d | L3 H3 rev 0, n=181, +0.82% |
| NVDA → MU | 0.079 | 0.027 | 0.137 | 4h | L2 H2 rev 0, n=49, +1.64% |
| NVDA → NVDL | 0.091 | 0.050 | 0.008 | 1d | L1 H1 rev p80, n=36, +1.59% |
| NVDA → SMH | 0.118 | 0.012 | 0.020 | 4h | L3 H3 rev 0, n=34, +1.20% |
| NVDA → SOXL | 0.018 | 0.005 | 0.015 | 4h | L2 H2 rev 0, n=49, +3.44% |
| QQQ → ARKK | 0.049 | 0.281 | 0.101 | 1h | L3 H3 cont p70, n=32, +0.27% |
| QQQ → IWM | 0.042 | 0.304 | 0.060 | 1h | L3 H3 rev p80, n=18, +0.31% |
| QQQ → SOXL | 0.001 | 0.027 | 0.038 | 1h | L3 H3 cont p70, n=32, +1.44% |
| QQQ → TQQQ | 0.004 | 0.054 | 0.030 | 1h | L3 H3 cont p70, n=32, +0.71% |
| SPY → ARKK | 0.170 | 0.275 | 0.162 | 1d | L3 H1 cont p70, n=161, +0.21% |
| SPY → IWM | 0.042 | 0.180 | 0.272 | 1h | L3 H3 rev p80, n=19, +0.30% |
| SPY → SOXL | 0.031 | 0.031 | 0.049 | 1h | L2 H2 cont p80, n=31, +1.03% |
| SPY → TQQQ | 0.063 | 0.078 | 0.049 | 1d | L3 H3 rev 0, n=185, +0.90% |
| TSLA → TSLL | 0.212 | 0.084 | 0.020 | 1d | L3 H3 cont p80, n=37, +4.97% |

All 210 F2 pairs and all 75 F3 pairs are in `per_pair_tf.csv`.

---

## Part C: data findings and caveats
- **Checks that passed.**
  - The daily equity bars are split-adjusted: no NVDA, TSLA, AVGO or MSTR split jumps.
  - Daily and hourly crypto agree exactly over Sep–Dec 2024, for both open and close.
  - The 5-minute equity file has 89 sessions × 78 bars. There are no early closes, and each session runs 09:30–15:55.
  - Crypto and equity timestamps line up: IBIT vs BTC correlates at 0.999 at lag 0.
  - Every hand-checked trade matched the code: equity 1h, including a 16:00 exit; equity 4h morning and 2-bar holds; crypto 4h; F3 4h; and daily SPY → ETH, including the BTC control.
- **SOXS daily is corrupt.**
  - Prices are off by 10× on 2022-03-15/16 and 2022-03-28, for example 11,400 → 110,600 → 9,440.
  - Volume is zero on 31% of days, covering all of 2021 and part of 2022.
  - SOXS is not used here, but any other study that uses it is affected.
- **COIN daily 2021-04-14 (listing day).** The open is 250, the reference price, and lies below the day's low of 310. This affects at most one daily COIN trade.
- **XRP-USD daily** has a 905-day gap, 2021-01-19 → 2023-07-13 (the Coinbase delisting). It is kept as NaN and never bridged. XRP daily tests therefore have almost no first-half data.
- **Crypto hourly** has two 6-hour gaps, in all 15 coins at the same hours: 2025-10-25 21:00 UTC and 2026-05-08 07:00 UTC. Windows that touch them are dropped.
- **XRP ETFs.** 8 of 10 are thin (TOXR has 14% of 5-minute bars, XRPR 26%, GXRP 40%), so they were excluded before the run. Even `XRP` (96%) and `XXRP` (99%) are thin compared with the 0.20% cost assumption.
- **No first-half daily data.** IBIT, ETHA, SBIT and ARM have no daily data before 2023, and NVDL, TSLL and CONL only a little. The coordinator removed 2025 daily data. So daily tests on these can never pass the both-halves gate. They are counted in the BH denominator anyway.
- **The equity-involved intraday sample is short.** It is 89 sessions (Feb 23–Jun 30, 2026), and 4h equity grids give at most about 176 events. Thresholded 4h rules mostly cannot reach 30 trades: only 35–43% of F1, F3 and F4 4h tests do. As in round 1, "nothing survived" at 1h and 4h means no evidence, not proof of no effect. That sample is also a single regime: a semiconductor rally, and the afternoon crypto reversal above.
- **Crypto intraday entry latency is zero.** Entries are at the next hourly or 4h open, exactly at the signal time, because there is no sub-hourly crypto data. This favours the F2 intraday rules, and they still all lose after costs.
- **Overlap of discovery and holdout.** The intraday discovery period (up to 2026-06) overlaps the daily holdout window (2025-01 → 2026-09). Daily bars were never rebuilt from intraday data, so no daily holdout information was used. A future verifier should also avoid doing that.
- **`equity_1h_broker.pkl` was not used**, following the brief and round 1's finding that its bars are defective.
- **SMCI 5-minute data** shows a −27% open gap on 2026-03-20. It was not investigated, because SMCI is not a target.

## Part D: files and rerun
- `discover.py` builds every event set and runs all 31,380 tests. It writes `all_tests.csv` (one row per test: frozen threshold, stats, halves, controls, q) and `run_meta.json`. It takes about 18 s on one core.
- `describe.py` produces descriptive output only: `placebo_calibration.csv`, `ic_summary.csv`, `per_pair_tf.csv` and `near_misses.csv`. It takes about 1 min.
- `discovery.json` is the machine-readable result: `total_tests`, `correction`, `candidates` (empty) and the near-misses.
- To rerun: `OMP_NUM_THREADS=1 python3 discover.py && OMP_NUM_THREADS=1 python3 describe.py`.
