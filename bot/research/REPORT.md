# Replication backtest — items 1–5 (2026-09-22)

Owner request: *"pull the full validated/not-validated breakdown across all 33
universe symbols. add more symbols if needed and backtest items 1-5 to find
our path forward."*

Nothing in this report changes live trading. Code: `bot/research/`.

## Method

- **Data.** Real hourly bars for 65 symbols, 2025-12-22 → 2026-09-22. That's
  the 33-symbol live universe plus 32 additions chosen in advance for
  liquidity and relevance, never by past performance. The additions are 12
  single names, 7 sector/commodity ETFs, 5 leveraged longs and 8 inverse ETFs.
  Earlier "hourly" history from the broker is synthetic filler (flat,
  zero-volume, flagged `interpolated`) and was excluded.
- **Out-of-sample window.** 2026-02-20 → 2026-09-22, which is 148 trading
  days and ~886 decision points per symbol. The model used on each day was
  trained only on the 60 days before it.
- **Fidelity.** The model, labels, features, walk-forward validation and
  setups are imported from `bot/strategy/`, not reimplemented. The simulator
  calls the engine's own `exit_decision`, `entry_ev`, `ensemble` and
  `recent_expectancy`.
  - Fills are at the next bar's open ±0.1%, and exits are priced off that
    open (the live "quote").
  - Cash-account runs model T+1 settlement and the same-day GFV deferral.
    Margin runs model the pattern-day-trader cap.
- **Replica check.** On the final day the replica's validation numbers match
  the live engine's (TSLA +0.0111 net in both; COIN +0.0098 vs +0.0101).
- **Tests.** `test_simulate.py`, 7 tests, all passing.
- **Approximation.** Live retrains every cycle; the backtest retrains once
  per day.

## 1. Validation census (all 65 symbols, 148 days)

Full table: `census.csv` in the run directory.

- **Live universe today:** 6 of 33 pass the cost-aware gate — TSLA, COIN, USO,
  MSFT, WMT, AMZN.
- **Over the whole window:** only **AMD (+0.41%)** and **TSLA (+0.23%)** have
  mean gross expectancy above the 0.2% round-trip cost. Only 2 of the 33 passed
  the cost-aware gate on more than half the days.
- **Across all 65:** 7 clear cost on average — SOXS, TSLQ, SOXL, MARA, TSLL,
  AMD, TSLA.
- **The XRP complex** (10 names) is negative on average across the window, not
  just on the day it was first flagged.
- **Validation is unstable day to day.** COIN reads +0.98% net today but
  averages +0.06% and passed on only 38% of days.

**Correction to an earlier claim.** A same-day snapshot is mostly noise. The
~$118k year-end figure given earlier in this conversation was built on
today's snapshot of the "1%+ tier" and is retracted.

## 2. Finding outside the five items: the expectancy block latches

`recent_expectancy` blocks entries when the last 20 closed trades average ≤ 0.
Entries are the only way new trades close, so once it trips it can never
reopen. In the pre-committed grid it blocked entries at 787 of ~886 decision
points in the live replica. Most variants show exactly +0.0% in the second
half: they went flat and stayed flat.

The same failure mode is described in the engine's own comment for the
win-rate version it replaced. Live has not latched only because recent
realized trades are net positive. Owner trades count toward that.

The grid was re-run with the block disabled so the items could be compared on
their merits. Both runs are in `results.json` / `results_nolatch.json`.

## 3. Items 1–5 (latch disabled, live universe unless noted)

| Variant | Return | 1st half | 2nd half | @0.2% slip | Max DD | Trades |
|---|---:|---:|---:|---:|---:|---:|
| V0 live replica | +4.4% | +1.9% | +2.0% | +9.0% | −12.6% | 200 |
| **1 — cost-aware ML gate** | +35.0% | +19.9% | +4.9% | +26.7% | −7.3% | 120 |
| **2 — enter validated names only (dynamic)** | **+40.5%** | **+15.1%** | **+19.2%** | **+32.4%** | −10.0% | 171 |
| 2 — static list, honest split¹ | +5.0% | | | | −16.3% | 117 |
| 3 — tier ≥ 0.5% net | +18.2% | | | | −7.0% | 75 |
| 3 — tier ≥ 1.0% net | +4.8% | | | | −8.2% | 41 |
| 4 — margin 1x (no T+1) | +5.7% | | | | −24.3% | 260 |
| 5 — shorts (margin) | +1.5% | | | | −15.4% | 411 |
| 5 — inverse ETFs (cash) | +3.7% | | | | −13.0% | 209 |
| all five stacked | −8.6% | | | | −23.3% | 214 |
| 2x leverage on the stack | −27.2% | | | | −41.1% | 253 |
| **E2 — item 2 on 65 symbols** | **+50.6%** | **+16.1%** | **+25.1%** | **+39.6%** | **−6.1%** | 205 |
| E3 — E2 + 0.5% tier² | +80.9% | +27.0% | +30.5% | +64.0% | −13.3% | 239 |

¹ Selected on Feb 20–Mar 19 and tested after. The live replica made +8.8%
over the same test window, so the static list did worse.
² Picked after seeing the grid. 62% of its gains came from three names: SOXS,
AMD and TSLQ. Two of those are inverse ETFs added for this study.

**Benchmarks, same window, buy-and-hold:** SPY +12.7%, QQQ +23.1%, and an
equal-weight basket of the 33 live symbols +20.0%. The live strategy lagged
all three.

**Verdicts**

- **Works, robustly:** item 1 with item 2 in its dynamic form. Only enter
  names whose cost-aware validation passes that day. It is positive in both
  halves and survives doubled slippage. Its gains are concentrated in the base
  universe (85% from three names). Spread across 65 symbols (E2), it has lower
  drawdown and a top-3 share of 45%.
- **Does not work here:**
  - item 2 as a static list, because validation is too unstable to fix a list;
  - the 1% tier, which is too few trades;
  - item 4, because PDT absorbs the settlement benefit at this account size;
  - item 5 shorts;
  - leverage.
- **Time stop:** inconclusive. It flips sign between halves and under
  slippage. Neither the retirement nor a revert is supported by this data.
- **Stops overshoot.** Polled hourly stops filled at a mean of −7.0% to −7.4%
  against a −5% trigger. That is a large, recurring drag, and the case for
  resting intraday stops is stronger than it looked when that idea was
  shelved.

## 4. Projection to 2026-12-31 (70 trading days, block bootstrap, 20,000 paths)

| Variant | P10 | Median | P90 | Best of 20k | P(loss) | P($1MM) |
|---|---:|---:|---:|---:|---:|---:|
| Live replica | $1,317 | $1,466 | $1,635 | $2,079 | 42% | 0 |
| Item 2 | $1,402 | $1,683 | $2,010 | $2,852 | 14% | 0 |
| **E2** | **$1,510** | **$1,738** | **$2,016** | $2,856 | **4%** | 0 |
| E3 | $1,449 | $1,852 | $2,371 | $3,650 | 9% | 0 |

$1MM by year end needs 9.80%/day. The best variant delivered 0.40%/day. No
path out of 20,000, for any variant, came within a factor of 270 of the
target.

## Limitations

- **One 7-month window, mostly rising.** Long-biased results partly reflect
  the market.
- **Many variants were tested on one window,** so the best row is flattered.
  Both-halves agreement is the main defence.
- **Model and fills are simplified:** daily rather than hourly retrains, and
  0.1%/side slippage with no partial fills.
- **Hindsight in the additions.** They were chosen in 2026 by names that are
  liquid and active in 2026.
- **Short borrow costs** are ignored.

## 5. Implemented (owner instruction 2026-09-22)

The owner said *"implement all three, dormant and reversible with tests."*
Each change is a `bot/config.json` switch, and each switch restores the
pre-change behaviour exactly.

| Change | Switch | Revert |
|---|---|---|
| Expectancy block becomes a cooldown | `strategy.expectancy_block_expiry_days: 3` | `null` = permanent latch |
| Validation charges round-trip cost | `strategy.validation_cost_pct: 0.2` | `0` = gross gate |
| Entries require a validated model | `strategy.entry_requires_validation: true` | `false` |
| 65-symbol entry universe | `universe_expansion_enabled: true` | `false` = 33 core |

**Why 3 days.** The cooldown length was chosen by a rule fixed before
looking: take the longest cooldown that doesn't materially hurt either half
versus no block, on the deployed configuration.

| Block | Full | H1 | H2 | @0.2% | Max DD |
|---|---:|---:|---:|---:|---:|
| none | +50.6% | +16.1% | +25.1% | +39.6% | −6.1% |
| latch (before) | +12.5% | +12.6% | +12.0% | +5.3% | −6.6% |
| **3-day cooldown (deployed)** | **+53.5%** | **+15.1%** | **+30.7%** | **+33.8%** | **−9.2%** |
| 5-day | +31.8% | +14.1% | +31.1% | +18.0% | −6.8% |

2 days scored highest (+63.5%) and was not chosen; picking the best cell is
exactly the overfitting the rule exists to prevent.

**Operational costs.** One live cycle now runs 49 s of engine time instead of
~25 s. The historicals fetch goes from 4 batches to 7.

**Tests.**
- `bot/strategy/test_replication_changes.py`: 18 tests, one pair per change
  (enabled behaviour, and the switch restoring the old one). They also check
  that today's live history does not trip the block.
- The existing suites still pass. `test_quote_pricing` disables the entry
  gate locally, because its synthetic bars cannot validate.
