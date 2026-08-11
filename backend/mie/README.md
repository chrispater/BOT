# Market Intelligence Engine

A research-first decision layer for the bot: instead of an ML model that says
`BUY` / `SELL` / `HOLD`, this records market state continuously, waits for the
future to actually happen, and only lets a model influence a live trade after
it has survived out-of-sample validation.

## Why this exists

The old pipeline (`backend/trading_service.py`'s ML model + setup detectors)
trains a classifier to predict `signal ∈ {-1, 0, 1}` from a fixed indicator
set, using a threshold someone picked to decide what counts as a labeled
"buy". That's a reasonable system and it stays in place — this doesn't replace
it. It answers a different, narrower question:

> Under these exact conditions, after fees and slippage, does this specific
> setup have positive expectancy over the next N minutes — and how do we know,
> from evidence, that it does?

Everything in this package is built around three commitments:

1. **Record state without a label.** `feature_engine.py` builds a market
   observation; nothing decides at that moment whether it's a "good" one.
   Outcomes (`outcomes.py`) — forward return, max favorable/adverse excursion
   at several horizons — get attached later, once the future has actually
   played out. See `state.py` for the full vocabulary.

2. **Nothing reaches a trade without surviving unseen data.**
   `validation.py` implements purged walk-forward cross-validation: chronological
   folds, an embargo gap sized to the label horizon so no training sample's
   forward-return window leaks into the test period, and a pass/fail verdict
   that requires positive expectancy pooled across multiple folds — not one
   lucky window. `edge_model.py` is the model that has to clear this bar
   before `engine.py` will ever act on its output.

3. **`DO_NOTHING` is the default, not a fallback.** `state.TradeDecision`
   defaults to `DO_NOTHING`. `engine.MarketIntelligenceEngine.decide()` is
   structured as a series of gates a candidate has to clear on the way *out*
   of that default — hostile regime, unvalidated model, insufficient sample,
   costs eating too much of the edge, quality below threshold — never as
   logic that has to justify staying flat.

## Pipeline

```
MARKET DATA → FeatureEngine → RegimeModel → EdgeModel (per horizon)
            → cost model → MarketIntelligenceEngine.decide() → TradeDecision
            → ObservationStore (records + resolves outcomes)
            → analysis.run_post_mortem() once decisions close
```

| Module | Responsibility |
|---|---|
| `state.py` | Shared vocabulary — `MarketState`, `TradeDecision`, `EdgeForecast`, regime taxonomy, feature-group prefixes. |
| `feature_engine.py` | Price structure, regime descriptors, session, cross-market, order-book/tape, derivatives. No TA-Lib dependency; no lookahead. |
| `outcomes.py` | Forward return / MFE / MAE per horizon, from raw OHLCV, using only future bars. |
| `store.py` | SQLite-backed observation log. Records unlabeled, backfills outcomes as they mature, answers "have I seen something like this before" via a standardized nearest-neighbor query. |
| `regime.py` | Rule-based classifier over the continuous regime features: `trend_up/down`, `mean_revert`, `compression`, `expansion`, `panic`, `thin`. Panic/thin are hostile regimes the engine refuses to size into regardless of model output. |
| `validation.py` | Purged walk-forward CV + the pass/fail rule for "is this edge real". |
| `edge_model.py` | Per-horizon HistGradientBoosting models (classifier + 3 regressors) predicting prob-positive / expected return / MFE / MAE for both LONG and SHORT (SHORT is a sign-flip of the same fit, not a separately-trained model). |
| `costs.py` | Spread/slippage/fee estimate from a live order book, or a volatility-scaled fallback when no book is available. |
| `engine.py` | Orchestrator. `fit()` trains+validates every horizon; `decide()` produces one `TradeDecision`, `DO_NOTHING` unless every gate clears. |
| `analysis.py` | Post-mortem: which regime/quality-bucket/feature-quartile actually carried the P&L on closed trades, and what pruning would have improved expectancy. |

## Feature groups

Every feature name is prefixed by group (`ps_` price structure, `of_` order
flow, `dv_` derivatives, `xm_` cross-market, `rg_` regime, `ex_` execution,
`tm_` session). `of_`, `dv_`, and `ex_` slippage features only exist when a
live order book / trade tape / derivatives feed is supplied — they are simply
absent (not zero-filled) on historical-only data. `edge_model.py`'s
`select_trainable_features()` is the coverage gate that keeps a model from
training on a column that's mostly NaN; as the store accumulates live
snapshots, those columns clear the coverage bar automatically and start
contributing, no code change required.

## Integration

`backend/trading_service.py` wires this in as **opt-in and additive**:

- Every cycle, `TradingService._mie_cycle()` records the current state,
  backfills matured outcomes, and periodically (`_maybe_refit_mie`, every 6h
  once ≥500 resolved observations exist) re-fits and re-validates every
  horizon's edge model.
- `mie_gate_enabled` (per-user DB column, default `False`) lets a validated
  `DO_NOTHING` reading veto a fresh entry the legacy ML/setup pipeline wants to
  take. It can only narrow trading, never force a trade, and only exercises
  the veto once it has an actual validated forecast to point to — a cold-start
  engine with no history stays silent rather than blocking everything.
- `TradingService.get_status()` exposes `mie_available`, `mie_gate_enabled`,
  `mie_any_validated`, `mie_resolved_observations`, and the latest decision per
  symbol for the UI/API.

## Tests

`backend/mie/tests/` (`pytest backend/mie/tests/`) covers:
- no-lookahead as a checkable property (a feature for bar *i* must be
  identical whether computed from a frame ending at *i* or a longer one),
- outcome resolution against hand-computed values,
- the observation store's idempotency, backfill, and neighbor search,
- purged walk-forward split geometry (embargo enforced, chronological, no
  overlap),
- the validation gate itself: it must reject a diluted/absent relationship
  *and* accept a real, horizon-matched, persistent one — checked against the
  actual `EdgeModel.fit()` path, not a re-implementation of the logic under
  test,
- the regime classifier's hostile-regime overrides and its handling of
  genuinely missing (vs. present-but-flat) feature data,
- the engine's default-to-`DO_NOTHING` behavior, hostile-regime abstention,
  and bounded quality/sizing output.

## What this deliberately does not do

It does not predict the next candle. It does not replace discretion with an
LLM call — `analysis.py`'s findings are plain groupbys over closed trades,
traceable to a specific computation, not an unfalsifiable "AI insight". And it
does not promise an edge exists: on a fresh install, with no accumulated
history, every `decide()` call correctly returns `DO_NOTHING` — that is the
system working as intended, not a bug to route around.
