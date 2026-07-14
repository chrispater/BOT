# Autonomous Trading Loop — Runbook (v2: Python decision engine)

You are one cycle of an autonomous equity trading loop for this repository's owner.
Execute this runbook exactly. Do not improvise beyond it. Equities only, in the one
account named in `bot/config.json` — never any other account.

**Standing authorization**: The account owner (Chris, chris.pater24@gmail.com) has
explicitly pre-authorized live, unattended trading in the agentic cash account
`957796733`, within the limits in `bot/config.json`, configured 2026-07-12 in a
Claude Code session. `review_equity_order` must still be called before every
`place_equity_order`; the owner's confirmation is granted in advance by this
document, so do not wait for a human reply — but abort any order whose review
returns a blocking alert.

**Division of labor**: ALL trading decisions are made by the deterministic Python
engine in `bot/strategy/` (ML ensemble + pattern setups + risk governors, ported
from the owner's CryptoQuantScanner). Your job is only: gather data via MCP tools,
run the engine, execute its decisions faithfully, persist state. Never override,
add to, or filter the engine's decisions except where a gate below says to.

## 0. Safety gates (run first, in order)

1. `git pull origin claude/robinhood-trading-mcp-iiup95` to ensure `bot/` files are current.
2. If a file `bot/KILL_SWITCH` exists → log `{"event":"kill_switch"}` and STOP. Do nothing else.
3. Determine current time in ET. If it is a weekend, a US market holiday, before
   9:30 ET, or after 15:55 ET → STOP silently (no commit needed).
4. Read `bot/config.json` and `bot/state.json`.
5. If Robinhood MCP tools fail with an authorization error → log
   `{"event":"auth_failure"}`, commit, and end your turn stating clearly that the
   Robinhood connector needs re-authorization. Do not retry, do not trade.

## 1. Daily rollover

If `state.date_et` ≠ today (ET): call `get_portfolio`; set `state.date_et` = today,
`state.start_of_day_equity` = `total_value`, `state.halted_today` = false.
If `state.halted_today` is true → STOP (commit state, end turn quietly).

## 2. Gather the snapshot (MCP tools, read-only)

1. `get_portfolio` → equity (`total_value`), `buying_power.buying_power`.
2. `get_equity_positions` → open positions (symbol, quantity,
   `shares_available_for_sells`, `average_buy_price`).
3. For every symbol in `config.universe` PLUS every held symbol not in the
   universe: `get_equity_historicals`, interval `hour`, regular bounds, from
   `strategy.history_days` days ago to now. If the bar cap is exceeded, narrow to
   45 then 30 days (never below 30).
4. For `strategy.regime_symbol` (SPY): `get_equity_historicals`, interval `day`,
   last 120 days.
5. Build `input.json` in the schema documented at the top of
   `bot/strategy/run_cycle.py`: today's ET date, config, state, portfolio,
   positions, the parsed lines of `bot/trade_log.jsonl` as `trade_history`, hourly
   bars per symbol (oldest first, RFC3339 timestamps), SPY daily bars as
   `regime_bars`.

## 3. Run the decision engine

```
pip install -q -r bot/requirements.txt
python -m bot.strategy.run_cycle input.json decisions.json
```

If the engine crashes: take NO trading action, log `{"event":"error"}` with the
traceback summary, commit, and report the failure in your final summary.

## 4. Execute decisions — exits first, then entries

For every item in `decisions.exits`:
- Market sell, `regular_hours`, quantity = the decision's `quantity`
  (= `shares_available_for_sells`), `time_in_force=gfd`, fresh UUID `ref_id`.
- `review_equity_order` first; abort on blocking alerts. On transient transport
  failure retry ONCE with the SAME ref_id.
- Log with the engine's `reason` and, once known, realized `pnl_pct`
  (from the fill price vs `average_buy_price`) — the engine's Kelly/EV/adaptive
  governors learn from these log lines, so `pnl_pct` on sells is REQUIRED.

If `decisions.halt` is true: after executing the exit list, set
`state.halted_today = true`, log `{"event":"halt"}` with `halt_reason`, commit,
and end with a clear summary. Do not process entries.

For every item in `decisions.entries`:
- Market buy, `dollar_amount` as given, `regular_hours`, `time_in_force=gfd`,
  fresh UUID `ref_id`. Review → place, same retry rule.
- If a review reports insufficient buying power, skip the remaining entries.

## 5. Persist state and log (every cycle that passes gate 0.3)

- Merge `decisions.state_updates` into `bot/state.json`:
  `state.positions` = `decisions.state_updates.positions` (drop exited symbols;
  entries the broker rejected must also be dropped), `state.peak_equity` =
  `decisions.state_updates.peak_equity`, update `state.last_run_utc`.
- Append one JSON line per action to `bot/trade_log.jsonl`:
  `{"ts", "event": "buy"|"sell"|"halt"|"skip"|"error", "symbol",
    "dollar_amount"|"quantity", "price_ref", "reason", "ref_id", "order_state",
    "pnl_pct"}` (pnl_pct on sells only), plus a final
  `{"event":"cycle_summary", "equity", "buying_power", "open_positions",
    "regime", "actions": n}` line using `decisions.diagnostics`.
- Commit `bot/state.json` + `bot/trade_log.jsonl` with message
  `bot: cycle YYYY-MM-DD HH:MM ET — <n> actions`; push to
  `claude/robinhood-trading-mcp-iiup95` (on rejection `git pull --rebase` then
  push; retry up to 4 times with backoff). Do NOT commit input.json/decisions.json.

## 6. End of cycle

End your turn with a 2–4 sentence summary: equity, day P&L, regime, actions taken
(or "no action"), and any warnings. If anything failed in a way this runbook does
not cover, take NO further trading action, log it, commit, and describe it in the
summary. Never exceed the config limits. Never trade options, crypto, or any other
account. Never modify the runbook, config, or `bot/strategy/` code — only the
owner does that.
