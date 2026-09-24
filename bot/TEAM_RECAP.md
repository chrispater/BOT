# Team recap — the morning scoreboard toward $1MM

Owner request, 2026-09-24: "we're a team after all.. give a morning recap of how
we're averaging towards our $1MM goal!" The bot trades stocks in Agentic
(••6733); the owner trades crypto by hand in their default individual account
(••3416). This procedure only READS both accounts. It never reviews, previews,
places or cancels an order in either one, and it changes no trading setting.

It runs from the daily "team-recap-morning" trigger at 12:30 UTC (08:30 ET in
daylight time, 07:30 ET after Nov 1), every day including weekends, because
crypto trades around the clock.

## Privacy

This repo is public. `bot/team/` (the scoreboard) is gitignored and is never
committed, and the ••3416 account number is never written into the repo: look
it up each time with `get_accounts` (the account whose number ends 3416). In
user-facing text, mask account numbers to their last 4 digits.

## 1. Gates
1. `git pull origin claude/robinhood-trading-mcp-iiup95`.
2. If the Robinhood tools are missing or unauthorized, say so in one line and stop.
3. One snapshot per ET day. If the last record in `bot/team/scoreboard.jsonl` is
   already dated today, skip to step 3 and only print the report.

## 2. Read both accounts (read-only) and snapshot
- `get_accounts` → the accounts ending 6733 (key `agentic`, label
  "Agentic ••6733") and 3416 (key `investing`, label "Investing ••3416").
- Both accounts: `get_portfolio` → `total_value` and `cash`;
  `get_equity_positions` → quantity per symbol; `get_equity_quotes` → price
  (the newer of `last_trade_price` / `last_non_reg_trade_price`).
- Investing: `get_crypto_positions` → `quantity` per coin; `get_crypto_quotes`
  with `<CODE>-USD` → `mark_price`. Holdings key is the coin code.
- Fills since the previous record's `ts_utc`: `get_crypto_orders`
  (`state=filled`, `updated_at_gte` = that time) and `get_equity_orders`
  (`state=filled`, `created_at_gte` = that date) for both accounts. Keep only
  orders filled after the previous `ts_utc`. For each, write
  `{symbol, side, qty, notional}`, where notional is the total cash that changed
  hands, fees included. The math relies on this to tell a trade from a deposit.
- `get_realized_pnl` with `span=week` and `span=month` → `total_returns`, as
  `realized: {"7d", "30d"}`.
- Write the spec with the Write tool to `team_spec.json` in the scratchpad
  (format in `bot/tools/team_recap.py`). Carry `other_accounts_usd` forward from
  the last record unless the owner has given a new figure.
- `python -m bot.tools.team_recap snapshot <spec>`.

## 3. Report
- `python -m bot.tools.team_recap report`, then post it with at most two
  sentences of plain-English context. Deposits and transfers are never
  performance, and no average or ETA is claimed before 3 days are tracked.
- Send one PushNotification with the headline: team value, % of $1MM, and the
  change since the last recap.
- Nothing to commit. The scoreboard stays out of this public repo; code or
  runbook changes are committed as usual.

If the container was reset overnight and `bot/team/scoreboard.jsonl` is gone,
today's snapshot becomes a new baseline. Say so in the report.
