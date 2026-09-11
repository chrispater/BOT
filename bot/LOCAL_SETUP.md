# Running the bot locally (persistent Robinhood auth)

The cloud/scheduled-session setup cannot hold the Robinhood MCP OAuth: the
container recycles roughly hourly and the connector detaches every time, so
the loop can only trade in the rare cycle where a live `/mcp reconnect all`
lands. Running the identical bot from your own machine fixes this — the CLI
persists and auto-refreshes the OAuth token, so the connector stays attached
across cycles. Nothing about the strategy, config, or audit trail changes.

## One-time setup (~15 min)

1. Install the Claude Code CLI on a machine that is on during US market hours
   (any Mac/PC/Linux box; a cheap always-on mini-PC or a cloud VM you control
   both work). See https://code.claude.com/docs for the installer.

2. Clone the repo and check out the bot branch:
   ```
   git clone https://github.com/chrispater/BOT.git
   cd BOT
   git checkout claude/robinhood-trading-mcp-iiup95
   ```

3. Python deps for the decision engine:
   ```
   python3 -m pip install -r bot/requirements.txt
   ```

4. Start Claude Code in the repo and authorize Robinhood ONCE:
   ```
   claude
   ```
   then inside the session:
   ```
   /mcp
   ```
   Authorize `robinhood-trading` when prompted (the same OAuth consent you did
   in the browser). Locally this token is cached and auto-refreshed, so you do
   this once, not every hour. Confirm it worked by asking Claude to run
   `get_accounts` — it should return your accounts without an auth error.

## Running the loop

Two options, pick one:

- **Interactive loop** (simplest): in the running `claude` session, use the
  `/loop` skill to fire the runbook hourly during market hours, e.g.
  `/loop 1h execute one full cycle of bot/AGENT_LOOP.md`. Leave the terminal
  open through the session.

- **Cron / Task Scheduler** (hands-off): schedule `claude -p` (headless prompt
  mode) at :30 past each hour 9–16 ET on weekdays, passing the same one-cycle
  instruction. The token cached in step 4 is reused, so no re-auth per run.
  (Windows: Task Scheduler; macOS/Linux: cron or a launchd/systemd timer.)

Either way each cycle does exactly what it does now: pull the branch, run the
gates, snapshot via MCP, run `python -m bot.strategy.run_cycle`, place decisions
with review-before-place, persist `bot/state.json` + `bot/trade_log.jsonl`, and
push to `claude/robinhood-trading-mcp-iiup95`. The same commits land on the same
branch and PR #1, so the record is continuous whether a cycle ran in the cloud
or locally.

## Guardrails still apply

- `bot/config.json` limits (25%/position, 10% daily halt, max 8 positions) are
  unchanged and enforced by the engine regardless of where it runs.
- `bot/KILL_SWITCH`: commit an empty file by that name to halt all trading on
  the next cycle, from anywhere.
- Do not run the cloud Routine and a local loop against the SAME account at the
  same time — two schedulers on one book can double-act. When you switch to
  local, pause/delete the cloud `trading-loop-hourly-selfbind` Routine (ask
  Claude, or do it from the Routines UI) so only one driver is live.

## Why this is the fix, briefly

Robinhood's agentic connector is OAuth-only. OAuth tokens live with the client
that authorized them; a fresh cloud container each hour is effectively a new
client with no token, hence the endless re-auth. A persistent local client
authorizes once and refreshes silently — the exact model OAuth is designed for.
