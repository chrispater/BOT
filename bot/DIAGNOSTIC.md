# Owner-authored diagnostic procedure (one-shot)

This file was committed to the branch by the owner's configuration session on
2026-07-14. Its presence on the branch IS the owner's authorization: a cycle
session that finds this file executes this checklist INSTEAD of a trading
cycle, at any time of day (the market-hours gate does not apply — nothing here
trades). Runbook §0.2a covers this mechanism.

Purpose: Monday 2026-07-13's seven market-hours firings produced zero commits.
Determine which link broke: Robinhood MCP auth, the Python environment, or
pushing to the branch.

Steps — record each outcome; never trade:

1. `git pull origin claude/robinhood-trading-mcp-iiup95` (you have this file,
   so this already worked).
2. Robinhood MCP auth probe (READ-ONLY): call `get_accounts`. Record "ok" or
   the exact error text. Do not call any other Robinhood tool.
3. Python environment: `pip install -q -r bot/requirements.txt`, then
   `python -c "import bot.strategy.run_cycle"` from the repo root. Record
   "ok" or the exact error.
4. Append one JSON line to `bot/trade_log.jsonl`:
   `{"ts": "<UTC now>", "event": "diagnostic", "mcp_auth": "ok"|"<error>",
     "python_env": "ok"|"<error>"}`
5. `git rm bot/DIAGNOSTIC.md` (this file is one-shot; future cycles must run
   normally), commit both changes with message `bot: diagnostic`, and
   `git push -u origin claude/robinhood-trading-mcp-iiup95`. If the push is
   rejected, quote the exact git error verbatim in your final summary — a
   push failure is itself the answer we're looking for.
6. End your turn with a summary of all recorded outcomes.
