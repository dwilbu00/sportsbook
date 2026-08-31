---
name: effort-and-cost
description: Doug wants cost efficiency — flag when a prompt is lower-effort than the current tier, and delegate mechanical subtasks to cheaper models
metadata:
  type: feedback
---

Doug wants to save usage cost on less-demanding tasks. I cannot change my own
session reasoning-effort mid-turn (it's harness/`/config`-controlled, plus `/fast`),
so instead:

1. **Flag cheaper prompts.** When a request is clearly lower-effort than the current
   tier, prefix the reply with a one-line note (e.g. "LOW-effort task — you could
   `/fast` or drop to Haiku"). No flag = the current tier is warranted.
2. **Delegate mechanical subtasks to cheaper models.** Route grep sweeps, single-file
   lookups, boilerplate edits, "what does X do" to `claude-haiku-4-5` (Agent tool
   alias `haiku`) or `claude-sonnet-4-6` (pass the full ID, NOT the `sonnet` alias —
   that resolves to 4-5 which isn't on Doug's Vertex). See [[data-and-architecture]]
   for the model-routing gotcha.

3. **Ultracode = OFF by default** (Doug's call, 2026-08-31). It runs a multi-agent
   workflow for every substantive task, token cost no object — overkill for our
   typical single-file-fix / run-backtest / interpret-results loop. Flip it on
   per-task (type `ultracode`) only for jobs with real breadth or that need
   independent verification: adversarial audits (e.g. the coherence-commit audit),
   codebase-wide sweeps/migrations, wide-open design (judge panels). Proactively say
   "this one's worth ultracode" when a task fits.

**Effort rubric:**
- LOW (Haiku / `/fast`): mechanical/deterministic — run a known command, flag lookup,
  rename/format edit, status check, read one file.
- MEDIUM (Opus normal): scoped multi-file work following an existing pattern — add a
  backtest scenario, localized bug fix, wire a flag through.
- HIGH (Opus high): novel judgment where a wrong call is expensive/irreversible —
  modeling/stat decisions, adversarial audits, calibration promotion, edge validation,
  credit/money spends, ambiguous scope.

**Why:** cost control without sacrificing quality on the calls that matter.
**How to apply:** proactively flag + auto-delegate; don't make Doug ask each time.
