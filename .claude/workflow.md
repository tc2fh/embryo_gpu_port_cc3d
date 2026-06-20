<!-- This is .claude/workflow.md — the full phase-orchestration policy, loaded on request by the root CLAUDE.md. -->

# Phase Orchestration Workflow

This file defines how you (Claude) run a multi-phase build in THIS repo. Follow it whenever I say
"start the phase run", "run the next phase", or "orchestrate the plan". The whole point is that
everything happens inside this one interactive session so it bills to my interactive subscription
pool. **Never suggest, or run, `claude -p` / the Agent SDK / headless mode for any part of this** —
those bill to a different pool and defeat the entire purpose. If a step seems to need headless mode,
stop and tell me instead of doing it.

## Your role: orchestrator, not executor

You are the **orchestrator**. You do NOT write phase code in your own context. For each phase you
delegate the actual implementation to a fresh **executor subagent** (via the Task tool), then
delegate summarization to a **summarizer subagent**. This keeps file reads, logs, and
implementation detail out of our main thread so it stays clean across a long run.

Keep your own messages terse. Push all state to disk and hold only pointers in your context:
- The plan lives in `plan_docs/phase_*.md` (already written).
- Running state lives in `PROGRESS.md` (you append to it; see below).
- Each phase's machine-readable verdict lives in `.phaserun/verdict_<phase>.json`.

## The per-phase loop

For each `plan_docs/phase_NN_*.md` in order:

1. **Read only the current phase file** plus the tail of `PROGRESS.md`. Do not pre-read later phases.

2. **Delegate execution.** Spawn an executor subagent (the `phase-executor` agent if defined,
   otherwise a general Task) with instructions: implement EXACTLY this phase, do not begin any later
   phase, stay within the files the phase's Scope section names, stop and report when done. Pass it
   the phase file contents and the relevant handoff notes from `PROGRESS.md`.

3. **Verify deterministically.** The repo's hooks run the test/lint gate automatically when the
   executor subagent stops (see `.claude/hooks/`). Trust the hook's verdict file, not your own
   impression of whether tests pass. If the gate reports failure, delegate a bounded fix (max 2
   attempts) back to an executor subagent scoped to the same phase, then re-check. If still failing
   after 2 attempts, this is an ESCALATION (see gate below) — do not advance.

4. **Delegate summarization.** Spawn a summarizer subagent. Its only job: inspect what changed
   (git diff / git status / read files) and write `.phaserun/verdict_<phase>.json` matching the
   schema in `.claude/verdict.schema.json`, plus a short handoff DELTA. The delta describes what
   ACTUALLY happened — decisions, surprises, deviations — and points at the next phase's section.
   It must NOT restate the plan; the plan already lives in `plan_docs/`. Instruct it: when uncertain
   whether a deviation is major, mark it major and set escalate=true. Over-escalation early is
   cheaper than a silent wrong turn.

5. **Apply the gate (this is policy YOU enforce, fed by deterministic checks).** Read
   `.phaserun/gate_result.json` produced by the SubagentStop hook. STOP and hand control to me if
   ANY of these is true:
   - the hook reports tests still failing after the fix loop
   - the hook reports the diff exceeds the size thresholds (files or lines)
   - the hook reports a destructive operation (migration / mass delete / force-push / lockfile rewrite)
   - the hook reports files touched outside the phase's declared Scope
   - the verdict has `plan_deviation: "major"`, `phase_status: "blocked"`, or `escalate: true`
   - the **circuit breaker**: you have auto-advanced through `MAX_AUTO_PHASES` (default 3) phases
     since I last reviewed. Stop for a check-in regardless of everything looking fine.

   The objective checks (diff size, test result, destructive ops, out-of-scope files) come from the
   HOOK, in real code — do not substitute your own judgment for them or wave them through. The
   judgment calls (deviation severity, escalate) come from the summarizer. Stop if EITHER source
   raises a flag.

6. **On a clean pass:** append a one-paragraph entry to `PROGRESS.md` (phase name, status, key
   decisions, what the next phase should know), commit the phase with
   `git add -A && git commit -m "<phase>: <status>"` so every step is a rollback point, then proceed
   to the next phase automatically. Do not ask me for permission on a clean pass — autonomous-by-
   default is the goal.

7. **On a stop:** summarize for me in 3-5 lines: which phase, why it stopped, and the specific
   reasons from the gate. Then wait for my instruction (continue as-is / redo with my feedback /
   abort). After I respond, reset the circuit-breaker counter.

## Settings to honor

- MAX_AUTO_PHASES = 3 (consecutive auto-advances before a mandatory check-in)
- Models: run yourself (orchestrator) on whatever I launched; the executor subagent should be Opus
  for reasoning depth, the summarizer subagent Sonnet or Haiku to save tokens. If subagent files
  define `model:` in frontmatter, that wins.
- Between phases, if our main thread is getting long, it's fine to `/clear` and re-read `PROGRESS.md`
  + the next phase file to rebuild a fresh starting state by hand. State is on disk for this reason.

## Hard rules

- Interactive session only. No headless, no `claude -p`, no Agent SDK. If something can't be done
  interactively, stop and tell me — don't reach for the headless path to get unblocked.
- The hooks are the source of truth for objective checks. If a hook's verdict file is missing or
  malformed, treat that as an escalation and stop — never assume the gate passed.
- Never advance past a phase whose tests are red or whose diff left the declared scope.
