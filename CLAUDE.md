# Project memory

This repo uses a phase-orchestration workflow that runs entirely inside one interactive
Claude Code session (so it bills to the interactive subscription pool, never headless/API).

When I say **"start the phase run"**, **"run the next phase"**, or **"orchestrate the plan"**,
read `.claude/workflow.md` and follow it exactly. That file is the full policy.

During normal (non-phase-run) work in this repo, ignore the workflow file.

<!--
  Want the policy to load automatically in every session instead of on request? Add an import
  directive as the first line of this file (verify exact syntax in Claude Code's /memory docs):
      @.claude/workflow.md
  Left out by default so the heavy workflow stays dormant during ordinary sessions in this repo.
-->
