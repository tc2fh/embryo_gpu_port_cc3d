#!/usr/bin/env python3
"""
no_headless.py — PreToolUse(Bash) guard. Exit 2 BLOCKS the tool call before it runs.

Hard-enforces the "interactive subscription pool only" rule: blocks any attempt to shell out to
headless Claude (`claude -p` / `--print` / Agent SDK CLI), which would bill a different pool. This
is the structural backstop behind the CLAUDE.md instruction — instructions guide, this enforces.
"""
import json
import re
import sys

try:
    event = json.load(sys.stdin)
except Exception:
    sys.exit(0)  # can't parse -> don't block

cmd = (event.get("tool_input", {}) or {}).get("command", "")

# Block headless claude. Allow normal `claude` mentions in comments/echo by requiring the flag form.
BLOCKED = [
    r"\bclaude\b[^\n|&;]*\s-p\b",
    r"\bclaude\b[^\n|&;]*--print\b",
    r"\bclaude-agent-sdk\b",
    r"\bANTHROPIC_API_KEY=",          # setting a key inline is the other way to misroute billing
]

for pat in BLOCKED:
    if re.search(pat, cmd):
        sys.stderr.write(
            "BLOCKED: headless/API-billed Claude invocation is not allowed in this workflow. "
            "This run must stay in the interactive session to bill the subscription pool. "
            "Do the work via a subagent (Task tool) instead.\n"
        )
        sys.exit(2)  # exit 2 = block, message is fed back to Claude

sys.exit(0)
