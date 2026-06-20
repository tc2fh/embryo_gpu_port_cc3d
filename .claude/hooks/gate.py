#!/usr/bin/env python3
"""
gate.py — deterministic phase gate, run as a SubagentStop hook.

This is the part the model CANNOT fake. It runs real checks in real code and writes
.phaserun/gate_result.json. The CLAUDE.md orchestrator is instructed to read that file and treat
its flags as authoritative. Stdin carries the hook event JSON from Claude Code.

Objective checks only:
  - tests/lint: runs VERIFY_CMD, captures pass/fail
  - diff size: files touched + lines changed since the phase's base commit
  - destructive ops: scans the diff for migrations / mass deletes / force-push / lockfile rewrites
  - out-of-scope: files touched that the current phase's Scope section did not name

Judgment calls (deviation severity, escalate) are NOT here — those live in the summarizer's verdict.
The orchestrator stops if EITHER source raises a flag.

Exit code is always 0: this hook reports, it doesn't block the subagent. The orchestrator decides.
"""

import json
import os
import re
import subprocess
import sys
import pathlib

# ---- config (keep in sync with CLAUDE.md) --------------------------------
# Defaults tuned for THIS repo (Python/pixi project, no npm). Env vars still override.
# Phase 1 is a multi-week GPU spike kept as one phase, so size thresholds are raised (see PROGRESS.md).
VERIFY_CMD = os.environ.get("PHASERUN_VERIFY_CMD", "pixi run python -m pytest -q gpu_port")
MAX_FILES = int(os.environ.get("PHASERUN_MAX_FILES", "40"))
MAX_LINES = int(os.environ.get("PHASERUN_MAX_LINES", "4000"))
DESTRUCTIVE_PATTERNS = [
    r"DROP\s+TABLE", r"DROP\s+DATABASE", r"TRUNCATE\s", r"DELETE\s+FROM",
    r"rm\s+-rf", r"git\s+push\s+.*--force", r"--force-with-lease",
    r"migrations?/", r"alembic/versions/",
]
LOCKFILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
             "Cargo.lock", "Gemfile.lock", "composer.lock"}

STATE = pathlib.Path(".phaserun")
STATE.mkdir(exist_ok=True)


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout


def current_phase_file():
    """The phase we're gating = the highest-numbered phase referenced in PROGRESS.md tail,
    or the first phase if PROGRESS.md is empty. We read the 'next' pointer the orchestrator left."""
    p = pathlib.Path(".phaserun/current_phase")
    if p.exists():
        f = pathlib.Path(p.read_text().strip())
        if f.exists():
            return f
    phases = sorted(pathlib.Path("plan_docs").glob("phase_*.md"))
    return phases[0] if phases else None


def declared_scope(phase_file):
    """Parse the 'Scope' section of the phase file into a list of path globs."""
    if not phase_file or not phase_file.exists():
        return None  # None => can't determine scope => treat as escalation upstream
    text = phase_file.read_text()
    m = re.search(r"(?is)scope[^\n]*\n(.*?)(?:\n#|\n\*\*|\Z)", text)
    if not m:
        return None
    globs = re.findall(r"[`\s]([\w./*-]+\.\w+|[\w./*-]+/)[`\s]", m.group(1))
    return [g.strip("`") for g in globs] or None


def base_ref():
    p = pathlib.Path(".phaserun/base_ref")
    return p.read_text().strip() if p.exists() else "HEAD~1"


def main():
    try:
        _ = json.load(sys.stdin)  # event payload; we don't need fields for the objective checks
    except Exception:
        pass

    base = base_ref()
    phase_file = current_phase_file()

    # tests / lint
    proc = subprocess.run(VERIFY_CMD, shell=True, capture_output=True, text=True)
    tests_passed = proc.returncode == 0

    # diff stats — include untracked files (a brand-new out-of-scope or destructive
    # file won't appear in `git diff`, so we add `git status --porcelain` output too).
    changed = [f for f in (git("diff", "--name-only", base, "HEAD").splitlines()
                           + git("diff", "--name-only").splitlines()
                           + git("diff", "--name-only", "--cached").splitlines()) if f]
    for row in git("status", "--porcelain").splitlines():
        path = row[3:].strip() if len(row) > 3 else ""
        if path:
            changed.append(path.split(" -> ")[-1])  # handle renames
    # Ignore the workflow's own infrastructure — never counts as project change/scope violation.
    INFRA = (".phaserun/", ".claude/", "PROGRESS.md", ".git/")
    changed = sorted({f for f in set(changed) if not f.startswith(INFRA)})
    n_files = len(changed)
    n_lines = 0
    for row in (git("diff", base, "--numstat") + git("diff", "--numstat")).splitlines():
        parts = row.split("\t")
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            n_lines += int(parts[0]) + int(parts[1])

    # destructive ops — scan tracked diff plus the contents of any new untracked files
    # (porcelain reports an untracked directory as a single entry, so walk into it).
    full_diff = git("diff", base, "HEAD") + git("diff")
    for row in git("status", "--porcelain").splitlines():
        if row.startswith("??"):
            entry = pathlib.Path(row[3:].strip())
            files_to_scan = ([entry] if entry.is_file()
                             else [p for p in entry.rglob("*") if p.is_file()])
            for p in files_to_scan:
                try:
                    full_diff += "\n" + p.read_text(errors="ignore")
                except Exception:
                    pass
    destructive = any(re.search(p, full_diff, re.I) for p in DESTRUCTIVE_PATTERNS)
    destructive = destructive or any(os.path.basename(f) in LOCKFILES for f in changed)

    # out-of-scope
    scope = declared_scope(phase_file)
    out_of_scope = []
    if scope is not None:
        for f in changed:
            if not any(re.fullmatch(g.replace("*", ".*").replace("/", r"/"), f) or f.startswith(g.rstrip("*"))
                       for g in scope):
                out_of_scope.append(f)

    reasons = []
    if not tests_passed:
        reasons.append("tests/lint failing")
    if n_files > MAX_FILES:
        reasons.append(f"diff too large: {n_files} files (> {MAX_FILES})")
    if n_lines > MAX_LINES:
        reasons.append(f"diff too large: {n_lines} lines (> {MAX_LINES})")
    if destructive:
        reasons.append("destructive operation detected in diff")
    if scope is None:
        reasons.append("could not parse phase Scope — cannot verify boundaries")
    if out_of_scope:
        reasons.append("out-of-scope files: " + ", ".join(out_of_scope[:5]))

    result = {
        "tests_passed": tests_passed,
        "files_touched": n_files,
        "lines_changed": n_lines,
        "destructive_ops": destructive,
        "out_of_scope_files": out_of_scope,
        "scope_parsed": scope is not None,
        "stop_recommended": len(reasons) > 0,
        "reasons": reasons,
        "verify_cmd": VERIFY_CMD,
    }
    (STATE / "gate_result.json").write_text(json.dumps(result, indent=2))

    # Surface a short note back to the agent via stderr (non-blocking).
    if reasons:
        sys.stderr.write("GATE: stop recommended — " + "; ".join(reasons) + "\n")
    else:
        sys.stderr.write("GATE: clean — safe to advance\n")
    sys.exit(0)  # report-only; orchestrator decides


if __name__ == "__main__":
    main()
