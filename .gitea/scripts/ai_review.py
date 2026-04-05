#!/usr/bin/env python3
"""AI peer review script.

Routes to Claude Agent SDK (for Codex-authored PRs) or Codex CLI
(for Claude-authored PRs) based on the REVIEWER env var.

Usage:
  Normal:   python ai_review.py
  Dry run:  python ai_review.py --dry-run
"""
import asyncio
import os
import subprocess
import sys

import httpx

DRY_RUN = "--dry-run" in sys.argv

base_sha = os.environ["BASE_SHA"]
head_sha = os.environ["HEAD_SHA"]
reviewer = os.environ.get("REVIEWER", "claude")
reason = os.environ.get("REVIEWER_REASON", "")

# ── Gather diff ───────────────────────────────────────────────────────────────

diff = subprocess.check_output(["git", "diff", base_sha, head_sha], text=True)
changed = subprocess.check_output(
    ["git", "diff", "--name-only", base_sha, head_sha], text=True
).strip()

MAX_DIFF = 80_000
truncated = len(diff) > MAX_DIFF
if truncated:
    diff = diff[:MAX_DIFF] + "\n\n[diff truncated at 80K chars]"

if not diff.strip():
    print("Empty diff — nothing to review.")
    sys.exit(0)

PROMPT = f"""You are peer-reviewing code changes made by an AI coding agent. \
Be thorough and constructively critical. You have access to the full repository — \
use it to understand context (imports, callers, tests) before drawing conclusions.

Changed files:
{changed}

Diff:
{diff}

Structure your review as:

## Summary
Brief overview of what changed and why.

## Issues
- 🚨 CRITICAL: security holes, data loss, broken logic, race conditions
- ⚠️ WARNING: bugs, bad patterns, missing error handling, test gaps
- 💡 SUGGESTION: improvements, simplifications, naming

If there are no issues in a category, omit it.

## Verdict
One of: APPROVED | NEEDS_WORK | CRITICAL_ISSUES
"""

# ── Run reviewer ──────────────────────────────────────────────────────────────

review = ""

if reviewer == "claude":
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    async def run_claude() -> str:
        async for msg in query(
            prompt=PROMPT,
            options=ClaudeAgentOptions(
                cwd=os.getcwd(),
                allowed_tools=["Read", "Glob", "Grep"],
                permission_mode="default",
                max_turns=20,
            ),
        ):
            if isinstance(msg, ResultMessage):
                return msg.result
        return ""

    review = asyncio.run(run_claude())

elif reviewer == "codex":
    result = subprocess.run(
        ["codex", "--full-auto", PROMPT],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=os.getcwd(),
    )
    review = result.stdout or result.stderr
    if result.returncode != 0 and not review.strip():
        print(f"Codex exited {result.returncode} with no output.", file=sys.stderr)
        sys.exit(1)

else:
    print(f"Unknown REVIEWER={reviewer!r}", file=sys.stderr)
    sys.exit(1)

if not review.strip():
    print("Reviewer produced no output — skipping comment.")
    sys.exit(0)

# ── Format comment ────────────────────────────────────────────────────────────

badge = "🤖 Claude" if reviewer == "claude" else "🤖 Codex"
body = f"## {badge} Peer Review\n\n_{reason}_\n\n{review}"
if truncated:
    body += "\n\n> ⚠️ Diff exceeded 80K chars and was truncated."

print(body)

if DRY_RUN:
    print("\n[dry-run] Gitea comment not posted.")
    sys.exit(0)

# ── Post comment ──────────────────────────────────────────────────────────────

r = httpx.post(
    f"{os.environ['GITEA_API']}/repos/{os.environ['REPO']}/issues"
    f"/{os.environ['PR_NUMBER']}/comments",
    headers={
        "Authorization": f"token {os.environ['GITEA_TOKEN']}",
        "Content-Type": "application/json",
    },
    json={"body": body},
    timeout=30,
)
r.raise_for_status()
print(f"\nComment posted (id={r.json()['id']})")

# ── Verdict ───────────────────────────────────────────────────────────────────

if "CRITICAL_ISSUES" in review or "🚨 CRITICAL:" in review:
    print("Critical issues flagged — failing check.")
    sys.exit(1)
