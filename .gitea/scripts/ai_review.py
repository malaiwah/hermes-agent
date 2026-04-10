#!/usr/bin/env python3
"""AI peer review script.

Uses Qwen 3.5 397B MoE on aibeast (10.15.0.166:8000) via the OpenAI-
compatible vLLM endpoint. The model has a 524K context window, so even
large diffs fit without truncation in most cases.

Usage:
  Normal:   python ai_review.py
  Dry run:  python ai_review.py --dry-run
"""
import os
import re
import subprocess
import sys

import httpx
from openai import OpenAI

DRY_RUN = "--dry-run" in sys.argv

base_sha = os.environ["BASE_SHA"]
head_sha = os.environ["HEAD_SHA"]

# ── Gather diff ───────────────────────────────────────────────────────────────

diff = subprocess.check_output(["git", "diff", base_sha, head_sha], text=True)
changed = subprocess.check_output(
    ["git", "diff", "--name-only", base_sha, head_sha], text=True
).strip()

MAX_DIFF = 200_000  # 200K chars — Qwen 3.5 has 524K context
truncated = len(diff) > MAX_DIFF
if truncated:
    diff = diff[:MAX_DIFF] + "\n\n[diff truncated at 200K chars]"

if not diff.strip():
    print("Empty diff — nothing to review.")
    sys.exit(0)

# ── Gather extra context for key changed files ────────────────────────────────

# Read the full content of up to 5 key changed files so the reviewer can
# see imports, callers, and surrounding code — not just the diff hunks.
context_files = ""
key_extensions = {".py", ".yml", ".yaml", ".sh", ".toml"}
files_added = 0
for fname in changed.split("\n"):
    if files_added >= 5:
        break
    fname = fname.strip()
    if not fname:
        continue
    if not any(fname.endswith(ext) for ext in key_extensions):
        continue
    if not os.path.isfile(fname):
        continue
    try:
        content = open(fname).read()
        if len(content) > 20_000:
            content = content[:20_000] + "\n[file truncated at 20K chars]"
        context_files += f"\n\n--- {fname} (full file) ---\n{content}"
        files_added += 1
    except Exception:
        pass

PROMPT = f"""You are peer-reviewing code changes. Be thorough and constructively critical.

Changed files:
{changed}

Diff:
{diff}
{context_files}

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

# ── Call Qwen 3.5 397B MoE on aibeast ────────────────────────────────────────

print(f"Reviewing {len(changed.split(chr(10)))} files "
      f"({len(diff):,} chars diff) with Qwen 3.5 397B MoE...")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
    base_url=os.environ.get("OPENAI_BASE_URL", "http://10.15.0.166:8000/v1"),
)

try:
    response = client.chat.completions.create(
        model="qwen35-397b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert code reviewer. Be thorough, specific, "
                    "and cite file:line when pointing out issues. Focus on "
                    "correctness, security, and maintainability."
                ),
            },
            {"role": "user", "content": PROMPT},
        ],
        max_tokens=8192,
        temperature=0.7,
        # Qwen 3.5 MoE supports extended thinking
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
            },
        },
    )
    review = response.choices[0].message.content or ""

    # Strip thinking blocks if the model returned them
    review = re.sub(r"<think>.*?</think>", "", review, flags=re.DOTALL).strip()

except Exception as e:
    review = f"API Error: {e}"

if not review.strip():
    print("Reviewer produced no output — skipping comment.")
    sys.exit(0)

# ── Format comment ────────────────────────────────────────────────────────────

body = f"## 🤖 Qwen 3.5 Peer Review\n\n{review}"
if truncated:
    body += "\n\n> ⚠️ Diff exceeded 200K chars and was truncated."

# Add model info footer
usage = getattr(response, "usage", None)
if usage:
    body += (
        f"\n\n<sub>Model: Qwen 3.5 397B MoE on aibeast | "
        f"Tokens: {usage.prompt_tokens:,} in / {usage.completion_tokens:,} out</sub>"
    )

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
    timeout=120,
)
r.raise_for_status()
print(f"\nComment posted (id={r.json()['id']})")

# ── Verdict ───────────────────────────────────────────────────────────────────

if "CRITICAL_ISSUES" in review or "🚨 CRITICAL:" in review:
    print("Critical issues flagged — failing check.")
    sys.exit(1)
