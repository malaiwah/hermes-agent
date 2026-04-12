#!/usr/bin/env python3
"""AI peer review script.

Uses Qwen 3.5 397B MoE on aibeast (10.15.0.166:8000) via the OpenAI-
compatible vLLM endpoint. The model has a 524K context window, so even
large diffs fit without truncation in most cases.

Usage:
  Normal:   python ai_review.py
  Dry run:  python ai_review.py --dry-run
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from agent.model_metadata import estimate_tokens_rough
from agent.usage_pricing import CanonicalUsage, normalize_usage

DRY_RUN = "--dry-run" in sys.argv
MAX_TARGETED_SNIPPET_CHARS = 12_000
MAX_TARGETED_SNIPPET_FILES = 8
TARGETED_SNIPPET_LINE_WINDOW = 25
MAX_INLINE_COMMENTS = 5
MAX_VERIFICATION_ISSUES = 3
MAX_VERIFICATION_QUERIES = 5
MAX_VERIFICATION_RESULTS = 5
MAX_VERIFICATION_CHARS = 4_000
MAX_VERIFICATION_LINE_WINDOW = 40
MAX_PARALLEL_ISSUE_VERIFIERS = 10
MODEL_NAME = "qwen35-397b"
MAX_REVIEW_PROMPT_TOKENS = 300_000
PRIMARY_REVIEW_MAX_TOKENS = 32_768
AI_REVIEW_HEADER = "## 🤖 Qwen 3.5 Peer Review"
AI_REVIEW_MARKER = "<!-- hermes-ai-review -->"
VERDICTS = {"APPROVED", "NEEDS_WORK", "CRITICAL_ISSUES"}
SEVERITIES = ("CRITICAL", "WARNING", "SUGGESTION")
INLINE_COMMENT_SEVERITIES = {"CRITICAL", "WARNING"}
SEVERITY_MARKERS = {
    "CRITICAL": "🚨 CRITICAL",
    "WARNING": "⚠️ WARNING",
    "SUGGESTION": "💡 SUGGESTION",
}
INLINE_REJECTION_PATTERNS = (
    re.compile(r"\bthis pattern is correct\b", re.IGNORECASE),
    re.compile(r"\bsimilar patterns elsewhere\b", re.IGNORECASE),
    re.compile(r"\breview all\b", re.IGNORECASE),
    re.compile(r"\belsewhere in the file\b", re.IGNORECASE),
)
ISSUE_REJECTION_PATTERNS = (
    re.compile(r"\bmissing docstring\b", re.IGNORECASE),
    re.compile(r"\bmissing type annotation\b", re.IGNORECASE),
    re.compile(r"\bunused import\b", re.IGNORECASE),
    re.compile(r"\bduplicate code\b", re.IGNORECASE),
    re.compile(r"\bdry violation\b", re.IGNORECASE),
    re.compile(r"\bmissing test\b", re.IGNORECASE),
    re.compile(r"\bedge cases?\b", re.IGNORECASE),
    re.compile(r"\bredundant .*strip", re.IGNORECASE),
    re.compile(r"\badd a comment linking\b", re.IGNORECASE),
    re.compile(r"\bconsider if this warrants\b", re.IGNORECASE),
)
SUMMARY_PLACEHOLDER_PATTERNS = (
    re.compile(r"^\|\s*severity\s*\|\s*count\s*\|", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\*\*priority fixes:\*\*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\*\*recommendations?:\*\*", re.IGNORECASE | re.MULTILINE),
)
BULLET_ISSUE_RE = re.compile(
    r"^-\s+"
    r"(?:(?:🚨\s+)?(?P<critical>CRITICAL):|"
    r"(?:⚠️\s+)?(?P<warning>WARNING):|"
    r"(?:💡\s+)?(?P<suggestion>SUGGESTION):)"
    r"\s*"
    r"(?:`(?P<location>[^`]+)`\s*)?"
    r"(?:(?:\*\*(?P<title_bold>.+?)\*\*)|(?P<title_plain>.+?))"
    r"(?:\s+[—-]\s*(?P<rest>.*))?$"
)
# Qwen3.5 model card sampling guidance:
# - Thinking mode for precise coding tasks: temperature=0.6, top_p=0.95,
#   top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
# - Instruct / non-thinking mode for general tasks: temperature=0.7, top_p=0.8,
#   top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
# - Adequate output length: 32,768 tokens for most queries
THINKING_PRECISE_CODING_SAMPLING: dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "extra_body": {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},
    },
}
NON_THINKING_INSTRUCT_SAMPLING: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "presence_penalty": 1.5,
    "frequency_penalty": 0.0,
    "extra_body": {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    },
}
REVIEW_JSON_SCHEMA: dict[str, Any] = {
    "name": "review_payload",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "verdict": {
                "type": "string",
                "enum": ["APPROVED", "NEEDS_WORK", "CRITICAL_ISSUES"],
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["CRITICAL", "WARNING", "SUGGESTION"],
                        },
                        "title": {"type": "string"},
                        "details": {"type": "string"},
                        "path": {"type": ["string", "null"]},
                        "line": {"type": ["integer", "null"]},
                        "side": {
                            "type": ["string", "null"],
                            "enum": ["NEW", "OLD", None],
                        },
                    },
                    "required": ["severity", "title", "details", "path", "line", "side"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "verdict", "issues"],
        "additionalProperties": False,
    },
    "strict": True,
}
VERIFICATION_REQUEST_SCHEMA: dict[str, Any] = {
    "name": "review_verification_requests",
    "schema": {
        "type": "object",
        "properties": {
            "requests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue_index": {"type": "integer"},
                        "kind": {
                            "type": "string",
                            "enum": ["read_file_snippet", "search_repo"],
                        },
                        "path": {"type": ["string", "null"]},
                        "query": {"type": ["string", "null"]},
                        "start_line": {"type": ["integer", "null"]},
                        "end_line": {"type": ["integer", "null"]},
                    },
                    "required": [
                        "issue_index",
                        "kind",
                        "path",
                        "query",
                        "start_line",
                        "end_line",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["requests"],
        "additionalProperties": False,
    },
    "strict": True,
}


def run_git(args: list[str]) -> str:
    return subprocess.check_output(args, text=True)


def _clone_sampling_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(profile)


def _sampling_trace(profile_name: str, profile: dict[str, Any]) -> dict[str, Any]:
    extra_body = profile.get("extra_body") or {}
    chat_kwargs = extra_body.get("chat_template_kwargs") or {}
    return {
        "profile": profile_name,
        "temperature": profile.get("temperature"),
        "top_p": profile.get("top_p"),
        "top_k": extra_body.get("top_k"),
        "min_p": extra_body.get("min_p"),
        "presence_penalty": profile.get("presence_penalty"),
        "repetition_penalty": extra_body.get("repetition_penalty"),
        "enable_thinking": bool(chat_kwargs.get("enable_thinking")),
    }


def _build_completion_kwargs(
    *,
    messages: list[dict[str, Any]],
    max_tokens: int,
    sampling_profile: dict[str, Any],
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    kwargs.update(_clone_sampling_profile(sampling_profile))
    if response_format is not None:
        kwargs["response_format"] = response_format
    return kwargs


def build_openai_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        base_url=os.environ.get("OPENAI_BASE_URL", "http://10.15.0.166:8000/v1"),
    )


@lru_cache(maxsize=256)
def estimate_prompt_tokens(prompt: str, *, model: str = MODEL_NAME, base_url: str | None = None) -> int:
    endpoint = (base_url or os.environ.get("OPENAI_BASE_URL") or "http://10.15.0.166:8000/v1").rstrip("/")
    tokenize_url = f"{endpoint}/tokenize"
    try:
        with httpx.Client(timeout=10) as client:
            response = client.post(
                tokenize_url,
                json={"model": model, "prompt": prompt},
            )
            response.raise_for_status()
            payload = response.json()
            count = payload.get("count")
            if isinstance(count, int) and count >= 0:
                return count
    except Exception:
        pass
    return estimate_tokens_rough(prompt)


def _empty_usage_bucket() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "request_count": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
    }


def _canonical_usage_to_bucket(usage: CanonicalUsage) -> dict[str, int]:
    return {
        "input_tokens": int(usage.input_tokens),
        "output_tokens": int(usage.output_tokens),
        "cache_read_tokens": int(usage.cache_read_tokens),
        "cache_write_tokens": int(usage.cache_write_tokens),
        "reasoning_tokens": int(usage.reasoning_tokens),
        "request_count": int(usage.request_count),
        "prompt_tokens": int(usage.prompt_tokens),
        "total_tokens": int(usage.total_tokens),
    }


def _merge_usage_bucket(
    target: dict[str, int] | None,
    addition: dict[str, int] | None,
) -> dict[str, int]:
    merged = dict(target or _empty_usage_bucket())
    if not addition:
        return merged
    for key in _empty_usage_bucket():
        merged[key] = int(merged.get(key, 0)) + int(addition.get(key, 0))
    return merged


def _record_usage(
    trace: dict[str, Any] | None,
    *,
    stage: str,
    response_usage: Any | None,
) -> dict[str, int]:
    if response_usage:
        canonical = normalize_usage(response_usage, provider="openai")
        bucket = _canonical_usage_to_bucket(canonical)
    else:
        bucket = _empty_usage_bucket()
    if trace is None:
        return bucket
    trace.setdefault("usage_by_stage", []).append({"stage": stage, **bucket})
    trace["usage_totals"] = _merge_usage_bucket(trace.get("usage_totals"), bucket)
    return bucket


def _merge_usage_trace(
    target_trace: dict[str, Any] | None,
    source_trace: dict[str, Any] | None,
) -> None:
    if target_trace is None or not source_trace:
        return
    stage_usage = list(source_trace.get("usage_by_stage") or [])
    if stage_usage:
        target_trace.setdefault("usage_by_stage", []).extend(stage_usage)
    target_trace["usage_totals"] = _merge_usage_bucket(
        target_trace.get("usage_totals"),
        source_trace.get("usage_totals"),
    )


def call_json_schema_completion(
    client: OpenAI,
    *,
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int,
    trace: dict[str, Any] | None = None,
    trace_stage: str | None = None,
) -> tuple[dict[str, Any] | None, Any | None]:
    sampling_profile = NON_THINKING_INSTRUCT_SAMPLING
    usage_stage = trace_stage or "json_schema_completion"
    if trace is not None and trace_stage:
        trace.setdefault("schema_calls", []).append(
            {
                "stage": trace_stage,
                "schema": schema.get("name"),
                "sampling": _sampling_trace("non_thinking_instruct", sampling_profile),
                "max_tokens": max_tokens,
            }
        )
    try:
        response = client.chat.completions.create(
            **_build_completion_kwargs(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                sampling_profile=sampling_profile,
                response_format={"type": "json_schema", "json_schema": schema},
            )
        )
    except Exception:
        return None, None
    _record_usage(trace, stage=usage_stage, response_usage=getattr(response, "usage", None))
    message = response.choices[0].message
    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, dict):
        return parsed, getattr(response, "usage", None)
    if isinstance(parsed, str):
        try:
            return _extract_json_payload(parsed), getattr(response, "usage", None)
        except Exception:
            pass
    content = message.content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
        raw = strip_thinking_blocks("\n".join(text_parts).strip())
    else:
        raw = strip_thinking_blocks(content or "")
    if not raw:
        return None, getattr(response, "usage", None)
    try:
        return _extract_json_payload(raw), getattr(response, "usage", None)
    except Exception:
        return None, getattr(response, "usage", None)


def gather_diff(base_sha: str, head_sha: str) -> tuple[str, str]:
    diff = run_git(["git", "diff", base_sha, head_sha])
    changed = run_git(["git", "diff", "--name-only", base_sha, head_sha]).strip()
    return diff, changed


def _diff_header_path(line: str) -> str | None:
    match = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
    if not match:
        return None
    return _normalize_path(match.group(2)) or _normalize_path(match.group(1))


def _merge_line_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _extract_section_ranges(section_lines: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for line in section_lines:
        if not line.startswith("@@ "):
            continue
        match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if not match:
            continue
        start = int(match.group(1))
        length = int(match.group(2) or "1")
        if length <= 0:
            length = 1
        ranges.append(
            (
                max(1, start - TARGETED_SNIPPET_LINE_WINDOW),
                start + length - 1 + TARGETED_SNIPPET_LINE_WINDOW,
            )
        )
    return _merge_line_ranges(ranges)


def _split_diff_sections(diff: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_lines: list[str] = []
    current_path: str | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if current_lines:
                sections.append(
                    {
                        "path": current_path,
                        "text": "\n".join(current_lines),
                        "ranges": _extract_section_ranges(current_lines),
                    }
                )
            current_lines = [line]
            current_path = _diff_header_path(line)
            continue
        if current_lines:
            if line.startswith("+++ "):
                candidate = line[4:].strip()
                if candidate != "/dev/null":
                    current_path = _normalize_path(candidate)
            current_lines.append(line)
    if current_lines:
        sections.append(
            {
                "path": current_path,
                "text": "\n".join(current_lines),
                "ranges": _extract_section_ranges(current_lines),
            }
        )
    return sections


def _build_targeted_snippet_block(
    path: str,
    ranges: list[tuple[int, int]],
    *,
    max_chars: int = MAX_TARGETED_SNIPPET_CHARS,
) -> str | None:
    snippets: list[str] = []
    remaining = max_chars
    for start, end in ranges[:3]:
        snippet = read_file_snippet(
            path,
            start_line=start,
            end_line=end,
            max_chars=max(400, remaining),
        )
        if not snippet:
            continue
        addition = snippet if not snippets else f"\n\n{snippet}"
        if len(addition) > remaining:
            break
        snippets.append(snippet)
        remaining -= len(addition)
        if remaining < 400:
            break
    if not snippets:
        return None
    return (
        f"--- {path} (targeted snippets around changed hunks) ---\n"
        + "\n\n".join(snippets)
    )


def pack_review_context(
    diff: str,
    *,
    changed: str = "",
    max_prompt_tokens: int = MAX_REVIEW_PROMPT_TOKENS,
    max_snippet_files: int = MAX_TARGETED_SNIPPET_FILES,
) -> tuple[str, bool, dict[str, Any]]:
    sections = _split_diff_sections(diff)
    if not sections:
        return diff, False, {
            "included_diff_paths": [],
            "omitted_diff_paths": [],
            "targeted_snippet_paths": [],
            "estimated_prompt_tokens": estimate_prompt_tokens(build_prompt(changed, diff, "")),
            "prompt_token_budget": max_prompt_tokens,
        }

    included_sections: list[str] = []
    included_paths: list[str] = []
    omitted_sections: list[dict[str, Any]] = []
    for section in sections:
        text = str(section.get("text") or "")
        candidate_diff = "\n\n".join([*included_sections, text])
        candidate_tokens = estimate_prompt_tokens(build_prompt(changed, candidate_diff, ""))
        if candidate_tokens <= max_prompt_tokens:
            included_sections.append(text)
            if section.get("path"):
                included_paths.append(str(section["path"]))
        else:
            omitted_sections.append(section)

    targeted_blocks: list[str] = []
    targeted_paths: list[str] = []
    for section in omitted_sections:
        if len(targeted_paths) >= max_snippet_files:
            break
        path = str(section.get("path") or "").strip()
        ranges = list(section.get("ranges") or [])
        if not path or not ranges:
            continue
        block = _build_targeted_snippet_block(
            path,
            ranges,
            max_chars=MAX_TARGETED_SNIPPET_CHARS,
        )
        if not block:
            continue
        candidate_blocks = [*targeted_blocks, block]
        candidate_context = (
            "Targeted source snippets for diff sections omitted from the initial patch context:\n\n"
            + "\n\n".join(candidate_blocks)
        )
        candidate_tokens = estimate_prompt_tokens(
            build_prompt(changed, "\n\n".join(included_sections), candidate_context)
        )
        if candidate_tokens > max_prompt_tokens:
            continue
        targeted_blocks.append(block)
        targeted_paths.append(path)

    packed_parts = [part for part in ["\n\n".join(included_sections)] if part]
    if targeted_blocks:
        packed_parts.append(
            "Targeted source snippets for diff sections omitted from the initial patch context:\n\n"
            + "\n\n".join(targeted_blocks)
        )

    omitted_paths = [str(section.get("path") or "") for section in omitted_sections if section.get("path")]
    omitted_without_snippets = [
        path for path in omitted_paths if path and path not in set(targeted_paths)
    ]
    if omitted_sections:
        summary = (
            "[review context packed: "
            f"kept full diff for {len(included_sections)}/{len(sections)} file sections; "
            f"added targeted snippets for {len(targeted_paths)} omitted files"
        )
        if omitted_without_snippets:
            summary += f"; omitted {len(omitted_without_snippets)} files entirely"
        summary += "]"
        candidate_packed = "\n\n".join([*packed_parts, summary])
        if estimate_prompt_tokens(build_prompt(changed, candidate_packed, "")) <= max_prompt_tokens:
            packed_parts.append(summary)

    packed = "\n\n".join(part for part in packed_parts if part)
    truncated = bool(omitted_sections)
    return packed, truncated, {
        "included_diff_paths": included_paths,
        "omitted_diff_paths": omitted_paths,
        "targeted_snippet_paths": targeted_paths,
        "omitted_without_snippets": omitted_without_snippets,
        "packed_section_count": len(included_sections),
        "total_section_count": len(sections),
        "estimated_prompt_tokens": estimate_prompt_tokens(build_prompt(changed, packed, "")),
        "prompt_token_budget": max_prompt_tokens,
    }


def build_prompt(changed: str, diff: str, context_files: str) -> str:
    return f"""You are peer-reviewing code changes. Be thorough and constructively critical.

Changed files:
{changed}

Diff:
{diff}
{context_files}

Review only the changed lines and the behavior directly introduced by this diff.
Do not speculate about broader architecture, theoretical races, path traversal, or security issues
unless the changed code in this diff shows a concrete exploit path or unsynchronized mutation.
Ignore low-value nits such as duplicate helpers, missing docstrings/type annotations, unused imports,
extra tests you would like to see, or style/cleanup ideas unless they cause a real bug in this PR.

Structure your review as:

## Summary
Brief overview of what changed and why.

## Issues
- 🚨 CRITICAL: security holes, data loss, broken logic, race conditions
- ⚠️ WARNING: bugs, bad patterns, missing error handling, test gaps
- 💡 SUGGESTION: improvements, simplifications, naming

When you point out a concrete code issue, include file:line references where possible.
For each issue, put the primary location in the issue heading as `relative/path.py:123-125`.
Do not use vague location labels like `Lines: 25` without a file path.
If there are no issues in a category, omit it.
Do not replace the Issues section with a scorecard, counts table, or generic checklist.
If your verdict is NEEDS_WORK or CRITICAL_ISSUES, include at least one concrete issue with evidence from the diff.
Prefer the 1-5 most important issues over exhaustive nitpicking.
Prefer APPROVED if you cannot support a finding from the changed lines themselves.

## Verdict
One of: APPROVED | NEEDS_WORK | CRITICAL_ISSUES
"""


def strip_thinking_blocks(review: str) -> str:
    return re.sub(r"<think>.*?</think>", "", review, flags=re.DOTALL).strip()


def extract_verdict(review: str) -> str | None:
    match = re.search(r"^## Verdict\s*$\n+([A-Z_]+)", review, flags=re.MULTILINE)
    if not match:
        return None
    verdict = match.group(1).strip().upper()
    return verdict if verdict in VERDICTS else None


def _fallback_review(summary: str) -> dict[str, Any]:
    return {
        "summary": summary,
        "verdict": "NEEDS_WORK",
        "issues": [],
    }


def extract_section(review: str, heading: str) -> str:
    pattern = rf"^{re.escape(heading)}\s*$\n+(.*?)(?=^#{{2,}}\s|\Z)"
    match = re.search(pattern, review, flags=re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_lead_summary(review: str) -> str:
    lines = review.splitlines()
    paragraph: list[str] = []
    in_code = False
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("```"):
            in_code = not in_code
            if paragraph:
                break
            continue
        if in_code:
            continue
        if not line:
            if paragraph:
                break
            continue
        if line.startswith("#") or line.startswith("|") or line.startswith(">"):
            if paragraph:
                break
            continue
        if re.match(r"^(?:\*\*File:\*\*|File:)\s", line):
            if paragraph:
                break
            continue
        if re.match(r"^(?:[-*]|\d+\.)\s", line):
            if paragraph:
                break
            continue
        paragraph.append(line)
    return " ".join(paragraph).strip()


def clean_issue_details(details: str) -> str:
    text = str(details or "").strip()
    if not text:
        return "No details provided."
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*---+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\*\*Recommendation:\*\*.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\*\*Fix:\*\*.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\*\*Issue:\*\*\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > 900:
        text = text[:897].rstrip() + "..."
    return text or "No details provided."


def parse_location(location: str | None) -> tuple[str | None, int | None]:
    normalized = _normalize_path(location)
    if not normalized:
        return None, None
    match = re.match(r"(.+?):(\d+)(?:-\d+)?$", normalized)
    if match:
        return _normalize_path(match.group(1)), int(match.group(2))
    return normalized, None


def get_repo_root() -> Path:
    return Path.cwd().resolve()


def resolve_repo_file(path: str | None) -> Path | None:
    normalized = _normalize_path(path)
    if not normalized:
        return None
    candidate = Path(normalized)
    if candidate.is_absolute():
        return None
    repo_root = get_repo_root()
    try:
        resolved = (repo_root / candidate).resolve()
        resolved.relative_to(repo_root)
    except Exception:
        return None
    if not resolved.is_file():
        return None
    return resolved


def read_file_snippet(
    path: str | None,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    max_chars: int = MAX_VERIFICATION_CHARS,
) -> str | None:
    resolved = resolve_repo_file(path)
    if resolved is None:
        return None
    lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return f"{resolved.relative_to(get_repo_root())}: empty file"
    start = max(1, int(start_line or 1))
    end = int(end_line or min(len(lines), start + MAX_VERIFICATION_LINE_WINDOW - 1))
    if end < start:
        end = start
    start = min(start, len(lines))
    end = min(end, len(lines))
    excerpt = lines[start - 1 : end]
    body = "\n".join(f"{start + idx}: {line}" for idx, line in enumerate(excerpt))
    header = f"{resolved.relative_to(get_repo_root())}:{start}-{end}"
    text = f"{header}\n{body}".strip()
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n[snippet truncated]"
    return text


def search_repo(
    query: str | None,
    *,
    max_results: int = MAX_VERIFICATION_RESULTS,
    allowed_paths: list[str] | None = None,
    path_glob: str | None = None,
    max_chars: int = MAX_VERIFICATION_CHARS,
) -> str | None:
    needle = str(query or "").strip()
    if not needle:
        return None
    rg = shutil.which("rg")
    if not rg:
        return None
    args = [rg, "-F", "-n", "--no-heading", "--color", "never", needle]
    if path_glob:
        args.extend(["--glob", path_glob])
    resolved_allowed: list[str] = []
    for path in allowed_paths or []:
        resolved = resolve_repo_file(path)
        if resolved is None:
            continue
        try:
            resolved_allowed.append(str(resolved.relative_to(get_repo_root())))
        except Exception:
            continue
    if allowed_paths is not None and not resolved_allowed:
        return None
    if resolved_allowed:
        args.extend(["--", *resolved_allowed])
    proc = subprocess.run(
        args,
        cwd=get_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in {0, 1}:
        return None
    matches = [line for line in proc.stdout.splitlines() if line.strip()][:max_results]
    if not matches:
        return None
    text = "\n".join(matches)
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n[search truncated]"
    return text


def _extract_json_payload(review: str) -> dict[str, Any]:
    review = review.strip()
    if review.startswith("```"):
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", review, flags=re.DOTALL)
        if fenced:
            review = fenced.group(1).strip()
    try:
        payload = json.loads(review)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", review, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("review payload is not a JSON object")
    return payload


def parse_review_payload(review: str) -> dict[str, Any]:
    payload = _extract_json_payload(strip_thinking_blocks(review))
    summary = str(payload.get("summary") or "").strip()
    verdict = str(payload.get("verdict") or "").strip().upper()
    if verdict not in VERDICTS:
        raise ValueError(f"invalid verdict: {verdict!r}")
    issues: list[dict[str, Any]] = []
    raw_issues = payload.get("issues") or []
    if not isinstance(raw_issues, list):
        raise ValueError("issues must be a list")
    for issue in raw_issues:
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity") or "").strip().upper()
        if severity not in SEVERITIES:
            continue
        title = str(issue.get("title") or "").strip()
        details = str(issue.get("details") or "").strip()
        path = issue.get("path")
        if path is not None:
            path = str(path).strip()
        side = issue.get("side")
        if side is not None:
            side = str(side).strip().upper()
        line = issue.get("line")
        try:
            line = int(line) if line is not None else None
        except (TypeError, ValueError):
            line = None
        issues.append(
            {
                "severity": severity,
                "title": title or "Untitled finding",
                "details": details or "No details provided.",
                "path": path or None,
                "line": line if line and line > 0 else None,
                "side": side if side in {"NEW", "OLD"} else None,
                "location_source": "field" if path and line else None,
            }
        )
    return {
        "summary": summary or "No summary provided.",
        "verdict": verdict,
        "issues": issues,
    }


def parse_freeform_review(review: str) -> dict[str, Any]:
    cleaned = strip_thinking_blocks(review)
    verdict = extract_verdict(cleaned) or "NEEDS_WORK"
    summary = (
        extract_section(cleaned, "## Summary")
        or extract_section(cleaned, "## Executive Summary")
        or "No summary provided."
    )
    issues: list[dict[str, Any]] = []
    current_severity: str | None = None
    lines = cleaned.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        lower_line = line.lower()
        if line.startswith("## "):
            if "critical issues" in lower_line:
                current_severity = "CRITICAL"
                index += 1
                continue
            if "major issues" in lower_line or "warning issues" in lower_line:
                current_severity = "WARNING"
                index += 1
                continue
            if "minor issues" in lower_line or "suggestion" in lower_line:
                current_severity = "SUGGESTION"
                index += 1
                continue
        if line.startswith("### "):
            if "critical" in lower_line:
                current_severity = "CRITICAL"
                index += 1
                continue
            elif "warning" in lower_line or "major issues" in lower_line:
                current_severity = "WARNING"
                index += 1
                continue
            elif "suggestion" in lower_line or "improvement" in lower_line or "minor issues" in lower_line:
                current_severity = "SUGGESTION"
                index += 1
                continue
        if line.startswith("- "):
            bullet_match = re.match(
                r"^-\s+(?:(?:🚨\s+)?CRITICAL:|(?:⚠️\s+)?WARNING:|(?:💡\s+)?SUGGESTION:)",
                line,
            )
            if bullet_match:
                bullet_issue_match = BULLET_ISSUE_RE.match(line)
                if "CRITICAL" in line:
                    current_severity = "CRITICAL"
                elif "WARNING" in line:
                    current_severity = "WARNING"
                elif "SUGGESTION" in line:
                    current_severity = "SUGGESTION"
                if bullet_issue_match:
                    title = (
                        bullet_issue_match.group("title_bold")
                        or bullet_issue_match.group("title_plain")
                        or "Untitled finding"
                    ).strip()
                    path, line_number = parse_location(bullet_issue_match.group("location"))
                    detail_lines: list[str] = []
                    rest = (bullet_issue_match.group("rest") or "").strip()
                    if rest:
                        detail_lines.append(rest)
                    index += 1
                    while index < len(lines):
                        next_line = lines[index]
                        stripped = next_line.strip()
                        if (
                            stripped.startswith("## ")
                            or stripped.startswith("### ")
                            or BULLET_ISSUE_RE.match(stripped)
                            or re.match(
                                r"^-\s+(?:(?:🚨\s+)?CRITICAL:|(?:⚠️\s+)?WARNING:|(?:💡\s+)?SUGGESTION:)",
                                stripped,
                            )
                        ):
                            break
                        if stripped == "---":
                            index += 1
                            break
                        detail_lines.append(next_line)
                        index += 1
                    issues.append(
                        {
                            "severity": current_severity or "WARNING",
                            "title": title,
                            "details": clean_issue_details("\n".join(detail_lines)),
                            "path": path,
                            "line": line_number,
                            "side": None,
                            "location_source": "field" if path and line_number else None,
                        }
                    )
                    continue
        heading_prefix = None
        if line.startswith("#### "):
            heading_prefix = "#### "
        elif line.startswith("### "):
            heading_prefix = "### "
        if heading_prefix:
            heading_text = line[len(heading_prefix):].strip()
            heading_match = re.search(
                r"(?:\d+\.\s+)?\*\*(?P<title>.+?)\*\*(?:\s*(?:[-—]\s*)?`(?P<location>[^`]+)`)?",
                heading_text,
            )
            title = heading_text
            path = None
            line_number = None
            location_source = None
            if heading_match:
                title = heading_match.group("title").strip()
                path, line_number = parse_location(heading_match.group("location"))
                if path and line_number:
                    location_source = "heading"
            detail_lines: list[str] = []
            index += 1
            while index < len(lines):
                next_line = lines[index]
                stripped = next_line.strip()
                if stripped.startswith("#### ") or stripped.startswith("### ") or stripped.startswith("## "):
                    break
                detail_lines.append(next_line)
                index += 1
            details = "\n".join(detail_lines).strip()
            if (not path or not line_number) and details:
                file_match = re.search(
                    r"(?:\*\*File:\*\*|File:)\s*`(?P<location>[^`]+)`",
                    details,
                )
                if file_match:
                    path, line_number = parse_location(file_match.group("location"))
                    if path and line_number:
                        location_source = "details"
                    details = re.sub(
                        r"^\s*(?:\*\*File:\*\*|File:)\s*`[^`]+`\s*$\n?",
                        "",
                        details,
                        flags=re.MULTILINE,
                    ).strip()
            if details:
                details = re.sub(r"\n{2,}", "\n\n", details)
            issues.append(
                {
                    "severity": current_severity or "WARNING",
                    "title": title or "Untitled finding",
                    "details": clean_issue_details(details),
                    "path": path,
                    "line": line_number,
                    "side": None,
                    "location_source": location_source,
                }
            )
            continue
        index += 1
    return {"summary": summary, "verdict": verdict, "issues": issues}


def render_markdown_review(review: dict[str, Any]) -> str:
    lines = ["## Summary", "", str(review["summary"]).strip()]
    issues = [
        issue for issue in list(review.get("issues") or []) if not is_placeholder_issue(issue)
    ]
    if issues:
        lines.extend(["", "## Issues"])
        for severity in SEVERITIES:
            bucket = [issue for issue in issues if issue.get("severity") == severity]
            if not bucket:
                continue
            for issue in bucket:
                location = ""
                if issue.get("path") and issue.get("line"):
                    location = f" `{issue['path']}:{issue['line']}`"
                lines.append(
                    f"- {SEVERITY_MARKERS[severity]}:{location} "
                    f"**{issue['title']}** — {issue['details']}"
                )
    lines.extend(["", "## Verdict", str(review["verdict"])])
    return "\n".join(lines)


def is_placeholder_issue(issue: dict[str, Any]) -> bool:
    title = str(issue.get("title") or "").strip().lower()
    details = str(issue.get("details") or "")
    if title in {"summary", "recommendation", "recommendations"}:
        return True
    if "| Severity | Count |" in details or "|----------|-------|" in details:
        return True
    return False


def is_rejected_issue(issue: dict[str, Any]) -> bool:
    text = "\n".join(str(issue.get(key) or "") for key in ("title", "details"))
    return any(pattern.search(text) for pattern in ISSUE_REJECTION_PATTERNS)


def is_placeholder_summary(summary: str) -> bool:
    text = str(summary or "").strip()
    if not text or text == "No summary provided.":
        return True
    return any(pattern.search(text) for pattern in SUMMARY_PLACEHOLDER_PATTERNS)


def has_actionable_issues(review: dict[str, Any]) -> bool:
    issues = list(review.get("issues") or [])
    return any(
        not is_placeholder_issue(issue) and not is_rejected_issue(issue)
        for issue in issues
    )


def issue_is_diff_grounded(
    issue: dict[str, Any],
    diff_line_index: dict[str, dict[str, set[int]]],
) -> bool:
    if is_placeholder_issue(issue) or is_rejected_issue(issue):
        return False
    path, line, _ = extract_issue_location(issue)
    if not path:
        return False
    diff_entry = diff_line_index.get(path)
    if not diff_entry:
        return False
    if line is None:
        return True
    side = str(issue.get("side") or "").upper()
    if side == "NEW":
        return line in diff_entry["new"]
    if side == "OLD":
        return line in diff_entry["old"]
    return line in diff_entry["new"] or line in diff_entry["old"]


def filter_review_to_diff(
    review: dict[str, Any],
    diff_line_index: dict[str, dict[str, set[int]]],
) -> dict[str, Any]:
    filtered = dict(review)
    filtered["issues"] = [
        issue
        for issue in list(review.get("issues") or [])
        if issue_is_diff_grounded(issue, diff_line_index)
    ]
    return filtered


def normalize_review_after_filter(review: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(review)
    issues = [
        issue
        for issue in list(normalized.get("issues") or [])
        if not is_placeholder_issue(issue) and not is_rejected_issue(issue)
    ]
    normalized["issues"] = issues
    if not issues:
        normalized["verdict"] = "APPROVED"
        summary = str(normalized.get("summary") or "").strip()
        if is_placeholder_summary(summary):
            normalized["summary"] = "No concrete diff-grounded issues were identified."
    elif any(issue.get("severity") == "CRITICAL" for issue in issues):
        normalized["verdict"] = "CRITICAL_ISSUES"
    else:
        normalized["verdict"] = "NEEDS_WORK"
    return normalized


def build_verification_request_prompt(review: dict[str, Any], changed_files: list[str]) -> str:
    issue_lines = []
    for idx, issue in enumerate(list(review.get("issues") or [])[:MAX_VERIFICATION_ISSUES]):
        issue_lines.append(
            {
                "issue_index": idx,
                "severity": issue.get("severity"),
                "title": issue.get("title"),
                "details": issue.get("details"),
                "path": issue.get("path"),
                "line": issue.get("line"),
            }
        )
    return (
        "You are deciding whether a small amount of local source context is needed to verify "
        "candidate code review findings.\n\n"
        f"Changed files:\n{json.dumps(changed_files, indent=2)}\n\n"
        f"Candidate findings:\n{json.dumps(issue_lines, indent=2)}\n\n"
        "Rules:\n"
        f"- Return at most {MAX_VERIFICATION_QUERIES} requests total.\n"
        f"- Only inspect the first {MAX_VERIFICATION_ISSUES} findings.\n"
        "- Use read_file_snippet when a direct local code window would confirm or refute a finding.\n"
        "- Use search_repo only for exact symbol or exact text searches.\n"
        "- Do not ask for broad exploration.\n"
        "- Return an empty requests array if the current findings are already grounded enough or too weak to verify.\n"
    )


def build_issue_verification_request_prompt(
    review_prompt: str,
    primary_review: dict[str, Any],
    issue_index: int,
    issue_review: dict[str, Any],
    changed_files: list[str],
) -> str:
    return (
        "You are a focused peer reviewer validating exactly one candidate claim from a primary code review.\n\n"
        f"Initial review request context:\n{review_prompt}\n\n"
        f"Primary candidate review:\n{json.dumps(primary_review, indent=2)}\n\n"
        f"Focused claim index: {issue_index}\n"
        f"Focused claim:\n{json.dumps(issue_review, indent=2)}\n\n"
        f"Changed files:\n{json.dumps(changed_files, indent=2)}\n\n"
        "Rules:\n"
        f"- Return at most {MAX_VERIFICATION_QUERIES} requests total.\n"
        "- Validate only this focused claim.\n"
        "- Use read_file_snippet when a direct local code window would confirm or refute the claim.\n"
        "- Use search_repo only for exact symbol or exact text searches.\n"
        "- Do not ask for broad exploration.\n"
        "- Return an empty requests array if the claim is already grounded enough or too weak to verify.\n"
    )


def request_verification_queries(
    client: OpenAI,
    review: dict[str, Any],
    changed_files: list[str],
    *,
    trace: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    issues = list(review.get("issues") or [])
    if not issues:
        return []
    payload, _ = call_json_schema_completion(
        client,
        system=(
            "You request only the minimum local verification needed for code review. "
            "Return valid JSON only."
        ),
        user=build_verification_request_prompt(review, changed_files),
        schema=VERIFICATION_REQUEST_SCHEMA,
        max_tokens=1200,
        trace=trace,
        trace_stage="verification_requests",
    )
    if not isinstance(payload, dict):
        return []
    requests = payload.get("requests") or []
    if not isinstance(requests, list):
        return []
    normalized = [req for req in requests[:MAX_VERIFICATION_QUERIES] if isinstance(req, dict)]
    if trace is not None:
        trace["verification_requests"] = normalized
    return normalized


def request_issue_verification_queries(
    client: OpenAI,
    *,
    review_prompt: str,
    primary_review: dict[str, Any],
    issue_index: int,
    issue_review: dict[str, Any],
    changed_files: list[str],
    trace: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    payload, _ = call_json_schema_completion(
        client,
        system=(
            "You request only the minimum local verification needed for a single code review claim. "
            "Return valid JSON only."
        ),
        user=build_issue_verification_request_prompt(
            review_prompt,
            primary_review,
            issue_index,
            issue_review,
            changed_files,
        ),
        schema=VERIFICATION_REQUEST_SCHEMA,
        max_tokens=1200,
        trace=trace,
        trace_stage=f"issue_{issue_index}_verification_requests",
    )
    if not isinstance(payload, dict):
        return []
    requests = payload.get("requests") or []
    if not isinstance(requests, list):
        return []
    normalized = []
    for req in requests[:MAX_VERIFICATION_QUERIES]:
        if not isinstance(req, dict):
            continue
        normalized_req = dict(req)
        normalized_req["issue_index"] = 0
        normalized.append(normalized_req)
    if trace is not None:
        trace["verification_requests"] = normalized
    return normalized


def collect_verification_context(
    review: dict[str, Any],
    changed_files: list[str],
    requests: list[dict[str, Any]],
    *,
    trace: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    issues = list(review.get("issues") or [])
    allowed_paths = {
        path
        for path in changed_files + [str(issue.get("path") or "") for issue in issues]
        if path
    }
    searchable_paths = [
        path
        for path in changed_files
        if path and resolve_repo_file(path) is not None
    ]
    contexts: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for req in requests[:MAX_VERIFICATION_QUERIES]:
        try:
            issue_index = int(req.get("issue_index"))
        except (TypeError, ValueError):
            continue
        if issue_index < 0 or issue_index >= min(len(issues), MAX_VERIFICATION_ISSUES):
            continue
        issue = issues[issue_index]
        kind = str(req.get("kind") or "").strip()
        if kind == "read_file_snippet":
            path = _normalize_path(req.get("path")) or _normalize_path(issue.get("path"))
            if not path or path not in allowed_paths:
                continue
            issue_line = issue.get("line")
            try:
                issue_line = int(issue_line) if issue_line is not None else None
            except (TypeError, ValueError):
                issue_line = None
            start_line = req.get("start_line")
            end_line = req.get("end_line")
            if start_line is None and issue_line is not None:
                start_line = max(1, issue_line - 10)
            if end_line is None and issue_line is not None:
                end_line = issue_line + 10
            snippet = read_file_snippet(path, start_line=start_line, end_line=end_line)
            if not snippet:
                continue
            key = (issue_index, kind, path, int(start_line or 0), int(end_line or 0))
            if key in seen:
                continue
            seen.add(key)
            contexts.append(
                {
                    "issue_index": issue_index,
                    "kind": kind,
                    "path": path,
                    "start_line": int(start_line) if start_line is not None else None,
                    "end_line": int(end_line) if end_line is not None else None,
                    "content": snippet,
                }
            )
        elif kind == "search_repo":
            query = str(req.get("query") or "").strip()
            if not query:
                continue
            result = search_repo(query, allowed_paths=searchable_paths)
            if not result:
                continue
            key = (issue_index, kind, query)
            if key in seen:
                continue
            seen.add(key)
            contexts.append(
                {
                    "issue_index": issue_index,
                    "kind": kind,
                    "query": query,
                    "allowed_paths": searchable_paths,
                    "content": result,
                }
            )
    if trace is not None:
        trace["verification_context"] = [
            {
                "kind": ctx.get("kind"),
                "path": ctx.get("path"),
                "start_line": ctx.get("start_line"),
                "end_line": ctx.get("end_line"),
                "query": ctx.get("query"),
                "allowed_paths": list(ctx.get("allowed_paths") or []),
                "result_lines": len(str(ctx.get("content") or "").splitlines()),
            }
            for ctx in contexts
        ]
    return contexts


def build_verification_review_prompt(
    review: dict[str, Any],
    verification_context: list[dict[str, Any]],
) -> str:
    return (
        "You are validating candidate code review findings against small, read-only local source "
        "snippets and exact repo searches.\n\n"
        f"Candidate review:\n{json.dumps(review, indent=2)}\n\n"
        f"Verification context:\n{json.dumps(verification_context, indent=2)}\n\n"
        "Rules:\n"
        "- Do not invent new findings.\n"
        "- Keep only findings supported by the existing review plus the supplied verification context.\n"
        "- Omit unsupported findings entirely.\n"
        "- You may downgrade severity, but do not upgrade severity.\n"
        "- Prefer APPROVED if no findings remain.\n"
        "- Return final JSON only.\n"
    )


def build_issue_verification_review_prompt(
    review_prompt: str,
    primary_review: dict[str, Any],
    issue_index: int,
    issue_review: dict[str, Any],
    verification_context: list[dict[str, Any]],
) -> str:
    return (
        "You are a second-pass peer reviewer validating exactly one candidate code review claim.\n\n"
        f"Initial review request context:\n{review_prompt}\n\n"
        f"Primary candidate review:\n{json.dumps(primary_review, indent=2)}\n\n"
        f"Focused claim index: {issue_index}\n"
        f"Focused claim review:\n{json.dumps(issue_review, indent=2)}\n\n"
        f"Verification context:\n{json.dumps(verification_context, indent=2)}\n\n"
        "Rules:\n"
        "- Do not invent new findings.\n"
        "- Validate only this focused claim.\n"
        "- Keep the claim only if it is supported by the claim text plus the supplied verification context.\n"
        "- Omit the claim entirely if it is unsupported or too weak.\n"
        "- Reject claims that depend on hypothetical future changes, hidden unseen state, or style/readability concerns instead of a current concrete bug shown in the supplied context.\n"
        "- You may downgrade severity, but do not upgrade severity.\n"
        "- Return APPROVED with no issues if the claim should be rejected.\n"
        "- Return final JSON only.\n"
    )


def _summarize_issue_verification_results(results: list[dict[str, Any]]) -> str:
    confirmed = sum(1 for result in results if result.get("status") == "confirmed")
    rejected = sum(1 for result in results if result.get("status") != "confirmed")
    if confirmed == 0:
        return "No primary findings were confirmed by issue-specific verification."
    if rejected == 0:
        return f"Issue-specific verification confirmed all {confirmed} primary findings."
    return (
        f"Issue-specific verification confirmed {confirmed} primary findings "
        f"and rejected {rejected} others."
    )


def verify_single_issue_with_local_context(
    *,
    review_prompt: str,
    primary_review: dict[str, Any],
    issue_index: int,
    issue: dict[str, Any],
    changed_files: list[str],
) -> dict[str, Any]:
    client = build_openai_client()
    issue_trace: dict[str, Any] = {
        "issue_index": issue_index,
        "title": issue.get("title"),
        "path": issue.get("path"),
        "line": issue.get("line"),
    }
    issue_review = {
        "summary": primary_review.get("summary") or "Focused issue verification.",
        "verdict": primary_review.get("verdict") or "NEEDS_WORK",
        "issues": [copy.deepcopy(issue)],
    }
    requests = request_issue_verification_queries(
        client,
        review_prompt=review_prompt,
        primary_review=primary_review,
        issue_index=issue_index,
        issue_review=issue_review,
        changed_files=changed_files,
        trace=issue_trace,
    )
    verification_context = collect_verification_context(
        issue_review,
        changed_files,
        requests,
        trace=issue_trace,
    )
    payload, _ = call_json_schema_completion(
        client,
        system=(
            "You are a skeptical peer reviewer validating exactly one candidate code review claim. "
            "Reject speculative, hypothetical, future-risk, or readability-only claims unless the supplied context shows a current concrete bug. "
            "Return valid JSON only."
        ),
        user=build_issue_verification_review_prompt(
            review_prompt,
            primary_review,
            issue_index,
            issue_review,
            verification_context,
        ),
        schema=REVIEW_JSON_SCHEMA,
        max_tokens=2000,
        trace=issue_trace,
        trace_stage=f"issue_{issue_index}_verification_review",
    )
    verified_review = issue_review
    if isinstance(payload, dict):
        try:
            verified_review = parse_review_payload(json.dumps(payload))
        except Exception:
            verified_review = issue_review
    issues = list(verified_review.get("issues") or [])
    status = "confirmed" if issues else "rejected"
    summary = str(verified_review.get("summary") or "").strip()
    if not summary or is_placeholder_summary(summary):
        summary = f"Focused verification {status} the claim."
    issue_trace["status"] = status
    issue_trace["summary"] = summary
    return {
        "issue_index": issue_index,
        "status": status,
        "summary": summary,
        "review": verified_review,
        "trace": issue_trace,
        "usage_totals": dict(issue_trace.get("usage_totals") or _empty_usage_bucket()),
    }


def verify_review_with_local_context(
    client: OpenAI,
    review: dict[str, Any],
    changed_files: list[str],
    *,
    review_prompt: str = "",
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issues = list(review.get("issues") or [])
    if not issues:
        return review
    max_workers = min(MAX_PARALLEL_ISSUE_VERIFIERS, max(1, len(issues)))
    primary_review = {
        "summary": review.get("summary"),
        "verdict": review.get("verdict"),
        "issues": [copy.deepcopy(issue) for issue in issues],
    }
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                verify_single_issue_with_local_context,
                review_prompt=review_prompt,
                primary_review=primary_review,
                issue_index=idx,
                issue=copy.deepcopy(issue),
                changed_files=changed_files,
            ): idx
            for idx, issue in enumerate(issues)
        }
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "issue_index": idx,
                        "status": "rejected",
                        "summary": f"Issue-specific verification failed: {exc}",
                        "review": {"summary": "", "verdict": "APPROVED", "issues": []},
                        "trace": {
                            "issue_index": idx,
                            "title": issues[idx].get("title"),
                            "path": issues[idx].get("path"),
                            "line": issues[idx].get("line"),
                            "status": "rejected",
                            "summary": f"Issue-specific verification failed: {exc}",
                            "verification_requests": [],
                            "verification_context": [],
                        },
                    }
                )
    ordered_results = sorted(results, key=lambda item: int(item.get("issue_index", 0)))
    confirmed_issues: list[dict[str, Any]] = []
    for result in ordered_results:
        confirmed_issues.extend(list(result.get("review", {}).get("issues") or []))
    verified = dict(review)
    verified["issues"] = confirmed_issues
    verified["summary"] = _summarize_issue_verification_results(ordered_results)
    if trace is not None:
        trace["parallel_issue_verification"] = {
            "max_workers": max_workers,
            "issue_count": len(issues),
        }
        trace["issue_verifications"] = [result.get("trace") or {} for result in ordered_results]
        for result in ordered_results:
            _merge_usage_trace(trace, result.get("trace"))
        trace["verification_requests"] = [
            {
                "issue_index": issue_trace.get("issue_index"),
                **req,
            }
            for issue_trace in trace["issue_verifications"]
            for req in list(issue_trace.get("verification_requests") or [])
        ]
        trace["verification_context"] = [
            {
                "issue_index": issue_trace.get("issue_index"),
                **ctx,
            }
            for issue_trace in trace["issue_verifications"]
            for ctx in list(issue_trace.get("verification_context") or [])
        ]
    return verified


def merge_review_data(primary: dict[str, Any], fallback: dict[str, Any] | None) -> dict[str, Any]:
    if not fallback:
        return primary
    merged = {
        "summary": primary.get("summary") or fallback.get("summary") or "No summary provided.",
        "verdict": primary.get("verdict") or fallback.get("verdict") or "NEEDS_WORK",
        "issues": list(primary.get("issues") or []),
    }
    if is_placeholder_summary(merged["summary"]) and fallback.get("summary"):
        merged["summary"] = fallback["summary"]
    if not merged["issues"] and fallback.get("issues"):
        merged["issues"] = list(fallback["issues"])
    return merged


def build_diff_line_index(diff: str) -> dict[str, dict[str, set[int]]]:
    line_index: dict[str, dict[str, set[int]]] = {}
    current_path: str | None = None
    old_line = 0
    new_line = 0
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            current_path = None
            continue
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path != "/dev/null":
                current_path = path[2:] if path.startswith("b/") else path
                line_index.setdefault(current_path, {"old": set(), "new": set()})
            continue
        if line.startswith("@@ "):
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if match:
                old_line = int(match.group(1))
                new_line = int(match.group(2))
            continue
        if current_path is None or not line:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            line_index[current_path]["new"].add(new_line)
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            line_index[current_path]["old"].add(old_line)
            old_line += 1
        elif line.startswith(" "):
            old_line += 1
            new_line += 1
    return line_index


def _normalize_path(path: str | None) -> str | None:
    if not path:
        return None
    normalized = path.strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    return normalized or None


def extract_issue_location(issue: dict[str, Any]) -> tuple[str | None, int | None, str | None]:
    path = _normalize_path(issue.get("path"))
    line = issue.get("line")
    location_source = issue.get("location_source")
    try:
        line = int(line) if line is not None else None
    except (TypeError, ValueError):
        line = None
    if path and line:
        return path, line, str(location_source or "field")
    for source, haystack in (
        ("title", str(issue.get("title") or "")),
        ("details", str(issue.get("details") or "")),
    ):
        for pattern in (
            r"`(?P<location>[^`\n]+:\d+(?:-\d+)?)`",
            r"(?P<location>[A-Za-z0-9_./-]+:\d+(?:-\d+)?)",
        ):
            match = re.search(pattern, haystack)
            if not match:
                continue
            parsed_path, parsed_line = parse_location(match.group("location"))
            if parsed_path and parsed_line:
                return parsed_path, parsed_line, source
    return path, line, str(location_source or "")


def should_inline_issue(issue: dict[str, Any]) -> bool:
    if issue.get("severity") not in INLINE_COMMENT_SEVERITIES:
        return False
    if is_rejected_issue(issue):
        return False
    text = "\n".join(str(issue.get(key) or "") for key in ("title", "details"))
    if any(pattern.search(text) for pattern in INLINE_REJECTION_PATTERNS):
        return False
    _, _, location_source = extract_issue_location(issue)
    return location_source in {"field", "heading", "title"}


def build_inline_comments(
    review: dict[str, Any],
    diff_line_index: dict[str, dict[str, set[int]]],
) -> list[dict[str, Any]]:
    inline_comments: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()
    for issue in review.get("issues") or []:
        if not should_inline_issue(issue):
            continue
        path, line, _ = extract_issue_location(issue)
        side = issue.get("side")
        if not path or not line:
            continue
        diff_entry = diff_line_index.get(path)
        if not diff_entry:
            continue
        if side == "OLD":
            continue
        if side not in {"NEW", "OLD"}:
            in_new = line in diff_entry["new"]
            in_old = line in diff_entry["old"]
            if not in_new or in_old:
                continue
            side = "NEW"
        if line not in diff_entry[side.lower()]:
            continue
        body = f"**[{issue['severity']}] {issue['title']}**\n\n{issue['details']}"
        key = (path, int(line), side, body)
        if key in seen:
            continue
        seen.add(key)
        inline_comments.append(
            {
                "path": path,
                "body": body,
                "new_position": int(line) if side == "NEW" else 0,
                "old_position": int(line) if side == "OLD" else 0,
            }
        )
        if len(inline_comments) >= MAX_INLINE_COMMENTS:
            break
    return inline_comments


def build_inline_review_body(review: dict[str, Any], *, commit_id: str) -> str:
    summary = str(review.get("summary") or "").strip()
    if len(summary) > 240:
        summary = summary[:237].rstrip() + "..."
    return (
        f"{AI_REVIEW_MARKER}\n"
        f"{AI_REVIEW_HEADER}\n\n"
        f"Inline findings for `{commit_id[:12]}`.\n\n"
        f"Verdict: {review['verdict']}\n\n"
        f"{summary}"
    )


def build_structuring_prompt(review: str) -> str:
    return f"""Convert the following code review into ONLY valid JSON.

Return this exact shape:
{{
  "summary": "brief overview of what changed and why",
  "verdict": "APPROVED | NEEDS_WORK | CRITICAL_ISSUES",
  "issues": [
    {{
      "severity": "CRITICAL | WARNING | SUGGESTION",
      "title": "short finding title",
      "details": "one concise paragraph explaining the issue and why it matters",
      "path": "relative/path.py or null",
      "line": 123,
      "side": "NEW | OLD | null"
    }}
  ]
}}

Rules:
- Preserve the verdict from the review.
- Omit issues entirely if there are none.
- Set path/line/side to null when the review does not give enough information.
- Do not invent locations.
- Do not wrap the JSON in markdown fences.

Review:
{review}
"""


def convert_review_to_structured(
    client: OpenAI,
    review: str,
    *,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    payload, _ = call_json_schema_completion(
        client,
        system="You convert code reviews into strict JSON. Return valid JSON only.",
        user=build_structuring_prompt(review),
        schema=REVIEW_JSON_SCHEMA,
        max_tokens=3000,
        trace=trace,
        trace_stage="review_structuring",
    )
    if isinstance(payload, dict):
        try:
            return parse_review_payload(json.dumps(payload))
        except Exception:
            pass
    try:
        response = client.chat.completions.create(
            **_build_completion_kwargs(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You convert code reviews into strict JSON. "
                            "Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": build_structuring_prompt(review)},
                ],
                max_tokens=3000,
                sampling_profile=NON_THINKING_INSTRUCT_SAMPLING,
            )
        )
    except Exception:
        return None
    _record_usage(
        trace,
        stage="review_structuring_fallback",
        response_usage=getattr(response, "usage", None),
    )
    if trace is not None:
        trace.setdefault("schema_calls", []).append(
            {
                "stage": "review_structuring_fallback",
                "schema": None,
                "sampling": _sampling_trace(
                    "non_thinking_instruct",
                    NON_THINKING_INSTRUCT_SAMPLING,
                ),
                "max_tokens": 3000,
            }
        )
    raw = strip_thinking_blocks(response.choices[0].message.content or "")
    try:
        return parse_review_payload(raw)
    except Exception:
        return None


def request_review(
    client: OpenAI,
    prompt: str,
    *,
    trace: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any | None, bool]:
    try:
        response = client.chat.completions.create(
            **_build_completion_kwargs(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert code reviewer. Review only the changed lines, "
                            "be specific, and cite file:line for every real issue. "
                            "Do not speculate beyond the diff. Skip style-only, DRY-only, "
                            "docstring/type-hint/test wishlist, or theoretical security/concurrency "
                            "concerns unless the changed code itself makes the bug concrete."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=PRIMARY_REVIEW_MAX_TOKENS,
                sampling_profile=THINKING_PRECISE_CODING_SAMPLING,
            )
        )
    except Exception as exc:
        return _fallback_review(f"API Error: {exc}"), None, True
    primary_usage_bucket = _record_usage(
        trace,
        stage="primary_review",
        response_usage=getattr(response, "usage", None),
    )
    if trace is not None:
        trace["primary_review"] = {
            "sampling": _sampling_trace(
                "thinking_precise_coding",
                THINKING_PRECISE_CODING_SAMPLING,
            ),
            "max_tokens": PRIMARY_REVIEW_MAX_TOKENS,
            "usage": primary_usage_bucket,
        }

    raw_review = response.choices[0].message.content or ""
    cleaned_raw_review = strip_thinking_blocks(raw_review)
    freeform_review: dict[str, Any] | None = None
    try:
        freeform_review = parse_freeform_review(raw_review)
    except Exception:
        freeform_review = None
    review = convert_review_to_structured(client, raw_review, trace=trace)
    if review is None:
        try:
            review = parse_review_payload(raw_review)
        except Exception as exc:
            if freeform_review is not None:
                review = freeform_review
            else:
                preview = strip_thinking_blocks(raw_review).strip()
                if len(preview) > 800:
                    preview = preview[:800].rstrip() + "..."
                return (
                    _fallback_review(
                        f"Reviewer returned invalid structured output: {exc}.\n\nRaw output:\n{preview}"
                    ),
                    getattr(response, "usage", None),
                    True,
                )
    review = merge_review_data(review, freeform_review)
    if is_placeholder_summary(str(review.get("summary") or "")) and cleaned_raw_review:
        lead_summary = extract_lead_summary(cleaned_raw_review)
        if lead_summary:
            review["summary"] = lead_summary
    review["_raw_review"] = cleaned_raw_review
    return review, getattr(response, "usage", None), False


def _format_usage_summary(bucket: dict[str, Any] | None) -> str | None:
    if not bucket:
        return None
    prompt_tokens = int(bucket.get("prompt_tokens", 0))
    output_tokens = int(bucket.get("output_tokens", 0))
    request_count = int(bucket.get("request_count", 0))
    parts = [f"`{prompt_tokens:,}` in / `{output_tokens:,}` out"]
    cache_read = int(bucket.get("cache_read_tokens", 0))
    cache_write = int(bucket.get("cache_write_tokens", 0))
    if cache_read or cache_write:
        parts.append(f"cache `{cache_read:,}` read / `{cache_write:,}` write")
    reasoning = int(bucket.get("reasoning_tokens", 0))
    if reasoning:
        parts.append(f"reasoning `{reasoning:,}`")
    if request_count:
        parts.append(f"`{request_count}` requests")
    return ", ".join(parts)


def format_trace_details(trace: dict[str, Any]) -> str:
    lines = ["<details>", "<summary>Review Trace</summary>", ""]
    lines.append(f"- Model: `{trace.get('model') or MODEL_NAME}`")
    if trace.get("diff_chars") is not None:
        lines.append(f"- Diff chars reviewed: `{int(trace['diff_chars']):,}`")
    if trace.get("estimated_prompt_tokens") is not None:
        budget = trace.get("prompt_token_budget")
        if budget is not None:
            lines.append(
                f"- Estimated prompt tokens: `{int(trace['estimated_prompt_tokens']):,}` / `{int(budget):,}` budget"
            )
        else:
            lines.append(f"- Estimated prompt tokens: `{int(trace['estimated_prompt_tokens']):,}`")
    if trace.get("changed_files_count") is not None:
        lines.append(f"- Changed files: `{int(trace['changed_files_count'])}`")
    if trace.get("truncated"):
        lines.append("- Review context was packed before the primary pass.")
    packed_sections = trace.get("packed_section_count")
    total_sections = trace.get("total_section_count")
    if packed_sections is not None and total_sections is not None:
        lines.append(
            f"- Diff sections kept in full: `{int(packed_sections)}/{int(total_sections)}`"
        )
    primary = trace.get("primary_review")
    if isinstance(primary, dict):
        sampling = primary.get("sampling") or {}
        lines.append(
            "- Primary review sampling: "
            f"`temperature={sampling.get('temperature')}`, "
            f"`top_p={sampling.get('top_p')}`, "
            f"`top_k={sampling.get('top_k')}`, "
            f"`min_p={sampling.get('min_p')}`, "
            f"`presence_penalty={sampling.get('presence_penalty')}`, "
            f"`repetition_penalty={sampling.get('repetition_penalty')}`, "
            f"`thinking={sampling.get('enable_thinking')}`"
        )
        primary_usage_summary = _format_usage_summary(primary.get("usage"))
        if primary_usage_summary:
            lines.append(f"- Primary call usage: {primary_usage_summary}")
    total_usage_summary = _format_usage_summary(trace.get("usage_totals"))
    if total_usage_summary:
        lines.append(f"- Total workflow usage: {total_usage_summary}")
    parallel_info = trace.get("parallel_issue_verification") or {}
    if parallel_info:
        lines.append(
            f"- Issue-specific verification sessions: `{parallel_info.get('issue_count', 0)}` "
            f"(max `{parallel_info.get('max_workers', 0)}` concurrent)"
        )
    included_diff_paths = list(trace.get("included_diff_paths") or [])
    if included_diff_paths:
        lines.append("- Diff sections included for files:")
        lines.extend(f"  - `{path}`" for path in included_diff_paths[:12])
        if len(included_diff_paths) > 12:
            lines.append(f"  - `... and {len(included_diff_paths) - 12} more`")
    targeted_snippet_paths = list(trace.get("targeted_snippet_paths") or [])
    if targeted_snippet_paths:
        lines.append("- Targeted source snippets added for files:")
        lines.extend(f"  - `{path}`" for path in targeted_snippet_paths)
    omitted_without_snippets = list(trace.get("omitted_without_snippets") or [])
    if omitted_without_snippets:
        lines.append("- Omitted from initial review context:")
        lines.extend(f"  - `{path}`" for path in omitted_without_snippets[:12])
        if len(omitted_without_snippets) > 12:
            lines.append(f"  - `... and {len(omitted_without_snippets) - 12} more`")
    schema_calls = list(trace.get("schema_calls") or [])
    if schema_calls:
        lines.append("- Structured passes:")
        for call in schema_calls:
            lines.append(
                "  - "
                f"`{call.get('stage')}` using "
                f"`{call.get('schema') or 'fallback_text'}` with "
                f"`temperature={call.get('sampling', {}).get('temperature')}`, "
                f"`top_p={call.get('sampling', {}).get('top_p')}`, "
                f"`top_k={call.get('sampling', {}).get('top_k')}`, "
                f"`presence_penalty={call.get('sampling', {}).get('presence_penalty')}`, "
                f"`repetition_penalty={call.get('sampling', {}).get('repetition_penalty')}`, "
                f"`thinking={call.get('sampling', {}).get('enable_thinking')}`"
            )
    verification_requests = list(trace.get("verification_requests") or [])
    if verification_requests:
        lines.append("- Verification requests:")
        for req in verification_requests:
            kind = str(req.get("kind") or "")
            if kind == "read_file_snippet":
                path = req.get("path") or "unknown"
                start_line = req.get("start_line")
                end_line = req.get("end_line")
                if start_line is not None or end_line is not None:
                    lines.append(f"  - `read_file_snippet {path}:{start_line or '?'}-{end_line or '?'}`")
                else:
                    lines.append(f"  - `read_file_snippet {path}`")
            elif kind == "search_repo":
                lines.append(f"  - `search_repo {req.get('query')!r}`")
    verification_context = list(trace.get("verification_context") or [])
    if verification_context:
        lines.append("- Verification lookups used:")
        for ctx in verification_context:
            kind = str(ctx.get("kind") or "")
            if kind == "read_file_snippet":
                lines.append(
                    "  - "
                    f"`{ctx.get('path') or 'unknown'}:{ctx.get('start_line') or '?'}-{ctx.get('end_line') or '?'}`"
                )
            elif kind == "search_repo":
                lines.append(
                    "  - "
                    f"`search_repo {ctx.get('query')!r}` "
                    f"({ctx.get('result_lines') or 0} hits, "
                    f"scoped to {len(list(ctx.get('allowed_paths') or []))} files)"
                )
    issue_verifications = list(trace.get("issue_verifications") or [])
    if issue_verifications:
        lines.append("- Issue-specific verification outcomes:")
        for issue_trace in issue_verifications[:10]:
            label = str(issue_trace.get("title") or "Untitled finding")
            status = str(issue_trace.get("status") or "unknown")
            lines.append(f"  - `{status}`: {label}")
        if len(issue_verifications) > 10:
            lines.append(f"  - `... and {len(issue_verifications) - 10} more`")
    if trace.get("final_issue_count") is not None:
        lines.append(f"- Final diff-grounded issues kept: `{int(trace['final_issue_count'])}`")
    lines.extend(["", "</details>"])
    return "\n".join(lines)


def build_comment_body(
    review: dict[str, Any],
    *,
    truncated: bool,
    usage: Any | None,
    trace: dict[str, Any] | None = None,
) -> str:
    rendered = render_markdown_review(review)
    if review.get("verdict") in {"NEEDS_WORK", "CRITICAL_ISSUES"} and not has_actionable_issues(review):
        rendered += (
            "\n\n_Note: no concrete diff-grounded findings were extracted from the model output "
            "for this run._"
        )
    body = f"{AI_REVIEW_MARKER}\n{AI_REVIEW_HEADER}\n\n{rendered}"
    if trace:
        body += f"\n\n{format_trace_details(trace)}"
    if truncated:
        body += "\n\n> ⚠️ Review context was packed to fit the model budget; see the trace for included diff sections and targeted snippets."
    total_usage = trace.get("usage_totals") if trace else None
    primary_usage = trace.get("primary_review", {}).get("usage") if trace else None
    if total_usage:
        footer = (
            f"Total workflow: {_format_usage_summary(total_usage)}"
        )
        if primary_usage and primary_usage != total_usage:
            footer += f" | Primary call: {_format_usage_summary(primary_usage)}"
        body += f"\n\n<sub>Model: Qwen 3.5 397B MoE on aibeast | {footer}</sub>"
    elif usage:
        body += (
            f"\n\n<sub>Model: Qwen 3.5 397B MoE on aibeast | "
            f"Tokens: {usage.prompt_tokens:,} in / {usage.completion_tokens:,} out</sub>"
        )
    return body


def find_existing_review_comment(comments: list[dict[str, Any]]) -> int | None:
    for comment in reversed(comments):
        body = str(comment.get("body") or "")
        if AI_REVIEW_MARKER in body or body.startswith(AI_REVIEW_HEADER):
            comment_id = comment.get("id")
            if comment_id is not None:
                return int(comment_id)
    return None


def find_existing_review_ids(reviews: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for review in reviews:
        body = str(review.get("body") or "")
        if AI_REVIEW_MARKER in body or body.startswith(AI_REVIEW_HEADER):
            review_id = review.get("id")
            if review_id is not None:
                ids.append(int(review_id))
    return ids


def upsert_comment(
    client: httpx.Client,
    *,
    api_base: str,
    repo: str,
    pr_number: str,
    body: str,
) -> int:
    comments_response = client.get(f"{api_base}/repos/{repo}/issues/{pr_number}/comments")
    comments_response.raise_for_status()
    comment_id = find_existing_review_comment(comments_response.json())
    if comment_id is None:
        response = client.post(
            f"{api_base}/repos/{repo}/issues/{pr_number}/comments",
            json={"body": body},
        )
    else:
        response = client.patch(
            f"{api_base}/repos/{repo}/issues/comments/{comment_id}",
            json={"body": body},
        )
    response.raise_for_status()
    return int(response.json()["id"])


def sync_inline_review(
    client: httpx.Client,
    *,
    api_base: str,
    repo: str,
    pr_number: str,
    body: str,
    commit_id: str,
    comments: list[dict[str, Any]],
) -> int | None:
    if not comments:
        return None
    reviews_response = client.get(f"{api_base}/repos/{repo}/pulls/{pr_number}/reviews")
    reviews_response.raise_for_status()
    response = client.post(
        f"{api_base}/repos/{repo}/pulls/{pr_number}/reviews",
        json={
            "body": body,
            "event": "COMMENT",
            "commit_id": commit_id,
            "comments": comments,
        },
    )
    response.raise_for_status()
    review_id = int(response.json()["id"])
    for existing_review_id in find_existing_review_ids(reviews_response.json()):
        if existing_review_id == review_id:
            continue
        try:
            delete_response = client.delete(
                f"{api_base}/repos/{repo}/pulls/{pr_number}/reviews/{existing_review_id}"
            )
            delete_response.raise_for_status()
        except Exception:
            continue
    return review_id


def main() -> int:
    base_sha = os.environ["BASE_SHA"]
    head_sha = os.environ["HEAD_SHA"]

    diff, changed = gather_diff(base_sha, head_sha)
    if not diff.strip():
        print("Empty diff — nothing to review.")
        return 0

    packed_diff, truncated, packing_trace = pack_review_context(diff, changed=changed)
    prompt = build_prompt(changed, packed_diff, "")
    trace: dict[str, Any] = {
        "model": MODEL_NAME,
        "diff_chars": len(diff),
        "changed_files_count": len([line for line in changed.splitlines() if line.strip()]),
        "truncated": truncated,
        **packing_trace,
    }

    print(
        f"Reviewing {len([line for line in changed.splitlines() if line.strip()])} files "
        f"({len(diff):,} chars diff) with Qwen 3.5 397B MoE..."
    )

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        base_url=os.environ.get("OPENAI_BASE_URL", "http://10.15.0.166:8000/v1"),
    )
    review, usage, api_failed = request_review(client, prompt, trace=trace)
    changed_files = [line.strip() for line in changed.splitlines() if line.strip()]
    if not api_failed:
        review = verify_review_with_local_context(
            client,
            review,
            changed_files,
            review_prompt=prompt,
            trace=trace,
        )

    if not str(review.get("summary") or "").strip():
        review = _fallback_review("Reviewer produced no usable output.")
        api_failed = True

    diff_line_index = build_diff_line_index(diff)
    review = normalize_review_after_filter(filter_review_to_diff(review, diff_line_index))
    trace["final_issue_count"] = len(list(review.get("issues") or []))
    inline_comments = build_inline_comments(review, diff_line_index)
    body = build_comment_body(review, truncated=truncated, usage=usage, trace=trace)
    print(body)

    if DRY_RUN:
        print(
            f"\n[dry-run] Gitea comment not posted."
            f" Inline comments prepared: {len(inline_comments)}"
        )
        return 1 if api_failed else 0

    with httpx.Client(
        headers={
            "Authorization": f"token {os.environ['GITEA_TOKEN']}",
            "Content-Type": "application/json",
        },
        timeout=120,
    ) as http:
        comment_id = upsert_comment(
            http,
            api_base=os.environ["GITEA_API"],
            repo=os.environ["REPO"],
            pr_number=os.environ["PR_NUMBER"],
            body=body,
        )
        review_id = sync_inline_review(
            http,
            api_base=os.environ["GITEA_API"],
            repo=os.environ["REPO"],
            pr_number=os.environ["PR_NUMBER"],
            body=build_inline_review_body(review, commit_id=head_sha),
            commit_id=head_sha,
            comments=inline_comments,
        )
    print(f"\nComment upserted (id={comment_id})")
    if review_id is None:
        print("Inline review unchanged (no replacement comments).")
    else:
        print(f"Inline review submitted (id={review_id}, comments={len(inline_comments)})")

    verdict = review["verdict"]
    if api_failed:
        print("Review request failed — failing check.")
        return 1
    if verdict == "CRITICAL_ISSUES":
        print("Critical issues flagged — failing check.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
