import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_ai_review_module():
    script_path = Path(__file__).resolve().parents[2] / ".gitea" / "scripts" / "ai_review.py"
    spec = importlib.util.spec_from_file_location("gitea_ai_review", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_verdict_reads_explicit_verdict_section():
    module = _load_ai_review_module()

    review = """## Summary

Looks good overall.

## Verdict
CRITICAL_ISSUES
"""

    assert module.extract_verdict(review) == "CRITICAL_ISSUES"


def test_extract_verdict_ignores_critical_heading_without_verdict():
    module = _load_ai_review_module()

    review = """## Summary

## Issues
- 🚨 CRITICAL: none
"""

    assert module.extract_verdict(review) is None


def test_find_existing_review_comment_prefers_latest_marker_or_header():
    module = _load_ai_review_module()

    comments = [
        {"id": 10, "body": "plain user comment"},
        {"id": 11, "body": f"{module.AI_REVIEW_HEADER}\n\nolder review"},
        {"id": 12, "body": f"{module.AI_REVIEW_MARKER}\n{module.AI_REVIEW_HEADER}\n\nnewer review"},
    ]

    assert module.find_existing_review_comment(comments) == 12


def test_request_review_returns_needs_work_on_api_error():
    module = _load_ai_review_module()

    class BrokenClient:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    raise RuntimeError("boom")

    review, usage, api_failed = module.request_review(BrokenClient(), "prompt")

    assert api_failed is True
    assert usage is None
    assert review["verdict"] == "NEEDS_WORK"
    assert "API Error: boom" in review["summary"]


def test_fallback_review_provides_summary_and_needs_work_verdict():
    module = _load_ai_review_module()

    review = module._fallback_review("Reviewer produced no usable output.")

    assert review["summary"] == "Reviewer produced no usable output."
    assert review["verdict"] == "NEEDS_WORK"
    assert review["issues"] == []


def test_convert_review_to_structured_uses_json_schema_response_format():
    module = _load_ai_review_module()
    calls = []

    class Response:
        class Choice:
            class Message:
                content = '{"summary":"ok","verdict":"APPROVED","issues":[]}'

            message = Message()

        choices = [Choice()]

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    return Response()

    parsed = module.convert_review_to_structured(Client(), "## Verdict\nAPPROVED")

    assert parsed["verdict"] == "APPROVED"
    assert calls
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["name"] == "review_payload"
    assert calls[0]["temperature"] == 0.7
    assert calls[0]["top_p"] == 0.8
    assert calls[0]["presence_penalty"] == 1.5
    assert calls[0]["extra_body"]["top_k"] == 20
    assert calls[0]["extra_body"]["repetition_penalty"] == 1.0
    assert calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_request_review_uses_recommended_thinking_coding_sampling():
    module = _load_ai_review_module()
    calls = []

    class Response:
        class Choice:
            class Message:
                content = """## Summary

Looks good.

## Verdict
APPROVED
"""

            message = Message()

        choices = [Choice()]
        usage = None

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    return Response()

    review, _usage, api_failed = module.request_review(Client(), "prompt")

    assert api_failed is False
    assert review["verdict"] == "APPROVED"
    assert calls
    assert calls[0]["temperature"] == 0.6
    assert calls[0]["top_p"] == 0.95
    assert calls[0]["presence_penalty"] == 0.0
    assert calls[0]["max_tokens"] == module.PRIMARY_REVIEW_MAX_TOKENS
    assert calls[0]["extra_body"]["top_k"] == 20
    assert calls[0]["extra_body"]["repetition_penalty"] == 1.0
    assert calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_call_json_schema_completion_accepts_parsed_payload():
    module = _load_ai_review_module()

    class Response:
        class Choice:
            class Message:
                content = None
                parsed = {"summary": "ok"}

            message = Message()

        choices = [Choice()]
        usage = None

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return Response()

    payload, _usage = module.call_json_schema_completion(
        Client(),
        system="x",
        user="y",
        schema={"name": "dummy", "schema": {"type": "object"}, "strict": True},
        max_tokens=10,
    )

    assert payload == {"summary": "ok"}


def test_call_json_schema_completion_records_cache_aware_usage():
    module = _load_ai_review_module()
    trace = {}

    class Response:
        class Choice:
            class Message:
                content = None
                parsed = {"summary": "ok"}

            message = Message()

        choices = [Choice()]
        usage = SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            prompt_tokens_details=SimpleNamespace(cached_tokens=40, cache_write_tokens=10),
        )

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return Response()

    payload, _usage = module.call_json_schema_completion(
        Client(),
        system="x",
        user="y",
        schema={"name": "dummy", "schema": {"type": "object"}, "strict": True},
        max_tokens=10,
        trace=trace,
        trace_stage="schema_test",
    )

    assert payload == {"summary": "ok"}
    assert trace["usage_totals"] == {
        "input_tokens": 70,
        "output_tokens": 30,
        "cache_read_tokens": 40,
        "cache_write_tokens": 10,
        "reasoning_tokens": 0,
        "request_count": 1,
        "prompt_tokens": 120,
        "total_tokens": 150,
    }
    assert trace["usage_by_stage"][0]["stage"] == "schema_test"


def test_estimate_prompt_tokens_uses_tokenize_endpoint(monkeypatch):
    module = _load_ai_review_module()
    module.estimate_prompt_tokens.cache_clear()

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"count": 123}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, json):
            assert url.endswith("/v1/tokenize")
            assert json["model"] == module.MODEL_NAME
            assert json["prompt"] == "hello"
            return FakeResponse()

    monkeypatch.setattr(module.httpx, "Client", lambda timeout=10: FakeClient())

    assert module.estimate_prompt_tokens("hello") == 123


def test_estimate_prompt_tokens_falls_back_to_rough_estimate(monkeypatch):
    module = _load_ai_review_module()
    module.estimate_prompt_tokens.cache_clear()

    class BrokenClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, json):
            raise RuntimeError("boom")

    monkeypatch.setattr(module.httpx, "Client", lambda timeout=10: BrokenClient())
    monkeypatch.setattr(module, "estimate_tokens_rough", lambda text: 77)

    assert module.estimate_prompt_tokens("hello") == 77


def test_request_verification_queries_uses_json_schema_response_format():
    module = _load_ai_review_module()
    calls = []

    class Response:
        class Choice:
            class Message:
                content = '{"requests":[]}'

            message = Message()

        choices = [Choice()]

        usage = None

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    return Response()

    requests = module.request_verification_queries(
        Client(),
        {"issues": [{"severity": "WARNING", "title": "x", "details": "y"}]},
        ["app.py"],
    )

    assert requests == []
    assert calls
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["name"] == "review_verification_requests"


def test_read_file_snippet_respects_repo_root(tmp_path, monkeypatch):
    module = _load_ai_review_module()
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir(parents=True)
    target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    outside = tmp_path.parent / "outside.py"
    outside.write_text("oops\n", encoding="utf-8")

    snippet = module.read_file_snippet("src/app.py", start_line=2, end_line=3)

    assert snippet is not None
    assert "src/app.py:2-3" in snippet
    assert "2: two" in snippet
    assert module.read_file_snippet("../outside.py") is None


def test_collect_verification_context_limits_requests(monkeypatch):
    module = _load_ai_review_module()

    monkeypatch.setattr(module, "read_file_snippet", lambda *args, **kwargs: "snippet")
    search_calls = []

    def _search_repo(query, **kwargs):
        search_calls.append((query, kwargs))
        return "match"

    monkeypatch.setattr(module, "search_repo", _search_repo)
    monkeypatch.setattr(module, "resolve_repo_file", lambda path: Path(path))

    review = {
        "issues": [
            {"path": "app.py", "line": 10, "title": "one", "details": "x"},
            {"path": "other.py", "line": 5, "title": "two", "details": "y"},
        ]
    }
    requests = [
        {"issue_index": 0, "kind": "read_file_snippet", "path": "app.py", "start_line": 1, "end_line": 5},
        {"issue_index": 0, "kind": "read_file_snippet", "path": "forbidden.py", "start_line": 1, "end_line": 5},
        {"issue_index": 1, "kind": "search_repo", "query": "symbol_name"},
    ]

    contexts = module.collect_verification_context(review, ["app.py"], requests)

    assert len(contexts) == 2
    assert contexts[0]["kind"] == "read_file_snippet"
    assert contexts[1]["kind"] == "search_repo"
    assert search_calls == [("symbol_name", {"allowed_paths": ["app.py"]})]


def test_verify_review_with_local_context_uses_verified_review(monkeypatch):
    module = _load_ai_review_module()

    review = {
        "summary": "Initial summary",
        "verdict": "NEEDS_WORK",
        "issues": [
            {"severity": "WARNING", "title": "x", "details": "y", "path": "app.py", "line": 4, "side": None},
            {"severity": "WARNING", "title": "z", "details": "w", "path": "other.py", "line": 7, "side": None},
        ],
    }

    def _fake_verify_issue(**kwargs):
        if kwargs["issue_index"] == 0:
            return {
                "issue_index": 0,
                "status": "confirmed",
                "summary": "Confirmed x.",
                "review": {
                    "summary": "Confirmed x.",
                    "verdict": "NEEDS_WORK",
                    "issues": [kwargs["issue"]],
                },
                "trace": {
                    "issue_index": 0,
                    "title": "x",
                    "status": "confirmed",
                    "verification_requests": [{"kind": "read_file_snippet", "path": "app.py"}],
                    "verification_context": [{"kind": "read_file_snippet", "path": "app.py"}],
                    "usage_totals": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_read_tokens": 4,
                        "cache_write_tokens": 0,
                        "reasoning_tokens": 1,
                        "request_count": 2,
                        "prompt_tokens": 14,
                        "total_tokens": 19,
                    },
                    "usage_by_stage": [
                        {
                            "stage": "issue_0_verification_review",
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "cache_read_tokens": 4,
                            "cache_write_tokens": 0,
                            "reasoning_tokens": 1,
                            "request_count": 2,
                            "prompt_tokens": 14,
                            "total_tokens": 19,
                        }
                    ],
                },
            }
        return {
            "issue_index": 1,
            "status": "rejected",
            "summary": "Rejected z.",
            "review": {"summary": "Rejected z.", "verdict": "APPROVED", "issues": []},
            "trace": {
                "issue_index": 1,
                "title": "z",
                "status": "rejected",
                "verification_requests": [],
                "verification_context": [],
                "usage_totals": {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "reasoning_tokens": 0,
                    "request_count": 1,
                    "prompt_tokens": 3,
                    "total_tokens": 5,
                },
                "usage_by_stage": [
                    {
                        "stage": "issue_1_verification_review",
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                        "reasoning_tokens": 0,
                        "request_count": 1,
                        "prompt_tokens": 3,
                        "total_tokens": 5,
                    }
                ],
            },
        }

    monkeypatch.setattr(module, "verify_single_issue_with_local_context", _fake_verify_issue)

    trace = {}
    verified = module.verify_review_with_local_context(
        object(),
        review,
        ["app.py", "other.py"],
        review_prompt="prompt",
        trace=trace,
    )

    assert verified["summary"] == "Issue-specific verification confirmed 1 primary findings and rejected 1 others."
    assert [issue["title"] for issue in verified["issues"]] == ["x"]
    assert trace["parallel_issue_verification"]["issue_count"] == 2
    assert len(trace["issue_verifications"]) == 2
    assert trace["usage_totals"]["prompt_tokens"] == 17
    assert trace["usage_totals"]["output_tokens"] == 7
    assert trace["usage_totals"]["cache_read_tokens"] == 4
    assert len(trace["usage_by_stage"]) == 2


def test_parse_review_payload_accepts_json_with_fenced_wrapper():
    module = _load_ai_review_module()

    payload = """```json
{
  "summary": "Looks mostly good.",
  "verdict": "NEEDS_WORK",
  "issues": [
    {
      "severity": "warning",
      "title": "Missing test",
      "details": "The edge case is not covered.",
      "path": "tests/example.py",
      "line": 12,
      "side": "new"
    }
  ]
}
```"""

    parsed = module.parse_review_payload(payload)

    assert parsed["verdict"] == "NEEDS_WORK"
    assert parsed["issues"][0]["severity"] == "WARNING"
    assert parsed["issues"][0]["side"] == "NEW"


def test_build_inline_comments_filters_to_changed_lines_and_dedupes():
    module = _load_ai_review_module()

    diff = """diff --git a/app.py b/app.py
index 1111111..2222222 100644
--- a/app.py
+++ b/app.py
@@ -10,3 +10,4 @@ def handler():
     old = True
-    value = 1
+    value = 2
+    extra = 3
     return value
"""
    review = {
        "summary": "Needs work.",
        "verdict": "NEEDS_WORK",
        "issues": [
            {
                "severity": "WARNING",
                "title": "Changed line issue",
                "details": "This line changed.",
                "path": "app.py",
                "line": 11,
                "side": "NEW",
            },
            {
                "severity": "WARNING",
                "title": "Changed line issue",
                "details": "This line changed.",
                "path": "app.py",
                "line": 11,
                "side": "NEW",
            },
            {
                "severity": "WARNING",
                "title": "Unchanged line",
                "details": "Should be ignored.",
                "path": "app.py",
                "line": 10,
                "side": "NEW",
            },
        ],
    }

    comments = module.build_inline_comments(review, module.build_diff_line_index(diff))

    assert len(comments) == 1
    assert comments[0]["path"] == "app.py"
    assert comments[0]["new_position"] == 11
    assert comments[0]["old_position"] == 0


def test_parse_freeform_review_extracts_summary_verdict_and_location():
    module = _load_ai_review_module()

    review = """## Summary

Adds inline review support.

### 🔴 Critical Issues

#### 1. **Bad anchor** — `gateway/run.py:123-125`
This can break session isolation.

## Verdict
CRITICAL_ISSUES
"""

    parsed = module.parse_freeform_review(review)

    assert parsed["summary"] == "Adds inline review support."
    assert parsed["verdict"] == "CRITICAL_ISSUES"
    assert parsed["issues"][0]["severity"] == "CRITICAL"
    assert parsed["issues"][0]["path"] == "gateway/run.py"
    assert parsed["issues"][0]["line"] == 123


def test_parse_freeform_review_extracts_location_from_file_line():
    module = _load_ai_review_module()

    review = """## Summary

No summary.

### ⚠️ Warning Issues

#### 1. **Anchor is in details**
**File:** `gateway/platforms/base.py:684-685`

This should still anchor correctly.

## Verdict
NEEDS_WORK
"""

    parsed = module.parse_freeform_review(review)

    assert parsed["issues"][0]["path"] == "gateway/platforms/base.py"
    assert parsed["issues"][0]["line"] == 684
    assert "**File:**" not in parsed["issues"][0]["details"]


def test_extract_issue_location_falls_back_to_title_and_details():
    module = _load_ai_review_module()

    issue = {
        "severity": "CRITICAL",
        "title": "gateway/run.py:2940-2944",
        "details": "See also `gateway/platforms/base.py:684-685`.",
        "path": None,
        "line": None,
        "side": None,
    }

    assert module.extract_issue_location(issue) == ("gateway/run.py", 2940, "title")


def test_parse_freeform_review_supports_executive_summary_and_level3_issue_headings():
    module = _load_ai_review_module()

    review = """# Code Review

## Executive Summary

This adds platform context support.

## Critical Issues

### 1. **Path traversal risk** `.gitea/scripts/ai_review.py:113-118`
The path normalization does not reject `..` segments.

## Major Issues

### 2. **Config validation gap** `gateway/config.py:25-34`
The default and max limits can drift.

## Verdict
NEEDS_WORK
"""

    parsed = module.parse_freeform_review(review)

    assert parsed["summary"] == "This adds platform context support."
    assert [issue["severity"] for issue in parsed["issues"]] == ["CRITICAL", "WARNING"]
    assert parsed["issues"][0]["path"] == ".gitea/scripts/ai_review.py"
    assert parsed["issues"][0]["line"] == 113


def test_build_inline_comments_infers_side_when_missing():
    module = _load_ai_review_module()

    diff = """diff --git a/app.py b/app.py
index 1111111..2222222 100644
--- a/app.py
+++ b/app.py
@@ -4,3 +4,3 @@ def handler():
-    old = 1
+    old = 2
     return old
"""
    review = {
        "summary": "Needs work.",
        "verdict": "NEEDS_WORK",
        "issues": [
            {
                "severity": "WARNING",
                "title": "Inferred side",
                "details": "Side should be inferred from the diff.",
                "path": "app.py",
                "line": 4,
                "side": None,
            }
        ],
    }

    comments = module.build_inline_comments(review, module.build_diff_line_index(diff))

    assert comments == []


def test_is_placeholder_summary_detects_scorecard_table():
    module = _load_ai_review_module()

    summary = """| Severity | Count |
|----------|-------|
| CRITICAL | 4 |"""

    assert module.is_placeholder_summary(summary) is True


def test_build_inline_comments_skips_suggestions_and_nonlocal_findings():
    module = _load_ai_review_module()

    diff = """diff --git a/app.py b/app.py
index 1111111..2222222 100644
--- a/app.py
+++ b/app.py
@@ -4,3 +4,3 @@ def handler():
-    old = 1
+    old = 2
     return old
"""
    review = {
        "summary": "Needs work.",
        "verdict": "NEEDS_WORK",
        "issues": [
            {
                "severity": "SUGGESTION",
                "title": "Style nit",
                "details": "Consider renaming this.",
                "path": "app.py",
                "line": 4,
                "side": "NEW",
            },
            {
                "severity": "WARNING",
                "title": "Non-local audit",
                "details": "This pattern is correct, but review all similar patterns elsewhere.",
                "path": "app.py",
                "line": 4,
                "side": "NEW",
            },
            {
                "severity": "WARNING",
                "title": "Real issue",
                "details": "The new value skips validation.",
                "path": "app.py",
                "line": 4,
                "side": "NEW",
            },
        ],
    }

    comments = module.build_inline_comments(review, module.build_diff_line_index(diff))

    assert len(comments) == 1
    assert "Real issue" in comments[0]["body"]


def test_build_inline_comments_skips_details_only_locations():
    module = _load_ai_review_module()

    diff = """diff --git a/app.py b/app.py
index 1111111..2222222 100644
--- a/app.py
+++ b/app.py
@@ -4,3 +4,3 @@ def handler():
-    old = 1
+    old = 2
     return old
"""
    review = {
        "summary": "Needs work.",
        "verdict": "NEEDS_WORK",
        "issues": [
            {
                "severity": "WARNING",
                "title": "Cross-file concern",
                "details": "**Files:** `app.py:4` and `other.py:9`",
                "path": "app.py",
                "line": 4,
                "side": "NEW",
                "location_source": "details",
            }
        ],
    }

    comments = module.build_inline_comments(review, module.build_diff_line_index(diff))

    assert comments == []


def test_build_inline_comments_skips_old_side_only_locations():
    module = _load_ai_review_module()

    diff = """diff --git a/app.py b/app.py
index 1111111..2222222 100644
--- a/app.py
+++ b/app.py
@@ -4,3 +4,2 @@ def handler():
-    removed = 1
     return old
"""
    review = {
        "summary": "Needs work.",
        "verdict": "NEEDS_WORK",
        "issues": [
            {
                "severity": "WARNING",
                "title": "Removed line issue",
                "details": "This used to exist.",
                "path": "app.py",
                "line": 4,
                "side": "OLD",
                "location_source": "field",
            }
        ],
    }

    comments = module.build_inline_comments(review, module.build_diff_line_index(diff))

    assert comments == []


def test_merge_review_data_uses_freeform_summary_fallback():
    module = _load_ai_review_module()

    structured = {
        "summary": "No summary provided.",
        "verdict": "NEEDS_WORK",
        "issues": [],
    }
    freeform = {
        "summary": "This adds inline review support.",
        "verdict": "NEEDS_WORK",
        "issues": [{"severity": "WARNING", "title": "x", "details": "y"}],
    }

    merged = module.merge_review_data(structured, freeform)

    assert merged["summary"] == "This adds inline review support."
    assert merged["issues"] == freeform["issues"]


def test_has_actionable_issues_ignores_placeholder_summary_issue():
    module = _load_ai_review_module()

    review = {
        "summary": "No summary provided.",
        "verdict": "NEEDS_WORK",
        "issues": [
            {
                "severity": "SUGGESTION",
                "title": "Summary",
                "details": "| Severity | Count |\n|----------|-------|",
            }
        ],
    }

    assert module.has_actionable_issues(review) is False


def test_build_comment_body_notes_missing_grounded_issues_when_needs_work_has_no_actionable_issues():
    module = _load_ai_review_module()

    review = {
        "summary": "No summary provided.",
        "verdict": "NEEDS_WORK",
        "issues": [
            {
                "severity": "SUGGESTION",
                "title": "Summary",
                "details": "| Severity | Count |\n|----------|-------|",
            }
        ],
        "_raw_review": "## Summary\n\nReal raw review body.\n\n## Verdict\nNEEDS_WORK",
    }

    body = module.build_comment_body(review, truncated=False, usage=None)

    assert "Real raw review body." not in body
    assert "| Severity | Count |" not in body
    assert "no concrete diff-grounded findings were extracted" in body


def test_build_comment_body_includes_collapsed_trace_section():
    module = _load_ai_review_module()

    review = {
        "summary": "Looks good.",
        "verdict": "APPROVED",
        "issues": [],
    }
    trace = {
        "model": module.MODEL_NAME,
        "diff_chars": 1234,
        "estimated_prompt_tokens": 321,
        "prompt_token_budget": 300000,
        "changed_files_count": 2,
        "truncated": False,
        "included_diff_paths": ["gateway/run.py", ".gitea/scripts/ai_review.py"],
        "targeted_snippet_paths": ["gateway/session.py"],
        "packed_section_count": 2,
        "total_section_count": 3,
        "primary_review": {
            "sampling": module._sampling_trace(
                "thinking_precise_coding",
                module.THINKING_PRECISE_CODING_SAMPLING,
            ),
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_tokens": 30,
                "cache_write_tokens": 5,
                "reasoning_tokens": 7,
                "request_count": 1,
                "prompt_tokens": 135,
                "total_tokens": 155,
            },
        },
        "usage_totals": {
            "input_tokens": 150,
            "output_tokens": 40,
            "cache_read_tokens": 60,
            "cache_write_tokens": 5,
            "reasoning_tokens": 7,
            "request_count": 3,
            "prompt_tokens": 215,
            "total_tokens": 255,
        },
        "schema_calls": [
            {
                "stage": "review_structuring",
                "schema": "review_payload",
                "sampling": module._sampling_trace(
                    "non_thinking_instruct",
                    module.NON_THINKING_INSTRUCT_SAMPLING,
                ),
            }
        ],
        "verification_requests": [
            {
                "kind": "read_file_snippet",
                "path": "gateway/run.py",
                "start_line": 7078,
                "end_line": 7095,
            },
            {"kind": "search_repo", "query": "history_message_id"},
        ],
        "verification_context": [
            {
                "kind": "read_file_snippet",
                "path": "gateway/run.py",
                "start_line": 7078,
                "end_line": 7095,
            },
            {
                "kind": "search_repo",
                "query": "history_message_id",
                "allowed_paths": ["gateway/run.py"],
                "result_lines": 2,
            },
        ],
        "final_issue_count": 0,
    }

    body = module.build_comment_body(review, truncated=False, usage=None, trace=trace)

    assert "<details>" in body
    assert "<summary>Review Trace</summary>" in body
    assert "Primary review sampling" in body
    assert "Estimated prompt tokens" in body
    assert "Diff sections kept in full" in body
    assert "Targeted source snippets added for files" in body
    assert "Primary call usage" in body
    assert "Total workflow usage" in body
    assert "gateway/run.py:7078-7095" in body
    assert "search_repo 'history_message_id'" in body


def test_build_comment_body_footer_prefers_total_workflow_usage_and_cache():
    module = _load_ai_review_module()

    review = {
        "summary": "Looks good.",
        "verdict": "APPROVED",
        "issues": [],
    }
    trace = {
        "primary_review": {
            "usage": {
                "input_tokens": 70,
                "output_tokens": 20,
                "cache_read_tokens": 10,
                "cache_write_tokens": 0,
                "reasoning_tokens": 5,
                "request_count": 1,
                "prompt_tokens": 80,
                "total_tokens": 100,
            }
        },
        "usage_totals": {
            "input_tokens": 120,
            "output_tokens": 40,
            "cache_read_tokens": 30,
            "cache_write_tokens": 10,
            "reasoning_tokens": 6,
            "request_count": 4,
            "prompt_tokens": 160,
            "total_tokens": 200,
        },
    }

    body = module.build_comment_body(
        review,
        truncated=False,
        usage=SimpleNamespace(prompt_tokens=80, completion_tokens=20),
        trace=trace,
    )

    assert "Total workflow: `160` in / `40` out, cache `30` read / `10` write, reasoning `6`, `4` requests" in body
    assert "Primary call: `80` in / `20` out, cache `10` read / `0` write, reasoning `5`, `1` requests" in body
    assert "Tokens: 80 in / 20 out" not in body


def test_pack_review_context_keeps_later_sections_that_fit(monkeypatch):
    module = _load_ai_review_module()
    monkeypatch.setattr(module, "read_file_snippet", lambda *args, **kwargs: None)

    diff = """diff --git a/big.py b/big.py
+++ b/big.py
@@ -1,1 +1,1 @@
-a
+b
{big_body}
diff --git a/small.py b/small.py
+++ b/small.py
@@ -10,1 +10,1 @@
-x
+y
""".format(big_body="\n".join("+line" for _ in range(200)))

    packed, truncated, trace = module.pack_review_context(
        diff,
        changed="big.py\nsmall.py",
        max_prompt_tokens=420,
    )

    assert truncated is True
    assert "diff --git a/small.py b/small.py" in packed
    assert "small.py" in trace["included_diff_paths"]
    assert trace["estimated_prompt_tokens"] <= trace["prompt_token_budget"]


def test_pack_review_context_adds_targeted_snippets_for_omitted_sections(monkeypatch):
    module = _load_ai_review_module()

    def _fake_snippet(path, **kwargs):
        return f"{path}:{kwargs.get('start_line')}-{kwargs.get('end_line')}\ncontext"

    monkeypatch.setattr(module, "read_file_snippet", _fake_snippet)

    diff = """diff --git a/first.py b/first.py
+++ b/first.py
@@ -1,1 +1,1 @@
-a
+b
{big_body}
diff --git a/second.py b/second.py
+++ b/second.py
@@ -20,2 +20,2 @@
-x
+y
""".format(big_body="\n".join("+line" for _ in range(250)))

    packed, truncated, trace = module.pack_review_context(
        diff,
        changed="first.py\nsecond.py",
        max_prompt_tokens=520,
        max_snippet_files=2,
    )

    assert truncated is True
    assert "Targeted source snippets for diff sections omitted from the initial patch context" in packed
    assert "second.py" in trace["targeted_snippet_paths"] or "first.py" in trace["targeted_snippet_paths"]


def test_extract_lead_summary_uses_first_plain_paragraph():
    module = _load_ai_review_module()

    review = """# Code Review

This is a substantial PR with actionable issues.

## Critical Issues

### 1. **Problem** `app.py:10`
Details.
"""

    assert module.extract_lead_summary(review) == "This is a substantial PR with actionable issues."


def test_parse_freeform_review_extracts_bullet_style_issues():
    module = _load_ai_review_module()

    review = """## Summary

This refines the AI review workflow.

## Issues
- 🚨 CRITICAL: `.gitea/scripts/ai_review.py:659` **Potential path traversal in `_normalize_path`** —
The normalization accepts `../` segments unchanged.

---
- ⚠️ WARNING: `gateway/runtime_bridge.py:16` **Global state read without lock** —
Reads happen outside the lock.

## Verdict
NEEDS_WORK
"""

    parsed = module.parse_freeform_review(review)

    assert parsed["summary"] == "This refines the AI review workflow."
    assert [issue["severity"] for issue in parsed["issues"]] == ["CRITICAL", "WARNING"]
    assert parsed["issues"][0]["path"] == ".gitea/scripts/ai_review.py"
    assert parsed["issues"][0]["line"] == 659
    assert "The normalization accepts" in parsed["issues"][0]["details"]


def test_filter_review_to_diff_drops_ungrounded_issues():
    module = _load_ai_review_module()

    diff = """diff --git a/app.py b/app.py
index 1111111..2222222 100644
--- a/app.py
+++ b/app.py
@@ -4,3 +4,3 @@ def handler():
-    old = 1
+    old = 2
     return old
"""
    review = {
        "summary": "Needs work.",
        "verdict": "NEEDS_WORK",
        "issues": [
            {
                "severity": "WARNING",
                "title": "Changed line issue",
                "details": "Grounded in the diff.",
                "path": "app.py",
                "line": 4,
                "side": "NEW",
            },
            {
                "severity": "CRITICAL",
                "title": "Outside hunk",
                "details": "Not grounded in the changed lines.",
                "path": "app.py",
                "line": 99,
                "side": "NEW",
            },
            {
                "severity": "WARNING",
                "title": "Wrong file",
                "details": "Touches a file outside the diff.",
                "path": "other.py",
                "line": 4,
                "side": "NEW",
            },
        ],
    }

    filtered = module.filter_review_to_diff(review, module.build_diff_line_index(diff))

    assert len(filtered["issues"]) == 1
    assert filtered["issues"][0]["title"] == "Changed line issue"


def test_filter_review_to_diff_drops_rejected_low_value_issues():
    module = _load_ai_review_module()

    diff = """diff --git a/app.py b/app.py
index 1111111..2222222 100644
--- a/app.py
+++ b/app.py
@@ -4,3 +4,3 @@ def handler():
-    old = 1
+    old = 2
     return old
"""
    review = {
        "summary": "Needs work.",
        "verdict": "NEEDS_WORK",
        "issues": [
            {
                "severity": "WARNING",
                "title": "Missing docstring",
                "details": "This helper needs a docstring.",
                "path": "app.py",
                "line": 4,
                "side": "NEW",
            },
            {
                "severity": "WARNING",
                "title": "Concrete issue",
                "details": "The new value bypasses validation.",
                "path": "app.py",
                "line": 4,
                "side": "NEW",
            },
        ],
    }

    filtered = module.filter_review_to_diff(review, module.build_diff_line_index(diff))

    assert len(filtered["issues"]) == 1
    assert filtered["issues"][0]["title"] == "Concrete issue"


def test_normalize_review_after_filter_approves_empty_grounded_review():
    module = _load_ai_review_module()

    review = {
        "summary": "This PR is substantial.",
        "verdict": "NEEDS_WORK",
        "issues": [],
    }

    normalized = module.normalize_review_after_filter(review)

    assert normalized["verdict"] == "APPROVED"
    assert normalized["summary"] == "This PR is substantial."


def test_normalize_review_after_filter_downgrades_to_needs_work_without_critical_findings():
    module = _load_ai_review_module()

    review = {
        "summary": "There is one warning.",
        "verdict": "CRITICAL_ISSUES",
        "issues": [
            {
                "severity": "WARNING",
                "title": "Concrete warning",
                "details": "A real non-critical issue remains.",
                "path": "app.py",
                "line": 4,
                "side": "NEW",
            }
        ],
    }

    normalized = module.normalize_review_after_filter(review)

    assert normalized["verdict"] == "NEEDS_WORK"


def test_normalize_review_after_filter_preserves_critical_when_present():
    module = _load_ai_review_module()

    review = {
        "summary": "There is one critical finding.",
        "verdict": "APPROVED",
        "issues": [
            {
                "severity": "CRITICAL",
                "title": "Concrete critical",
                "details": "A real critical issue remains.",
                "path": "app.py",
                "line": 4,
                "side": "NEW",
            }
        ],
    }

    normalized = module.normalize_review_after_filter(review)

    assert normalized["verdict"] == "CRITICAL_ISSUES"


def test_find_existing_review_ids_returns_matching_reviews():
    module = _load_ai_review_module()

    reviews = [
        {"id": 40, "body": "plain user review"},
        {"id": 41, "body": f"{module.AI_REVIEW_HEADER}\n\nolder inline review"},
        {"id": 42, "body": f"{module.AI_REVIEW_MARKER}\n{module.AI_REVIEW_HEADER}\n\nnewer inline review"},
    ]

    assert module.find_existing_review_ids(reviews) == [41, 42]


def test_sync_inline_review_replaces_prior_bot_reviews_and_posts_new_one():
    module = _load_ai_review_module()

    class FakeResponse:
        def __init__(self, payload=None):
            self._payload = payload or {}

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self):
            self.deleted = []
            self.posted = None

        def get(self, url):
            assert url.endswith("/repos/angelos/hermes-agent/pulls/18/reviews")
            return FakeResponse(
                [
                    {"id": 51, "body": f"{module.AI_REVIEW_MARKER}\nold review"},
                    {"id": 52, "body": "human review"},
                ]
            )

        def delete(self, url):
            self.deleted.append(url)
            return FakeResponse()

        def post(self, url, json):
            self.posted = (url, json)
            return FakeResponse({"id": 77})

    client = FakeClient()
    review_id = module.sync_inline_review(
        client,
        api_base="http://gitea/api/v1",
        repo="angelos/hermes-agent",
        pr_number="18",
        body="inline review body",
        commit_id="abc123",
        comments=[
            {
                "path": "app.py",
                "body": "Issue",
                "new_position": 11,
                "old_position": 0,
            }
        ],
    )

    assert review_id == 77
    assert client.deleted == [
        "http://gitea/api/v1/repos/angelos/hermes-agent/pulls/18/reviews/51"
    ]
    assert client.posted[0].endswith("/repos/angelos/hermes-agent/pulls/18/reviews")
    assert client.posted[1]["event"] == "COMMENT"
    assert client.posted[1]["commit_id"] == "abc123"


def test_sync_inline_review_keeps_existing_reviews_when_no_replacement_comments():
    module = _load_ai_review_module()

    class FakeResponse:
        def __init__(self, payload=None):
            self._payload = payload or {}

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self):
            self.get_called = False
            self.deleted = []
            self.posted = []

        def get(self, url):
            self.get_called = True
            return FakeResponse([])

        def delete(self, url):
            self.deleted.append(url)
            return FakeResponse()

        def post(self, url, json):
            self.posted.append((url, json))
            return FakeResponse({"id": 77})

    client = FakeClient()
    review_id = module.sync_inline_review(
        client,
        api_base="http://gitea/api/v1",
        repo="angelos/hermes-agent",
        pr_number="18",
        body="inline review body",
        commit_id="abc123",
        comments=[],
    )

    assert review_id is None
    assert client.get_called is False
    assert client.deleted == []
    assert client.posted == []


def test_sync_inline_review_preserves_existing_reviews_when_post_fails():
    module = _load_ai_review_module()

    class FakeResponse:
        def __init__(self, payload=None, should_raise=False):
            self._payload = payload or {}
            self._should_raise = should_raise

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self._should_raise:
                raise RuntimeError("post failed")
            return None

    class FakeClient:
        def __init__(self):
            self.deleted = []

        def get(self, url):
            return FakeResponse(
                [
                    {"id": 51, "body": f"{module.AI_REVIEW_MARKER}\nold review"},
                    {"id": 52, "body": "human review"},
                ]
            )

        def delete(self, url):
            self.deleted.append(url)
            return FakeResponse()

        def post(self, url, json):
            return FakeResponse({"id": 77}, should_raise=True)

    client = FakeClient()
    try:
        module.sync_inline_review(
            client,
            api_base="http://gitea/api/v1",
            repo="angelos/hermes-agent",
            pr_number="18",
            body="inline review body",
            commit_id="abc123",
            comments=[
                {
                    "path": "app.py",
                    "body": "Issue",
                    "new_position": 11,
                    "old_position": 0,
                }
            ],
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected sync_inline_review to propagate post failure")

    assert client.deleted == []
