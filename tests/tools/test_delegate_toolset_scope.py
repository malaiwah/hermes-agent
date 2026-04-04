"""Tests for delegate_tool toolset scoping and delegation profiles."""

from types import SimpleNamespace

from toolsets import TOOLSETS, resolve_toolset
from tools.delegate_tool import (
    _build_child_blocked_tools,
    _build_child_enabled_toolsets,
    _resolve_delegation_profile,
    _strip_blocked_tools,
)


class TestToolsetIntersection:
    """Subagent toolsets must stay within parent + profile scope."""

    def test_strip_blocked_removes_delegation(self):
        child = _strip_blocked_tools(["terminal", "delegation", "clarify", "memory"])
        assert "delegation" not in child
        assert "clarify" not in child
        assert "memory" not in child
        assert "terminal" in child

    def test_requested_toolsets_are_scoped_by_profile_and_blocklist(self):
        parent = SimpleNamespace(enabled_toolsets=["hermes-cli"])
        blocked = _build_child_blocked_tools("none")
        child_toolsets, temp_toolset = _build_child_enabled_toolsets(
            task_index=0,
            requested_toolsets=["terminal", "web", "memory"],
            parent_agent=parent,
            profile_toolsets=["terminal", "web", "memory"],
            blocked_tools=blocked,
        )
        try:
            assert child_toolsets == [temp_toolset]
            resolved = set(resolve_toolset(temp_toolset))
            assert "terminal" in resolved
            assert "web_search" in resolved
            assert "memory" not in resolved
            assert "delegate_task" not in resolved
        finally:
            TOOLSETS.pop(temp_toolset, None)

    def test_requested_toolset_outside_profile_is_dropped(self):
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file", "web"])
        blocked = _build_child_blocked_tools("none")
        _, temp_toolset = _build_child_enabled_toolsets(
            task_index=1,
            requested_toolsets=["web"],
            parent_agent=parent,
            profile_toolsets=["file"],
            blocked_tools=blocked,
        )
        try:
            assert resolve_toolset(temp_toolset) == []
        finally:
            TOOLSETS.pop(temp_toolset, None)

    def test_legacy_default_toolsets_apply_without_profile(self):
        profile = _resolve_delegation_profile({"default_toolsets": ["file", "web"]}, None)
        assert profile["toolsets"] == ["file", "web"]
        assert profile["memory"] == "none"

    def test_privileged_profile_allows_memory_writes(self):
        profile = _resolve_delegation_profile({}, "privileged")
        blocked = _build_child_blocked_tools(profile["memory"])
        assert profile["memory"] == "write"
        assert "memory" not in blocked
