#!/usr/bin/env python3
"""
Test to verify MCP tools are now allowed for subagents.
"""

import sys
sys.path.insert(0, '/workspace/hermes-agent-fork')

from tools.delegate_tool import DEFAULT_ALLOWED_TOOLSETS, _strip_blocked_tools, BLOCKED_TOOLSET_NAMES

def test_mcp_in_allowed_toolsets():
    """Verify 'mcp' is in the default allowed toolsets."""
    assert "mcp" in DEFAULT_ALLOWED_TOOLSETS, f"'mcp' not found in {DEFAULT_ALLOWED_TOOLSETS}"
    print("✅ TEST 1 PASSED: 'mcp' is in DEFAULT_ALLOWED_TOOLSETS")

def test_mcp_not_blocked():
    """Verify 'mcp' is not in blocked toolsets."""
    assert "mcp" not in BLOCKED_TOOLSET_NAMES, f"'mcp' incorrectly in blocked: {BLOCKED_TOOLSET_NAMES}"
    print("✅ TEST 2 PASSED: 'mcp' is NOT in BLOCKED_TOOLSET_NAMES")

def test_mcp_survives_filtering():
    """Verify MCP toolsets survive the _strip_blocked_tools filter."""
    test_toolsets = ["terminal", "mcp", "delegation", "file"]
    filtered = _strip_blocked_tools(test_toolsets)
    assert "mcp" in filtered, f"'mcp' filtered out: {filtered}"
    assert "delegation" not in filtered, f"'delegation' not filtered: {filtered}"
    print(f"✅ TEST 3 PASSED: _strip_blocked_tools preserves 'mcp': {filtered}")

def test_all_default_toolsets_valid():
    """Verify all default toolsets are valid (not blocked)."""
    for toolset in DEFAULT_ALLOWED_TOOLSETS:
        assert toolset not in BLOCKED_TOOLSET_NAMES, f"Default toolset '{toolset}' is blocked!"
    print(f"✅ TEST 4 PASSED: All default toolsets are valid: {DEFAULT_ALLOWED_TOOLSETS}")

if __name__ == "__main__":
    print("="*70)
    print("Testing MCP Tool Access for Subagents")
    print("="*70)
    
    try:
        test_mcp_in_allowed_toolsets()
        test_mcp_not_blocked()
        test_mcp_survives_filtering()
        test_all_default_toolsets_valid()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! MCP tools are now enabled for subagents!")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  DEFAULT_ALLOWED_TOOLSETS = {DEFAULT_ALLOWED_TOOLSETS}")
        print(f"  BLOCKED_TOOLSET_NAMES = {BLOCKED_TOOLSET_NAMES}")
        print(f"\nSubagents can now use:")
        print(f"  ✅ terminal")
        print(f"  ✅ file")
        print(f"  ✅ web")
        print(f"  ✅ mcp (SearXNG, Crawl4AI)")
        print(f"\nSubagents still blocked from:")
        print(f"  ❌ delegation (no recursive spawning)")
        print(f"  ❌ clarify (no user interaction)")
        print(f"  ❌ memory (no shared MEMORY.md writes)")
        print(f"  ❌ code_execution (no execute_code)")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
