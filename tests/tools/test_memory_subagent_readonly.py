#!/usr/bin/env python3
"""
Test read-only memory access for subagents.
"""

import sys
import json
sys.path.insert(0, '/workspace/hermes-agent-fork')

from tools.memory_tool import memory_tool, MemoryStore

def test_subagent_read_only_blocks_writes():
    """Verify subagents in read_only mode cannot write to memory."""
    store = MemoryStore()
    
    # Try to add memory as subagent in read_only mode
    result = memory_tool(
        action="add",
        target="memory",
        content="Test observation from subagent",
        store=store,
        is_subagent=True,
        subagent_memory_mode="read_only"
    )
    
    result_dict = json.loads(result)
    assert result_dict["success"] == False, f"Should block write in read_only mode: {result}"
    assert "read_only" in result_dict["error"], f"Error should mention read_only: {result_dict['error']}"
    print("✅ TEST 1 PASSED: read_only mode blocks memory writes")

def test_subagent_full_allows_writes():
    """Verify subagents in full mode can write to memory."""
    store = MemoryStore()
    
    # Try to add memory as subagent in full mode
    result = memory_tool(
        action="add",
        target="memory",
        content="Test observation from subagent",
        store=store,
        is_subagent=True,
        subagent_memory_mode="full"
    )
    
    result_dict = json.loads(result)
    assert result_dict["success"] == True, f"Should allow write in full mode: {result}"
    print("✅ TEST 2 PASSED: full mode allows memory writes")

def test_normal_agent_always_allows_writes():
    """Verify normal (non-subagent) agents can always write."""
    store = MemoryStore()
    
    # Try to add memory as normal agent (not subagent)
    result = memory_tool(
        action="add",
        target="memory",
        content="Test observation from parent agent",
        store=store,
        is_subagent=False,
        subagent_memory_mode="read_only"  # Even if mode is read_only, non-subagent should work
    )
    
    result_dict = json.loads(result)
    assert result_dict["success"] == True, f"Normal agent should always write: {result}"
    print("✅ TEST 3 PASSED: Normal agents can always write to memory")

def test_subagent_none_mode_blocks_all():
    """Verify subagents in none mode get blocked completely."""
    store = MemoryStore()
    
    # Try to add memory as subagent in none mode
    result = memory_tool(
        action="add",
        target="memory",
        content="Test observation",
        store=store,
        is_subagent=True,
        subagent_memory_mode="none"
    )
    
    result_dict = json.loads(result)
    assert result_dict["success"] == False, f"Should block in none mode: {result}"
    assert "none" in result_dict["error"], f"Error should mention none mode: {result_dict['error']}"
    print("✅ TEST 4 PASSED: none mode blocks all memory access")

if __name__ == "__main__":
    print("="*70)
    print("Testing Read-Only Memory Access for Subagents")
    print("="*70)
    
    try:
        test_subagent_read_only_blocks_writes()
        test_subagent_full_allows_writes()
        test_normal_agent_always_allows_writes()
        test_subagent_none_mode_blocks_all()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nMemory access modes verified:")
        print("  ✅ read_only: Subagents can read but NOT write")
        print("  ✅ full: Subagents have full read/write access")
        print("  ✅ none: Subagents cannot access memory at all")
        print("  ✅ Normal agents: Always have full access")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
