"""Test interrupt propagation from parent to child agents.

Reproduces the CLI scenario: user sends a message while delegate_task is
running, main thread calls parent.interrupt(), child should stop.
"""

import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from tools.interrupt import set_interrupt, is_interrupted, _interrupt_event


class TestInterruptPropagationToChild(unittest.TestCase):
    """Verify interrupt propagates from parent to child agent."""

    def setUp(self):
        set_interrupt(False)

    def tearDown(self):
        set_interrupt(False)

    def test_parent_interrupt_sets_child_flag(self):
        """When parent.interrupt() is called, child._interrupt_requested should be set."""
        from run_agent import AIAgent

        parent = AIAgent.__new__(AIAgent)
        parent._interrupt_requested = False
        parent._interrupt_message = None
        parent._interrupt_event = threading.Event()
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        parent.quiet_mode = True

        child = AIAgent.__new__(AIAgent)
        child._interrupt_requested = False
        child._interrupt_message = None
        child._interrupt_event = threading.Event()
        child._active_children = []
        child._active_children_lock = threading.Lock()
        child.quiet_mode = True

        parent._active_children.append(child)

        parent.interrupt("new user message")

        assert parent._interrupt_requested is True
        assert child._interrupt_requested is True
        assert child._interrupt_message == "new user message"
        # Both per-agent events are set; the global fallback remains
        # untouched because per-agent isolation is now in effect.
        assert parent._interrupt_event.is_set() is True
        assert child._interrupt_event.is_set() is True
        assert _interrupt_event.is_set() is False

    def test_child_clear_interrupt_does_not_affect_global(self):
        """child.clear_interrupt() must clear only the child's per-agent event.

        Historical behaviour reset the process-wide ``_interrupt_event``
        too, which silently un-interrupted any other agent running
        concurrently in the same process.  The fix isolates each
        agent's interrupt state.
        """
        from run_agent import AIAgent

        child = AIAgent.__new__(AIAgent)
        child._interrupt_requested = True
        child._interrupt_message = "msg"
        child._interrupt_event = threading.Event()
        child._interrupt_event.set()
        child.quiet_mode = True
        child._active_children = []
        child._active_children_lock = threading.Lock()

        # Independently, the global fallback is set (representing some
        # other code path or an unrelated agent that has not yet been
        # migrated to per-agent events).
        set_interrupt(True)
        assert _interrupt_event.is_set() is True

        # Clearing the child only clears the child's own event.
        child.clear_interrupt()
        assert child._interrupt_requested is False
        assert child._interrupt_event.is_set() is False
        # Global fallback must remain untouched.
        assert _interrupt_event.is_set() is True
        # Manual cleanup so tearDown's set_interrupt(False) is a no-op
        # equivalent.
        set_interrupt(False)

    def test_interrupt_during_child_api_call_detected(self):
        """Interrupt set during _interruptible_api_call is detected within 0.5s."""
        from run_agent import AIAgent

        child = AIAgent.__new__(AIAgent)
        child._interrupt_requested = False
        child._interrupt_message = None
        child._interrupt_event = threading.Event()
        child._active_children = []
        child._active_children_lock = threading.Lock()
        child.quiet_mode = True
        child.api_mode = "chat_completions"
        child.log_prefix = ""
        child._client_kwargs = {"api_key": "test", "base_url": "http://localhost:1234"}

        # Mock a slow API call
        mock_client = MagicMock()
        def slow_api_call(**kwargs):
            time.sleep(5)  # Would take 5s normally
            return MagicMock()
        mock_client.chat.completions.create = slow_api_call
        mock_client.close = MagicMock()
        child.client = mock_client

        # Set interrupt after 0.2s from another thread
        def set_interrupt_later():
            time.sleep(0.2)
            child.interrupt("stop!")
        t = threading.Thread(target=set_interrupt_later, daemon=True)
        t.start()

        start = time.monotonic()
        try:
            child._interruptible_api_call({"model": "test", "messages": []})
            self.fail("Should have raised InterruptedError")
        except InterruptedError:
            elapsed = time.monotonic() - start
            # Should detect within ~0.5s (0.2s delay + 0.3s poll interval)
            assert elapsed < 1.0, f"Took {elapsed:.2f}s to detect interrupt (expected < 1.0s)"
        finally:
            t.join(timeout=2)
            set_interrupt(False)

    def test_concurrent_interrupt_propagation(self):
        """Simulates exact CLI flow: parent runs delegate in thread, main thread interrupts."""
        from run_agent import AIAgent

        parent = AIAgent.__new__(AIAgent)
        parent._interrupt_requested = False
        parent._interrupt_message = None
        parent._interrupt_event = threading.Event()
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        parent.quiet_mode = True

        child = AIAgent.__new__(AIAgent)
        child._interrupt_requested = False
        child._interrupt_message = None
        child._interrupt_event = threading.Event()
        child._active_children = []
        child._active_children_lock = threading.Lock()
        child.quiet_mode = True

        # Register child (simulating what _run_single_child does)
        parent._active_children.append(child)

        # Simulate child running (checking flag in a loop)
        child_detected = threading.Event()
        def simulate_child_loop():
            while not child._interrupt_requested:
                time.sleep(0.05)
            child_detected.set()

        child_thread = threading.Thread(target=simulate_child_loop, daemon=True)
        child_thread.start()

        # Small delay, then interrupt from "main thread"
        time.sleep(0.1)
        parent.interrupt("user typed something new")

        # Child should detect within 200ms
        detected = child_detected.wait(timeout=1.0)
        assert detected, "Child never detected the interrupt!"
        child_thread.join(timeout=1)
        set_interrupt(False)


if __name__ == "__main__":
    unittest.main()
