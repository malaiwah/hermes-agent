"""Regression tests for cross-agent interrupt isolation.

Two AIAgent instances running concurrently in the same process must not
share interrupt state. The historical implementation used a single
process-wide ``threading.Event`` in ``tools/interrupt.py``, which caused
one agent's ``interrupt()`` (e.g. an SSE client disconnecting on the API
server) to abort in-flight tool calls in every other concurrently-running
agent — producing truncated mid-task completions.

These tests pin the per-agent isolation contract:

1. Each ``AIAgent`` owns its own ``threading.Event``.
2. ``run_conversation()`` binds that event to a context variable so tools
   called via ``is_interrupted()`` observe only the current agent's signal.
3. ``ThreadPoolExecutor`` workers spawned by concurrent tool execution
   inherit the bound context.
4. When no event is bound (legacy CLI / test usage),
   ``is_interrupted()`` and ``set_interrupt()`` fall through to a
   module-level fallback so the existing single-agent API keeps working.

Run with:
    python -m pytest tests/tools/test_interrupt_isolation.py -v
"""

import concurrent.futures
import contextvars
import threading
import time

import pytest

from tools.interrupt import (
    _interrupt_event,
    bind_event,
    is_interrupted,
    set_interrupt,
    unbind_event,
)


@pytest.fixture(autouse=True)
def _reset_global_event():
    """Each test starts and ends with a clean global fallback event."""
    _interrupt_event.clear()
    yield
    _interrupt_event.clear()


class TestPerAgentBinding:
    """Direct tests of the bind_event / unbind_event API."""

    def test_bound_event_isolated_from_global(self):
        agent_event = threading.Event()
        token = bind_event(agent_event)
        try:
            assert not is_interrupted()
            agent_event.set()
            assert is_interrupted()
            # The module-level fallback must remain untouched.
            assert not _interrupt_event.is_set()
        finally:
            unbind_event(token)

    def test_unbind_restores_global_fallback(self):
        agent_event = threading.Event()
        token = bind_event(agent_event)
        unbind_event(token)
        # After unbinding, is_interrupted() reflects the global event.
        assert not is_interrupted()
        _interrupt_event.set()
        assert is_interrupted()

    def test_set_interrupt_writes_to_bound_event(self):
        agent_event = threading.Event()
        token = bind_event(agent_event)
        try:
            set_interrupt(True)
            assert agent_event.is_set()
            assert not _interrupt_event.is_set()
            set_interrupt(False)
            assert not agent_event.is_set()
        finally:
            unbind_event(token)

    def test_set_interrupt_falls_back_to_global_when_unbound(self):
        # No bind_event() in this context.
        set_interrupt(True)
        assert _interrupt_event.is_set()
        set_interrupt(False)
        assert not _interrupt_event.is_set()


class TestCrossContextIsolation:
    """Two independent contexts must not see each other's interrupts.

    These reproduce the API server scenario where two concurrent
    /v1/chat/completions requests run agents simultaneously and one
    client disconnects.
    """

    def test_two_bound_events_independent(self):
        event_a = threading.Event()
        event_b = threading.Event()

        results = {}

        def _agent_a():
            token = bind_event(event_a)
            try:
                # A is interrupted externally.
                event_a.set()
                results["a_sees_interrupt"] = is_interrupted()
            finally:
                unbind_event(token)

        def _agent_b():
            token = bind_event(event_b)
            try:
                # B should NOT observe A's interrupt.
                results["b_sees_interrupt"] = is_interrupted()
            finally:
                unbind_event(token)

        # Run each in its own contextvars.copy_context() so the bindings
        # cannot leak between threads via shared context references.
        ctx_a = contextvars.copy_context()
        ctx_b = contextvars.copy_context()
        ta = threading.Thread(target=ctx_a.run, args=(_agent_a,))
        tb = threading.Thread(target=ctx_b.run, args=(_agent_b,))
        ta.start()
        ta.join(timeout=2)
        tb.start()
        tb.join(timeout=2)

        assert results["a_sees_interrupt"] is True
        assert results["b_sees_interrupt"] is False

    def test_concurrent_threadpool_workers_inherit_correct_event(self):
        """Worker threads spawned via copy_context().run() see the bound event."""
        agent_event = threading.Event()
        token = bind_event(agent_event)
        observations = []

        def _worker():
            observations.append(is_interrupted())

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                # Submit a worker that should NOT see an interrupt.
                ctx = contextvars.copy_context()
                ex.submit(ctx.run, _worker).result(timeout=2)

                # Now signal the agent's event and submit another worker.
                agent_event.set()
                ctx2 = contextvars.copy_context()
                ex.submit(ctx2.run, _worker).result(timeout=2)
        finally:
            unbind_event(token)
            agent_event.clear()

        assert observations == [False, True]


class TestAIAgentInstanceIsolation:
    """End-to-end test using real AIAgent instances.

    This is the regression test for the API server multi-client scenario:
    interrupting agent A must not cause is_interrupted() to return True
    inside agent B's tool execution context.
    """

    def _make_minimal_agent(self):
        """Build the smallest viable AIAgent without touching the network."""
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent._interrupt_requested = False
        agent._interrupt_message = None
        agent._active_children = []
        agent._active_children_lock = threading.Lock()
        agent.quiet_mode = True
        # The fix gives every agent its own event.
        agent._interrupt_event = threading.Event()
        return agent

    def test_interrupting_agent_a_does_not_affect_agent_b(self):
        from run_agent import AIAgent

        agent_a = self._make_minimal_agent()
        agent_b = self._make_minimal_agent()

        b_observations = {}

        def _run_agent_b():
            # Simulate run_conversation()'s contextvar binding.
            token = bind_event(agent_b._interrupt_event)
            try:
                b_observations["before"] = is_interrupted()
                # While B is "running", another thread interrupts A.
                interrupter = threading.Thread(
                    target=AIAgent.interrupt, args=(agent_a, "client disconnect")
                )
                interrupter.start()
                interrupter.join(timeout=2)
                b_observations["after"] = is_interrupted()
            finally:
                unbind_event(token)

        # Run agent B in its own copied context so the binding is local.
        ctx = contextvars.copy_context()
        tb = threading.Thread(target=ctx.run, args=(_run_agent_b,))
        tb.start()
        tb.join(timeout=3)

        assert b_observations["before"] is False
        assert b_observations["after"] is False, (
            "Agent B's tools observed agent A's interrupt — cross-contamination "
            "regression. The per-agent threading.Event must isolate signals."
        )
        # Agent A's own state should reflect the interrupt.
        assert agent_a._interrupt_requested is True
        assert agent_a._interrupt_event.is_set() is True

    def test_clear_interrupt_on_one_agent_does_not_clear_other(self):
        from run_agent import AIAgent

        agent_a = self._make_minimal_agent()
        agent_b = self._make_minimal_agent()

        # Both agents are interrupted.
        AIAgent.interrupt(agent_a, "stop A")
        AIAgent.interrupt(agent_b, "stop B")
        assert agent_a._interrupt_event.is_set()
        assert agent_b._interrupt_event.is_set()

        # Clearing A must not clear B — historically ``clear_interrupt``
        # reset the process-wide event, so a fresh agent starting up
        # would silently un-interrupt every other concurrent agent.
        AIAgent.clear_interrupt(agent_a)
        assert not agent_a._interrupt_event.is_set()
        assert agent_b._interrupt_event.is_set(), (
            "clear_interrupt() leaked across agents — the per-agent event "
            "is not isolated from the global fallback."
        )
