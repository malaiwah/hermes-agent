"""Shared interrupt signaling for tools.

Each ``AIAgent`` owns its own ``threading.Event`` and binds it to a
context variable for the duration of ``run_conversation()``.  Tools
that poll :func:`is_interrupted` observe only the currently-bound
agent's signal, so two agents running concurrently in the same process
— for example two ``/v1/chat/completions`` requests on the API server,
or a gateway runner with multiple active sessions — cannot interrupt
each other's in-flight tool calls.

A module-level fallback ``threading.Event`` is retained for backwards
compatibility with code paths that have not bound a per-agent event:
single-agent CLI usage, the existing pre-tool interrupt tests, and any
caller that imports ``_interrupt_event`` directly.

Tools should poll the interrupt like this::

    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"output": "[interrupted]", "returncode": 130}

Threads spawned by an agent (e.g. ``ThreadPoolExecutor`` workers used by
``_execute_tool_calls_concurrent``) must inherit the calling thread's
context via ``contextvars.copy_context().run(...)`` — neither
``asyncio.loop.run_in_executor`` nor bare ``threading.Thread`` propagates
context automatically.
"""

import contextvars
import threading
from typing import Optional


# Module-level fallback event.  Used when no per-agent event is bound to
# ``_current_event`` below — preserves the original single-agent
# semantics for CLI usage, tests that import ``_interrupt_event``
# directly, and any code path running outside of
# ``AIAgent.run_conversation()``.
_interrupt_event = threading.Event()


# ContextVar holding the interrupt event for the currently active agent.
# ``AIAgent.run_conversation()`` binds the agent's own
# ``threading.Event`` here at the start of each turn so tools observe its
# per-instance signal rather than the process-wide fallback.
_current_event: "contextvars.ContextVar[Optional[threading.Event]]" = (
    contextvars.ContextVar("hermes_current_interrupt_event", default=None)
)


def _active_event() -> threading.Event:
    """Return the per-agent event if one is bound, else the global fallback."""
    bound = _current_event.get()
    return bound if bound is not None else _interrupt_event


def bind_event(event: threading.Event) -> "contextvars.Token":
    """Bind a per-agent interrupt event to the current context.

    Returns a token that must be passed to :func:`unbind_event` (or to
    ``_current_event.reset()``) to restore the previous binding when the
    agent's turn ends.
    """
    return _current_event.set(event)


def unbind_event(token: "contextvars.Token") -> None:
    """Restore the previous binding established before :func:`bind_event`.

    Safe to call on a token from a different context — the failure modes
    of ``ContextVar.reset`` are swallowed so cleanup paths can run
    unconditionally.
    """
    try:
        _current_event.reset(token)
    except (LookupError, ValueError):
        pass


def set_interrupt(active: bool) -> None:
    """Signal or clear the interrupt for the currently active agent.

    Operates on the per-agent event when one is bound, otherwise on the
    module-level fallback so legacy callers and tests continue to work
    without modification.
    """
    event = _active_event()
    if active:
        event.set()
    else:
        event.clear()


def is_interrupted() -> bool:
    """Check if an interrupt has been requested for the current agent.

    Safe to call from any thread that has inherited the agent's
    contextvars (the agent's own thread, or ``ThreadPoolExecutor``
    workers wrapped in ``contextvars.copy_context().run(...)``).
    """
    return _active_event().is_set()
