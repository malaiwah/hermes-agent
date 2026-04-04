"""Tool schemas for persistent ACP background subagents."""

from __future__ import annotations

import json

from agent.background_subagents import get_background_subagent_manager
from tools.registry import registry


def check_background_subagent_requirements() -> bool:
    return get_background_subagent_manager().check_requirements()


SPAWN_BACKGROUND_SUBAGENT_SCHEMA = {
    "name": "spawn_background_subagent",
    "description": (
        "Start a persistent background ACP subagent inside a sandbox container. "
        "Use this when work should continue across turns and you want to poll or "
        "message the subagent later instead of waiting for a delegate_task result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "purpose": {
                "type": "string",
                "description": "Why this background subagent exists and what role it should keep over time.",
            },
            "initial_task": {
                "type": "string",
                "description": "The first task or instruction to send to the subagent after it starts.",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory inside the sandbox container for this subagent.",
            },
            "agent_kind": {
                "type": "string",
                "description": "ACP-capable subagent kind to launch. Default comes from config.",
            },
        },
        "required": ["purpose", "initial_task", "cwd"],
    },
}

LIST_BACKGROUND_SUBAGENTS_SCHEMA = {
    "name": "list_background_subagents",
    "description": "List the current session's active background ACP subagents.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

SEND_BACKGROUND_SUBAGENT_SCHEMA = {
    "name": "send_background_subagent",
    "description": "Send a follow-up instruction to an existing background ACP subagent.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Background subagent identifier.",
            },
            "message": {
                "type": "string",
                "description": "Instruction or follow-up task to send through ACP.",
            },
        },
        "required": ["id", "message"],
    },
}

POLL_BACKGROUND_SUBAGENT_SCHEMA = {
    "name": "poll_background_subagent",
    "description": "Fetch unread buffered updates from a background ACP subagent.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Background subagent identifier.",
            },
            "since_seq": {
                "type": "integer",
                "description": "Optional event cursor; when omitted, returns unread events and marks them read.",
            },
        },
        "required": ["id"],
    },
}

GET_BACKGROUND_SUBAGENT_STATUS_SCHEMA = {
    "name": "get_background_subagent_status",
    "description": "Check liveness, deadlines, and queue state for a background ACP subagent.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Background subagent identifier.",
            },
        },
        "required": ["id"],
    },
}

STOP_BACKGROUND_SUBAGENT_SCHEMA = {
    "name": "stop_background_subagent",
    "description": "Stop a background ACP subagent and tear down its sandbox container.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Background subagent identifier.",
            },
            "reason": {
                "type": "string",
                "description": "Optional short reason for stopping it.",
            },
        },
        "required": ["id"],
    },
}


def _must_be_agent_loop(name: str) -> str:
    return json.dumps({"error": f"{name} must be handled by the agent loop"})


registry.register(
    name="spawn_background_subagent",
    toolset="delegation",
    schema=SPAWN_BACKGROUND_SUBAGENT_SCHEMA,
    handler=lambda args, **kw: _must_be_agent_loop("spawn_background_subagent"),
    check_fn=check_background_subagent_requirements,
    emoji="🧭",
)

registry.register(
    name="list_background_subagents",
    toolset="delegation",
    schema=LIST_BACKGROUND_SUBAGENTS_SCHEMA,
    handler=lambda args, **kw: _must_be_agent_loop("list_background_subagents"),
    check_fn=check_background_subagent_requirements,
    emoji="🧾",
)

registry.register(
    name="send_background_subagent",
    toolset="delegation",
    schema=SEND_BACKGROUND_SUBAGENT_SCHEMA,
    handler=lambda args, **kw: _must_be_agent_loop("send_background_subagent"),
    check_fn=check_background_subagent_requirements,
    emoji="📨",
)

registry.register(
    name="poll_background_subagent",
    toolset="delegation",
    schema=POLL_BACKGROUND_SUBAGENT_SCHEMA,
    handler=lambda args, **kw: _must_be_agent_loop("poll_background_subagent"),
    check_fn=check_background_subagent_requirements,
    emoji="📡",
)

registry.register(
    name="get_background_subagent_status",
    toolset="delegation",
    schema=GET_BACKGROUND_SUBAGENT_STATUS_SCHEMA,
    handler=lambda args, **kw: _must_be_agent_loop("get_background_subagent_status"),
    check_fn=check_background_subagent_requirements,
    emoji="🔎",
)

registry.register(
    name="stop_background_subagent",
    toolset="delegation",
    schema=STOP_BACKGROUND_SUBAGENT_SCHEMA,
    handler=lambda args, **kw: _must_be_agent_loop("stop_background_subagent"),
    check_fn=check_background_subagent_requirements,
    emoji="🛑",
)
