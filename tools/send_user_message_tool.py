#!/usr/bin/env python3
"""In-session user messaging tool.

Lets the agent send a natural-language status update to the current user
without ending its turn. Unlike ``send_message``, this stays inside the
current session/thread and is handled by the agent loop via a platform
callback.
"""

import json

from tools.registry import registry


SEND_USER_MESSAGE_SCHEMA = {
    "name": "send_user_message",
    "description": (
        "Send a natural-language sideband message to the current user in the "
        "current session without ending your turn. The user receives it "
        "immediately, before your turn is finished, so use it to improve "
        "responsiveness when you still need time to keep working. Prefer it "
        "for concise acknowledgements, progress updates, or short status "
        "changes before or between tool calls. Keep messages short and useful. "
        "Do not wait for your final assistant reply to say the same thing "
        "again. If this sideband message fully covers what the user should "
        "see for now, finish the turn with the exact token [SILENT]."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": (
                    "The short message to send to the current user in natural "
                    "language. Example: 'I found the relevant files and I am "
                    "patching the config path next.'"
                ),
            }
        },
        "required": ["message"],
    },
}


def check_send_user_message_requirements() -> bool:
    """Tool is schema-available on interactive platforms."""
    return True


def send_user_message_tool(args, **kwargs):
    """Stub dispatcher for the registry.

    Real handling happens in run_agent.py because it needs the live platform
    callback for the current session.
    """
    return json.dumps(
        {"error": "send_user_message must be handled by the agent loop"},
        ensure_ascii=False,
    )


registry.register(
    name="send_user_message",
    toolset="user_updates",
    schema=SEND_USER_MESSAGE_SCHEMA,
    handler=send_user_message_tool,
    check_fn=check_send_user_message_requirements,
    description="Send a short in-session progress update to the current user.",
    emoji="💬",
)
