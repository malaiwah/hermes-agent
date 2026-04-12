"""Create Thread Tool -- create a Discord thread from the current channel.

Allows the agent to decide when a conversation warrants its own thread,
rather than forcing auto-threading on every message. Only available when
the current session is on Discord and in a server text channel (not DMs).

Uses the Discord REST API directly (like send_message_tool), so it works
without needing a reference to the live DiscordAdapter instance.
"""

import json
import logging
import os

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# Discord REST API v10 — valid auto_archive_duration values (minutes)
_VALID_ARCHIVE_DURATIONS = {60, 1440, 4320, 10080}

CREATE_THREAD_SCHEMA = {
    "name": "create_thread",
    "description": (
        "Create a new Discord thread in the current channel. Use this when a "
        "conversation becomes complex enough to warrant its own thread — e.g. "
        "multi-step tasks, long debugging sessions, or topic changes. "
        "Returns the thread ID and name. After creating, your subsequent "
        "responses will be sent to the new thread.\n\n"
        "Only works on Discord server channels (not DMs). The thread is "
        "created from the user's original message."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Thread name (1-100 characters). Should be descriptive of the topic.",
            },
            "message": {
                "type": "string",
                "description": "Optional initial message to post in the thread.",
            },
            "auto_archive_duration": {
                "type": "integer",
                "enum": [60, 1440, 4320, 10080],
                "description": (
                    "Minutes of inactivity before auto-archiving. "
                    "60 (1h), 1440 (1d, default), 4320 (3d), 10080 (1w)."
                ),
            },
        },
        "required": ["name"],
    },
}


def _check_create_thread():
    """Only available on Discord sessions in server channels."""
    return os.getenv("HERMES_SESSION_PLATFORM", "") == "discord"


async def create_thread_handler(args, **kw):
    """Create a Discord thread via the REST API."""
    name = (args.get("name") or "").strip()
    if not name:
        return tool_error("Thread name is required.")
    if len(name) > 100:
        return tool_error("Thread name must be 100 characters or fewer.")

    message = (args.get("message") or "").strip()
    auto_archive = args.get("auto_archive_duration", 1440)
    if auto_archive not in _VALID_ARCHIVE_DURATIONS:
        return tool_error(
            f"auto_archive_duration must be one of: {sorted(_VALID_ARCHIVE_DURATIONS)}"
        )

    # Get current session context from env vars set by gateway
    channel_id = os.getenv("HERMES_SESSION_CHAT_ID", "")
    if not channel_id:
        return tool_error("No channel ID available — cannot create thread.")

    # If we're already in a thread, don't nest
    thread_id = os.getenv("HERMES_SESSION_THREAD_ID", "")
    if thread_id:
        return tool_error(
            "Already inside a thread. Cannot create a nested thread."
        )

    token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not token:
        return tool_error("DISCORD_BOT_TOKEN not set.")

    try:
        import aiohttp
    except ImportError:
        return tool_error("aiohttp not installed.")

    try:
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
    except Exception:
        _sess_kw, _req_kw = {}, {}

    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }

    # Discord REST: POST /channels/{channel_id}/threads
    # Type 11 = PUBLIC_THREAD (requires a message to start from in non-forum channels)
    # We'll first send a seed message, then create a thread from it.
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30), **_sess_kw,
        ) as session:
            # Step 1: Send a seed message to anchor the thread
            seed_content = message or f"\U0001f9f5 **{name}**"
            msg_url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
            async with session.post(
                msg_url,
                headers=headers,
                json={"content": seed_content},
                **_req_kw,
            ) as resp:
                if resp.status not in (200, 201):
                    body = await resp.text()
                    return tool_error(
                        f"Failed to send seed message ({resp.status}): {body}"
                    )
                msg_data = await resp.json()
                message_id = msg_data["id"]

            # Step 2: Create thread from that message
            thread_url = (
                f"https://discord.com/api/v10/channels/{channel_id}"
                f"/messages/{message_id}/threads"
            )
            async with session.post(
                thread_url,
                headers=headers,
                json={
                    "name": name,
                    "auto_archive_duration": auto_archive,
                },
                **_req_kw,
            ) as resp:
                if resp.status not in (200, 201):
                    body = await resp.text()
                    return tool_error(
                        f"Failed to create thread ({resp.status}): {body}"
                    )
                thread_data = await resp.json()

        result = {
            "success": True,
            "thread_id": thread_data["id"],
            "thread_name": thread_data.get("name", name),
            "parent_channel_id": channel_id,
        }
        logger.info(
            "create_thread: created '%s' (id=%s) in channel %s",
            name, thread_data["id"], channel_id,
        )
        return tool_result(result)

    except Exception as e:
        return tool_error(f"Discord thread creation failed: {e}")


registry.register(
    name="create_thread",
    toolset="messaging",
    schema=CREATE_THREAD_SCHEMA,
    handler=create_thread_handler,
    check_fn=_check_create_thread,
    is_async=True,
    emoji="\U0001f9f5",
)
