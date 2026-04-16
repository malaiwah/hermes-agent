#!/usr/bin/env python3
"""In-session sideband TTS messaging tool.

Lets the agent send spoken audio to the current user without ending its turn.
Unlike ``text_to_speech``, this tool is side-effecting: the agent loop handles
delivery immediately through the live gateway callbacks for the current session.
"""

import json

from tools.registry import registry


SEND_TTS_MESSAGE_SCHEMA = {
    "name": "send_tts_message",
    "description": (
        "Send a spoken TTS sideband message to the current user in the "
        "current session without ending your turn. Audio delivery happens "
        "immediately, before your turn is finished, so prefer this for "
        "low-latency voice replies and spoken sideband updates when user "
        "experience matters. This is side-effecting. The audio may reach the "
        "user while you are still wrapping up the turn. Optionally include "
        "short text for dual-modality delivery. Do not wait for your final "
        "assistant reply to repeat the same spoken content. If the sideband "
        "audio already fully handled the user-visible reply, finish the turn "
        "with the exact token [SILENT]."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to synthesize and send as audio.",
            },
            "voice": {
                "type": "string",
                "description": (
                    "Optional TTS voice override. For qwen3 this can be a preset "
                    "voice, OpenAI alias, or vc_<clone_id>."
                ),
            },
            "instruct": {
                "type": "string",
                "description": (
                    "Optional speaking style or emotion instruction for the "
                    "TTS provider."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "Optional short text to send alongside the audio in the "
                    "current session for dual-modality delivery."
                ),
            },
        },
        "required": ["text"],
    },
}


def check_send_tts_message_requirements() -> bool:
    """Expose the tool only when TTS generation is actually available."""
    try:
        from tools.tts_tool import check_tts_requirements

        return bool(check_tts_requirements())
    except Exception:
        return False


def send_tts_message_tool(args, **kwargs):
    """Stub dispatcher for the registry.

    Real handling happens in run_agent.py because it needs the live platform
    callbacks for the current session.
    """
    return json.dumps(
        {"error": "send_tts_message must be handled by the agent loop"},
        ensure_ascii=False,
    )


registry.register(
    name="send_tts_message",
    toolset="user_updates",
    schema=SEND_TTS_MESSAGE_SCHEMA,
    handler=send_tts_message_tool,
    check_fn=check_send_tts_message_requirements,
    description="Send a spoken in-session TTS message to the current user immediately.",
    emoji="🔊",
)
