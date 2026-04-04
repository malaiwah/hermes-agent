"""Persistent ACP background subagents for long-lived sandboxed work."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Optional

logger = logging.getLogger(__name__)

try:
    import acp

    _ACP_PROTOCOL_VERSION = acp.PROTOCOL_VERSION
except Exception:
    _ACP_PROTOCOL_VERSION = 1


_DEFAULT_BACKGROUND_CONFIG: dict[str, Any] = {
    "enabled": True,
    "max_per_session": 3,
    "max_global": 8,
    "idle_timeout_seconds": 900,
    "max_lifetime_seconds": 7200,
    "default_agent_kind": "opencode",
    "agents": {
        "opencode": {
            "command": "opencode",
            "args": ["acp"],
            "cwd_mode": "session",
        },
    },
}

_ACTIVE_STATUSES = {"starting", "running", "idle"}
_TERMINAL_STATUSES = {"completed", "stopped", "timed_out", "error"}

_NAME_LEFT = [
    "amber",
    "aster",
    "cedar",
    "cinder",
    "ember",
    "harbor",
    "hollow",
    "lumen",
    "marble",
    "meadow",
    "morrow",
    "river",
    "sable",
    "sierra",
    "silver",
    "solar",
]
_NAME_RIGHT = [
    "badger",
    "falcon",
    "finch",
    "fox",
    "heron",
    "ibis",
    "lynx",
    "otter",
    "owl",
    "panda",
    "raven",
    "robin",
    "starling",
    "stoat",
    "swift",
    "wolf",
]


def _now() -> float:
    return time.time()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_background_subagent_config() -> dict[str, Any]:
    runtime_cfg: dict[str, Any] = {}
    try:
        from cli import CLI_CONFIG

        runtime_cfg = dict(((CLI_CONFIG or {}).get("delegation", {}) or {}).get("background_subagents", {}) or {})
    except Exception:
        runtime_cfg = {}

    persistent_cfg: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        persistent_cfg = dict(((load_config() or {}).get("delegation", {}) or {}).get("background_subagents", {}) or {})
    except Exception:
        persistent_cfg = {}

    return _deep_merge(_deep_merge(_DEFAULT_BACKGROUND_CONFIG, persistent_cfg), runtime_cfg)


def _generate_display_name(subagent_id: str) -> str:
    sanitized = "".join(ch for ch in subagent_id if ch in "0123456789abcdefABCDEF")
    seed = int((sanitized or "0")[:8], 16)
    left = _NAME_LEFT[seed % len(_NAME_LEFT)]
    right = _NAME_RIGHT[(seed // len(_NAME_LEFT)) % len(_NAME_RIGHT)]
    return f"{left}-{right}"


def _truncate(text: str, limit: int = 120) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


@dataclass
class BufferedEvent:
    seq: int
    kind: str
    text: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "seq": self.seq,
            "kind": self.kind,
            "text": self.text,
            "timestamp": self.timestamp,
        }
        if self.data:
            payload["data"] = self.data
        return payload


@dataclass
class PendingPrompt:
    prompt_text: str
    queued_at: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingRequest:
    request_id: int
    method: str
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: dict[str, Any] | None = None
    callback: Optional[Callable[[Any, dict[str, Any] | None], None]] = None


@dataclass
class BackgroundSubagentRecord:
    id: str
    name: str
    owner_session_id: str
    purpose: str
    cwd: str
    agent_kind: str
    status: str
    started_at: float
    last_activity_at: float
    idle_timeout_at: float
    max_lifetime_at: float
    acp_session_id: str = ""
    container_id: str = ""
    acp_process_pid: int | None = None
    current_task: str = ""
    active_prompt_request_id: int | None = None
    pending_prompts: Deque[PendingPrompt] = field(default_factory=deque)
    events: list[BufferedEvent] = field(default_factory=list)
    last_polled_seq: int = 0
    last_seq: int = 0
    last_error: str = ""
    terminal_reason: str = ""
    environment: Any = None
    connection: Any = None

    @property
    def unread_count(self) -> int:
        return max(self.last_seq - self.last_polled_seq, 0)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "purpose": self.purpose,
            "cwd": self.cwd,
            "agent_kind": self.agent_kind,
            "unread_count": self.unread_count,
            "current_task": self.current_task,
            "started_at": self.started_at,
            "last_activity_at": self.last_activity_at,
            "idle_timeout_at": self.idle_timeout_at,
            "max_lifetime_at": self.max_lifetime_at,
            "container_id": self.container_id,
            "acp_process_pid": self.acp_process_pid,
            "acp_session_id": self.acp_session_id,
            "terminal_reason": self.terminal_reason,
        }


class ACPPeerConnection:
    """Persistent ACP stdio client over a long-lived exec session."""

    def __init__(
        self,
        exec_session: Any,
        *,
        cwd: str,
        on_update: Callable[[dict[str, Any]], None],
        on_stderr: Callable[[str], None],
        on_exit: Callable[[int | None], None],
        on_protocol_error: Callable[[str], None],
        on_activity: Callable[[], None],
        inbound_handler: Optional[Callable[[str, dict[str, Any]], tuple[Any, dict[str, Any] | None]]] = None,
    ):
        self._exec_session = exec_session
        self._cwd = cwd
        self._on_update = on_update
        self._on_stderr = on_stderr
        self._on_exit = on_exit
        self._on_protocol_error = on_protocol_error
        self._on_activity = on_activity
        self._inbound_handler = inbound_handler or self._default_inbound_handler
        self._lock = threading.Lock()
        self._next_id = 0
        self._pending: dict[int, PendingRequest] = {}
        self._closed = False
        self._exec_session.read_loop(
            stdout_handler=self._handle_stdout_line,
            stderr_handler=self._handle_stderr_line,
            exit_handler=self._handle_exit,
        )

    def initialize(self) -> None:
        self.request(
            "initialize",
            {
                "protocolVersion": _ACP_PROTOCOL_VERSION,
                "clientCapabilities": {
                    "fs": {
                        "readTextFile": False,
                        "writeTextFile": False,
                    }
                },
                "clientInfo": {
                    "name": "hermes-agent",
                    "title": "Hermes Agent",
                    "version": "0.0.0",
                },
            },
            timeout=20.0,
        )

    def open_session(self) -> str:
        result = self.request(
            "session/new",
            {
                "cwd": self._cwd,
                "mcpServers": [],
            },
            timeout=20.0,
        ) or {}
        session_id = str(result.get("sessionId") or "").strip()
        if not session_id:
            raise RuntimeError("ACP peer did not return a sessionId.")
        return session_id

    def send_prompt_async(
        self,
        *,
        session_id: str,
        prompt_text: str,
        meta: dict[str, Any] | None = None,
        on_response: Optional[Callable[[Any, dict[str, Any] | None], None]] = None,
    ) -> int:
        return self.request(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": prompt_text}],
                "_meta": dict(meta or {}),
            },
            wait=False,
            callback=on_response,
        )

    def cancel(self, session_id: str) -> None:
        try:
            self.request(
                "session/cancel",
                {"sessionId": session_id},
                wait=False,
            )
        except Exception:
            logger.debug("ACP cancel failed", exc_info=True)

    def close(self) -> None:
        self._closed = True
        try:
            self._exec_session.terminate_session()
        except Exception:
            pass

    def is_alive(self) -> bool:
        return not self._closed and bool(self._exec_session.is_session_alive())

    @property
    def pid(self) -> int | None:
        return getattr(self._exec_session, "pid", None)

    def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = 60.0,
        wait: bool = True,
        callback: Optional[Callable[[Any, dict[str, Any] | None], None]] = None,
    ) -> Any:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            pending = PendingRequest(
                request_id=request_id,
                method=method,
                callback=callback,
            )
            self._pending[request_id] = pending
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        try:
            self._exec_session.write_line(json.dumps(payload, ensure_ascii=True))
        except Exception as exc:
            with self._lock:
                self._pending.pop(request_id, None)
            raise RuntimeError(f"Failed to write ACP request {method}: {exc}") from exc

        self._on_activity()
        if not wait:
            return request_id

        if not pending.event.wait(timeout):
            with self._lock:
                self._pending.pop(request_id, None)
            raise TimeoutError(f"Timed out waiting for ACP response to {method}.")

        if pending.error:
            raise RuntimeError(
                f"ACP {method} failed: {pending.error.get('message') or pending.error}"
            )
        return pending.result

    def _handle_stdout_line(self, line: str) -> None:
        if self._closed:
            return
        try:
            msg = json.loads(line)
        except Exception:
            self._on_protocol_error(f"Malformed ACP stdout line: {line!r}")
            return

        self._on_activity()
        method = msg.get("method")
        if isinstance(method, str):
            self._handle_inbound_message(msg)
            return

        message_id = msg.get("id")
        if not isinstance(message_id, int):
            self._on_protocol_error(f"ACP message missing integer id: {msg!r}")
            return

        with self._lock:
            pending = self._pending.pop(message_id, None)
        if pending is None:
            logger.debug("Ignoring ACP response for unknown request id %s", message_id)
            return

        pending.result = msg.get("result")
        pending.error = msg.get("error")
        pending.event.set()
        if pending.callback:
            try:
                pending.callback(pending.result, pending.error)
            except Exception:
                logger.warning("ACP pending callback failed", exc_info=True)

    def _handle_stderr_line(self, line: str) -> None:
        if line.strip():
            self._on_stderr(line.rstrip("\n"))

    def _handle_exit(self, returncode: int | None) -> None:
        self._closed = True
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for item in pending:
            item.error = {"message": f"ACP process exited before {item.method} completed."}
            item.event.set()
            if item.callback:
                try:
                    item.callback(None, item.error)
                except Exception:
                    logger.warning("ACP exit callback failed", exc_info=True)
        self._on_exit(returncode)

    def _handle_inbound_message(self, msg: dict[str, Any]) -> None:
        method = str(msg.get("method") or "")
        if method == "session/update":
            self._on_update(msg)
            return

        message_id = msg.get("id")
        params = msg.get("params") or {}
        result, error = self._inbound_handler(method, params)
        if message_id is None:
            return
        if error:
            response = _jsonrpc_error(message_id, error.get("code", -32601), error.get("message", "Unsupported ACP method"))
        else:
            response = {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": result,
            }
        try:
            self._exec_session.write_line(json.dumps(response, ensure_ascii=True))
        except Exception as exc:
            self._on_protocol_error(f"Failed to write ACP inbound response: {exc}")

    @staticmethod
    def _default_inbound_handler(method: str, params: dict[str, Any]) -> tuple[Any, dict[str, Any] | None]:
        if method == "session/request_permission":
            return {
                "outcome": {
                    "outcome": "allow_once",
                }
            }, None
        return None, {"code": -32601, "message": f"ACP client method '{method}' is not supported by Hermes yet."}


class BackgroundSubagentManager:
    """Tracks persistent ACP background subagents across Hermes turns."""

    def __init__(self):
        self._lock = threading.RLock()
        self._records: dict[str, BackgroundSubagentRecord] = {}
        self._pending_nudges: dict[str, list[str]] = defaultdict(list)
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop = threading.Event()

    def check_requirements(self) -> bool:
        cfg = _load_background_subagent_config()
        return bool(cfg.get("enabled", True))

    def reset_for_tests(self) -> None:
        with self._lock:
            records = list(self._records.values())
            self._records.clear()
            self._pending_nudges.clear()
        for record in records:
            self._shutdown_record(record, "stopped", "reset")

    def spawn_subagent(
        self,
        *,
        owner_session_id: str,
        purpose: str,
        initial_task: str,
        cwd: str,
        agent_kind: str | None = None,
    ) -> dict[str, Any]:
        cfg = _load_background_subagent_config()
        if not cfg.get("enabled", True):
            return {"success": False, "error": "Background ACP subagents are disabled in config."}

        purpose = str(purpose or "").strip()
        initial_task = str(initial_task or "").strip()
        cwd = str(cwd or "").strip()
        if not purpose:
            return {"success": False, "error": "purpose is required."}
        if not initial_task:
            return {"success": False, "error": "initial_task is required."}
        if not cwd:
            return {"success": False, "error": "cwd is required."}

        resolved_agent_kind = str(agent_kind or cfg.get("default_agent_kind") or "opencode").strip() or "opencode"
        agent_cfg = self._resolve_agent_config(cfg, resolved_agent_kind)
        if "error" in agent_cfg:
            return {"success": False, "error": agent_cfg["error"]}

        with self._lock:
            session_active = [
                record for record in self._records.values()
                if record.owner_session_id == owner_session_id and record.status in _ACTIVE_STATUSES
            ]
            global_active = [record for record in self._records.values() if record.status in _ACTIVE_STATUSES]
            if len(session_active) >= int(cfg.get("max_per_session", 3)):
                return {
                    "success": False,
                    "error": "Background subagent limit reached for this session.",
                    "stoppable_subagents": [record.summary() for record in session_active],
                }
            if len(global_active) >= int(cfg.get("max_global", 8)):
                return {
                    "success": False,
                    "error": "Global background subagent limit reached.",
                    "stoppable_subagents": [record.summary() for record in session_active],
                }

        subagent_id = f"bg-{uuid.uuid4().hex[:10]}"
        record = BackgroundSubagentRecord(
            id=subagent_id,
            name=_generate_display_name(subagent_id),
            owner_session_id=owner_session_id,
            purpose=purpose,
            cwd=cwd,
            agent_kind=resolved_agent_kind,
            status="starting",
            started_at=_now(),
            last_activity_at=_now(),
            idle_timeout_at=_now() + int(cfg.get("idle_timeout_seconds", 900)),
            max_lifetime_at=_now() + int(cfg.get("max_lifetime_seconds", 7200)),
        )

        try:
            environment = self._create_environment(task_id=subagent_id, cwd=cwd)
            command = [agent_cfg["command"], *list(agent_cfg.get("args") or [])]
            exec_session = environment.start_persistent_exec(
                cwd=cwd,
                command=command,
                env=dict(agent_cfg.get("env") or {}),
            )
            connection = ACPPeerConnection(
                exec_session,
                cwd=cwd,
                on_update=lambda msg, record_id=subagent_id: self._handle_update(record_id, msg),
                on_stderr=lambda line, record_id=subagent_id: self._handle_stderr(record_id, line),
                on_exit=lambda returncode, record_id=subagent_id: self._handle_exit(record_id, returncode),
                on_protocol_error=lambda message, record_id=subagent_id: self._handle_protocol_error(record_id, message),
                on_activity=lambda record_id=subagent_id: self._touch(record_id),
            )
            connection.initialize()
            session_id = connection.open_session()
        except Exception as exc:
            try:
                environment.cleanup()  # type: ignore[name-defined]
            except Exception:
                pass
            return {
                "success": False,
                "error": f"Failed to start background ACP subagent: {exc}",
            }

        record.environment = environment
        record.connection = connection
        record.acp_session_id = session_id
        record.container_id = str(getattr(environment, "_container_id", "") or "")
        record.acp_process_pid = connection.pid
        record.status = "idle"

        with self._lock:
            self._records[record.id] = record
            self._ensure_monitor_thread()

        send_result = self.send_message(
            owner_session_id=owner_session_id,
            subagent_id=record.id,
            message=initial_task,
            source="spawn",
        )
        if not send_result.get("success"):
            return send_result

        return {
            "success": True,
            "id": record.id,
            "name": record.name,
            "status": record.status,
            "purpose": record.purpose,
            "cwd": record.cwd,
            "agent_kind": record.agent_kind,
            "container_id": record.container_id,
            "acp_session_id": record.acp_session_id,
            "initial_dispatch": send_result,
        }

    def list_subagents(self, *, owner_session_id: str) -> dict[str, Any]:
        with self._lock:
            subagents = [
                record.summary()
                for record in self._records.values()
                if record.owner_session_id == owner_session_id and record.status in _ACTIVE_STATUSES
            ]
        subagents.sort(key=lambda item: item["started_at"])
        return {"success": True, "subagents": subagents}

    def get_status(self, *, owner_session_id: str, subagent_id: str) -> dict[str, Any]:
        record = self._get_owned_record(owner_session_id, subagent_id)
        if record is None:
            return {"success": False, "error": f"Unknown background subagent: {subagent_id}"}
        alive = bool(record.connection and record.connection.is_alive())
        payload = record.summary()
        payload.update(
            {
                "success": True,
                "transport_alive": alive,
                "queued_messages": len(record.pending_prompts),
            }
        )
        if record.last_error:
            payload["last_error"] = record.last_error
        return payload

    def send_message(
        self,
        *,
        owner_session_id: str,
        subagent_id: str,
        message: str,
        source: str = "tool",
    ) -> dict[str, Any]:
        record = self._get_owned_record(owner_session_id, subagent_id)
        if record is None:
            return {"success": False, "error": f"Unknown background subagent: {subagent_id}"}
        text = str(message or "").strip()
        if not text:
            return {"success": False, "error": "message is required."}
        if record.status in _TERMINAL_STATUSES:
            return {"success": False, "error": f"Background subagent {record.name} is no longer running."}

        prompt = PendingPrompt(
            prompt_text=text,
            queued_at=_now(),
            meta={"source": source, "subagentId": record.id},
        )
        shutdown_record: BackgroundSubagentRecord | None = None
        with self._lock:
            if record.active_prompt_request_id is not None:
                record.pending_prompts.append(prompt)
                record.status = "running"
                queued = True
            else:
                queued = False
                try:
                    self._dispatch_prompt_locked(record, prompt)
                except Exception as exc:
                    record.last_error = str(exc)
                    self._append_event_locked(record, "error", record.last_error, data={"error": str(exc)})
                    shutdown_record = record
                else:
                    shutdown_record = None
        if shutdown_record is not None:
            self._shutdown_record(shutdown_record, "error", "dispatch_failed")
            return {
                "success": False,
                "error": f"Failed to dispatch prompt to {record.name}: {record.last_error}",
                "id": record.id,
                "name": record.name,
            }
        return {
            "success": True,
            "id": record.id,
            "name": record.name,
            "status": record.status,
            "queued": queued,
            "queued_messages": len(record.pending_prompts),
        }

    def poll_subagent(
        self,
        *,
        owner_session_id: str,
        subagent_id: str,
        since_seq: int | None = None,
    ) -> dict[str, Any]:
        record = self._get_owned_record(owner_session_id, subagent_id)
        if record is None:
            return {"success": False, "error": f"Unknown background subagent: {subagent_id}"}

        with self._lock:
            start_seq = int(since_seq or record.last_polled_seq)
            events = [event.to_dict() for event in record.events if event.seq > start_seq]
            if since_seq is None:
                record.last_polled_seq = record.last_seq

        return {
            "success": True,
            "id": record.id,
            "name": record.name,
            "status": record.status,
            "events": events,
            "unread_count": record.unread_count,
            "transport_alive": bool(record.connection and record.connection.is_alive()),
        }

    def stop_subagent(
        self,
        *,
        owner_session_id: str,
        subagent_id: str,
        reason: str = "",
    ) -> dict[str, Any]:
        record = self._get_owned_record(owner_session_id, subagent_id)
        if record is None:
            return {"success": False, "error": f"Unknown background subagent: {subagent_id}"}
        self._shutdown_record(record, "stopped", reason or "stopped_by_parent")
        return {
            "success": True,
            "id": record.id,
            "name": record.name,
            "status": record.status,
            "reason": record.terminal_reason,
        }

    def render_turn_context(self, owner_session_id: str) -> str:
        with self._lock:
            nudges = list(self._pending_nudges.get(owner_session_id, []))
            if nudges:
                self._pending_nudges[owner_session_id].clear()
            active = [
                record for record in self._records.values()
                if record.owner_session_id == owner_session_id and record.status in _ACTIVE_STATUSES
            ]

        if not nudges and not active:
            return ""

        parts: list[str] = []
        if nudges:
            parts.append(
                "Background subagent updates since your last turn:\n"
                + "\n".join(f"- {item}" for item in nudges)
            )
        if active:
            roster_lines = []
            for record in sorted(active, key=lambda item: item.started_at):
                detail = (
                    f"- {record.name} ({record.id}) [{record.status}] "
                    f"agent={record.agent_kind} cwd={record.cwd} unread={record.unread_count} "
                    f"purpose={_truncate(record.purpose, 140)}"
                )
                if record.current_task:
                    detail += f" current_task={_truncate(record.current_task, 100)}"
                roster_lines.append(detail)
            parts.append(
                "Open background ACP subagents you can manage with "
                "spawn_background_subagent/list_background_subagents/"
                "send_background_subagent/poll_background_subagent/"
                "get_background_subagent_status/stop_background_subagent:\n"
                + "\n".join(roster_lines)
            )
        return "\n\n".join(parts)

    def _resolve_agent_config(self, cfg: dict[str, Any], agent_kind: str) -> dict[str, Any]:
        agents = dict(cfg.get("agents") or {})
        agent_cfg = dict(agents.get(agent_kind) or {})
        if not agent_cfg:
            return {"error": f"Unknown background_subagents agent kind: {agent_kind}"}
        command = str(agent_cfg.get("command") or "").strip()
        if not command:
            return {"error": f"background_subagents agent '{agent_kind}' is missing a command."}
        args = agent_cfg.get("args") or []
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            return {"error": f"background_subagents agent '{agent_kind}' args must be a list of strings."}
        return {
            "command": command,
            "args": list(args),
            "cwd_mode": str(agent_cfg.get("cwd_mode") or "session"),
            "env": dict(agent_cfg.get("env") or {}),
        }

    def _create_environment(self, *, task_id: str, cwd: str) -> Any:
        from tools.terminal_tool import _create_environment, _get_env_config, _resolve_task_environment_settings

        settings = _resolve_task_environment_settings(task_id, _get_env_config())
        env_type = str(settings.get("env_type") or "")
        if env_type != "docker":
            raise RuntimeError(
                f"Background ACP subagents currently require terminal backend 'docker' (got {env_type!r})."
            )
        return _create_environment(
            env_type=env_type,
            image=settings.get("image", ""),
            cwd=cwd or settings.get("cwd", "/root"),
            timeout=max(int(settings.get("timeout", 180)), 180),
            ssh_config=settings.get("ssh_config"),
            container_config=settings.get("container_config"),
            local_config=settings.get("local_config"),
            task_id=task_id,
            host_cwd=settings.get("host_cwd"),
        )

    def _dispatch_prompt_locked(self, record: BackgroundSubagentRecord, prompt: PendingPrompt) -> None:
        if not record.connection or not record.acp_session_id:
            raise RuntimeError("Background subagent connection is not ready.")
        request_id = record.connection.send_prompt_async(
            session_id=record.acp_session_id,
            prompt_text=prompt.prompt_text,
            meta={
                "purpose": record.purpose,
                "backgroundSubagentId": record.id,
                **dict(prompt.meta or {}),
            },
            on_response=lambda result, error, record_id=record.id: self._handle_prompt_response(record_id, result, error),
        )
        record.active_prompt_request_id = request_id
        record.current_task = prompt.prompt_text
        record.status = "running"
        self._touch_locked(record)
        self._append_event_locked(
            record,
            kind="task_dispatched",
            text=_truncate(prompt.prompt_text, 300),
            data={"source": prompt.meta.get("source", "tool")},
        )

    def _handle_prompt_response(self, record_id: str, result: Any, error: dict[str, Any] | None) -> None:
        shutdown_record: BackgroundSubagentRecord | None = None
        with self._lock:
            record = self._records.get(record_id)
            if not record:
                return
            record.active_prompt_request_id = None
            record.current_task = ""
            if error:
                record.last_error = str(error.get("message") or error)
                self._append_event_locked(record, "error", record.last_error, data={"error": error})
                shutdown_record = record
            else:
                stop_reason = ""
                if isinstance(result, dict):
                    stop_reason = str(result.get("stopReason") or result.get("stop_reason") or "").strip()
                if stop_reason:
                    text = f"Latest task finished ({stop_reason})."
                else:
                    text = "Latest task finished."
                record.status = "idle"
                self._append_event_locked(record, "task_complete", text, data={"result": result or {}})
                self._queue_hidden_nudge_locked(
                    record.owner_session_id,
                    f"{record.name} ({record.id}) finished its latest task.",
                )
                if record.pending_prompts:
                    prompt = record.pending_prompts.popleft()
                    self._dispatch_prompt_locked(record, prompt)
                else:
                    self._touch_locked(record)

        if shutdown_record is not None:
            self._shutdown_record(shutdown_record, "error", "prompt_error")

    def _handle_update(self, record_id: str, msg: dict[str, Any]) -> None:
        with self._lock:
            record = self._records.get(record_id)
            if not record:
                return
            params = msg.get("params") or {}
            update = params.get("update") or {}
            kind = str(update.get("sessionUpdate") or update.get("type") or "update").strip() or "update"
            text = self._extract_update_text(update)
            self._append_event_locked(record, kind, text, data={"raw": update})
            if record.status not in _TERMINAL_STATUSES:
                record.status = "running"
            self._touch_locked(record)

    def _handle_stderr(self, record_id: str, line: str) -> None:
        with self._lock:
            record = self._records.get(record_id)
            if not record:
                return
            self._append_event_locked(record, "stderr", line, data={})
            self._touch_locked(record)

    def _handle_protocol_error(self, record_id: str, message: str) -> None:
        with self._lock:
            record = self._records.get(record_id)
            if not record:
                return
            record.last_error = message
            self._append_event_locked(record, "protocol_error", message, data={})
        self._shutdown_record(record, "error", "protocol_error")

    def _handle_exit(self, record_id: str, returncode: int | None) -> None:
        with self._lock:
            record = self._records.get(record_id)
            if not record:
                return
            if record.status in _TERMINAL_STATUSES:
                return
            status = "completed" if (returncode or 0) == 0 else "error"
            reason = f"channel_closed:{returncode if returncode is not None else 'unknown'}"
        self._shutdown_record(record, status, reason)

    def _append_event_locked(
        self,
        record: BackgroundSubagentRecord,
        kind: str,
        text: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> None:
        record.last_seq += 1
        record.events.append(
            BufferedEvent(
                seq=record.last_seq,
                kind=kind,
                text=str(text or ""),
                timestamp=_now(),
                data=dict(data or {}),
            )
        )

    def _touch(self, record_id: str) -> None:
        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                return
            self._touch_locked(record)

    def _touch_locked(self, record: BackgroundSubagentRecord) -> None:
        cfg = _load_background_subagent_config()
        record.last_activity_at = _now()
        record.idle_timeout_at = record.last_activity_at + int(cfg.get("idle_timeout_seconds", 900))

    def _queue_hidden_nudge_locked(self, owner_session_id: str, text: str) -> None:
        self._pending_nudges[owner_session_id].append(text)

    def _get_owned_record(self, owner_session_id: str, subagent_id: str) -> BackgroundSubagentRecord | None:
        with self._lock:
            record = self._records.get(subagent_id)
            if record is None or record.owner_session_id != owner_session_id:
                return None
            return record

    def _ensure_monitor_thread(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="background-subagent-monitor",
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(1.0):
            self._sweep_once()
            with self._lock:
                if not any(record.status in _ACTIVE_STATUSES for record in self._records.values()):
                    self._monitor_stop.set()

    def _sweep_once(self) -> None:
        now = _now()
        with self._lock:
            records = [record for record in self._records.values() if record.status in _ACTIVE_STATUSES]
        for record in records:
            if now >= record.max_lifetime_at:
                self._shutdown_record(record, "timed_out", "max_lifetime_exceeded")
                continue
            if now >= record.idle_timeout_at:
                self._shutdown_record(record, "timed_out", "idle_timeout")
                continue
            if record.connection and not record.connection.is_alive():
                self._shutdown_record(record, "error", "transport_not_alive")

    def _shutdown_record(self, record: BackgroundSubagentRecord, final_status: str, reason: str) -> None:
        with self._lock:
            if record.status in _TERMINAL_STATUSES:
                return
            record.status = final_status
            record.terminal_reason = reason
            if reason:
                self._append_event_locked(record, final_status, f"Session ended: {reason}", data={"reason": reason})
            self._queue_hidden_nudge_locked(
                record.owner_session_id,
                f"{record.name} ({record.id}) is now {final_status} ({reason}).",
            )
        if record.connection and record.acp_session_id:
            try:
                record.connection.cancel(record.acp_session_id)
            except Exception:
                pass
        if record.connection:
            try:
                record.connection.close()
            except Exception:
                pass
        if record.environment:
            try:
                record.environment.cleanup()
            except Exception:
                logger.debug("Background subagent cleanup failed", exc_info=True)

    @staticmethod
    def _extract_update_text(update: dict[str, Any]) -> str:
        content = update.get("content")
        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content["text"]
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts)
        message = update.get("message")
        if isinstance(message, str):
            return message
        return ""


_BACKGROUND_SUBAGENT_MANAGER = BackgroundSubagentManager()


def get_background_subagent_manager() -> BackgroundSubagentManager:
    return _BACKGROUND_SUBAGENT_MANAGER
