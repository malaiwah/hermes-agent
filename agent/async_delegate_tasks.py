"""Background async delegate_task runtime."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_ASYNC_DELEGATE_CONFIG: dict[str, Any] = {
    "enabled": True,
    "max_per_session": 2,
    "max_global": 4,
    "idle_timeout_seconds": 900,
    "max_duration_seconds": 1800,
    "output_dir": ".hermes-async-delegates",
}

_ACTIVE_STATUSES = {"starting", "running", "idle"}
_TERMINAL_STATUSES = {"completed", "timed_out", "error"}


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


def _load_async_delegate_config() -> dict[str, Any]:
    runtime_cfg: dict[str, Any] = {}
    try:
        from cli import CLI_CONFIG

        runtime_cfg = dict(((CLI_CONFIG or {}).get("delegation", {}) or {}).get("async_subagents", {}) or {})
    except Exception:
        runtime_cfg = {}

    persistent_cfg: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        persistent_cfg = dict(((load_config() or {}).get("delegation", {}) or {}).get("async_subagents", {}) or {})
    except Exception:
        persistent_cfg = {}

    return _deep_merge(_deep_merge(_DEFAULT_ASYNC_DELEGATE_CONFIG, persistent_cfg), runtime_cfg)


def _truncate(text: str, limit: int = 120) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _display_name(delegate_id: str) -> str:
    short = delegate_id.replace("async-delegate-", "")[:8]
    return f"delegate-{short}"


def _resolve_workspace_root(parent_agent) -> Path:
    candidates = [
        getattr(parent_agent, "_current_workspace", None),
        os.getenv("TERMINAL_CWD"),
        os.getcwd(),
    ]
    for item in candidates:
        if not item:
            continue
        try:
            path = Path(str(item)).expanduser()
            if path.is_dir():
                return path.resolve()
        except Exception:
            continue
    return Path(os.getcwd()).resolve()


@dataclass
class AsyncDelegateRecord:
    id: str
    name: str
    owner_session_id: str
    goal: str
    context: str
    output_file: str
    status: str
    started_at: float
    last_activity_at: float
    idle_timeout_at: float
    max_duration_at: float
    child_session_id: str = ""
    profile: str = ""
    toolsets: list[str] = field(default_factory=list)
    final_result: dict[str, Any] | None = None
    last_error: str = ""
    nudged_completion: bool = False
    nudged_timeout: bool = False
    thread: threading.Thread | None = None
    child: Any = None
    activity_log: list[str] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def is_alive(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def summary(self, now: float | None = None, idle_timeout_seconds: int = 900) -> dict[str, Any]:
        now_ts = now or _now()
        effective_status = self.status
        if self.status == "running" and now_ts >= self.last_activity_at + idle_timeout_seconds:
            effective_status = "idle"
        return {
            "id": self.id,
            "name": self.name,
            "status": effective_status,
            "goal": self.goal,
            "output_file": self.output_file,
            "started_at": self.started_at,
            "last_activity_at": self.last_activity_at,
            "max_duration_at": self.max_duration_at,
            "child_session_id": self.child_session_id,
            "profile": self.profile,
            "toolsets": list(self.toolsets),
        }


class AsyncDelegateManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._records: dict[str, AsyncDelegateRecord] = {}
        self._pending_nudges: dict[str, list[str]] = {}
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop = threading.Event()

    def reset_for_tests(self) -> None:
        with self._lock:
            records = list(self._records.values())
            self._records.clear()
            self._pending_nudges.clear()
        for record in records:
            child = record.child
            if child and hasattr(child, "interrupt"):
                try:
                    child.interrupt()
                except Exception:
                    pass

    def spawn(
        self,
        *,
        owner_session_id: str,
        parent_agent,
        goal: str,
        context: str = "",
        toolsets: list[str] | None = None,
        profile: str | None = None,
        max_iterations: int | None = None,
        creds: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = _load_async_delegate_config()
        if not cfg.get("enabled", True):
            return {"success": False, "error": "Async delegate subagents are disabled in config."}

        goal = str(goal or "").strip()
        if not goal:
            return {"success": False, "error": "goal is required."}
        if getattr(parent_agent, "_delegate_depth", 0) >= 1:
            return {"success": False, "error": "Async delegate_task is only supported from the top-level agent."}

        with self._lock:
            session_active = [
                record for record in self._records.values()
                if record.owner_session_id == owner_session_id and record.status in _ACTIVE_STATUSES and record.is_alive()
            ]
            global_active = [
                record for record in self._records.values()
                if record.status in _ACTIVE_STATUSES and record.is_alive()
            ]
            if len(session_active) >= int(cfg.get("max_per_session", 2)):
                return {
                    "success": False,
                    "error": "Async delegate limit reached for this session.",
                    "active_delegates": [record.summary(idle_timeout_seconds=int(cfg.get("idle_timeout_seconds", 900))) for record in session_active],
                }
            if len(global_active) >= int(cfg.get("max_global", 4)):
                return {
                    "success": False,
                    "error": "Global async delegate limit reached.",
                    "active_delegates": [record.summary(idle_timeout_seconds=int(cfg.get("idle_timeout_seconds", 900))) for record in session_active],
                }

        workspace_root = _resolve_workspace_root(parent_agent)
        output_dir = workspace_root / str(cfg.get("output_dir") or ".hermes-async-delegates")
        output_dir.mkdir(parents=True, exist_ok=True)
        delegate_id = f"async-delegate-{uuid.uuid4().hex[:10]}"
        output_file = output_dir / f"{delegate_id}.md"

        from tools.delegate_tool import (
            _build_child_agent,
            _load_config as _load_delegate_cfg,
            _resolve_delegation_profile,
            _run_single_child,
        )
        import model_tools as _model_tools

        delegate_cfg = _load_delegate_cfg()
        try:
            resolved_profile = _resolve_delegation_profile(delegate_cfg, profile)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        parent_tool_names = list(getattr(_model_tools, "_last_resolved_tool_names", []))
        try:
            child = _build_child_agent(
                task_index=0,
                goal=goal,
                context=context,
                toolsets=toolsets,
                profile=resolved_profile,
                model=(creds or {}).get("model"),
                max_iterations=max_iterations or delegate_cfg.get("max_iterations", 50),
                parent_agent=parent_agent,
                override_provider=(creds or {}).get("provider"),
                override_base_url=(creds or {}).get("base_url"),
                override_api_key=(creds or {}).get("api_key"),
                override_api_mode=(creds or {}).get("api_mode"),
            )
        finally:
            _model_tools._last_resolved_tool_names = parent_tool_names
        child._delegate_saved_tool_names = parent_tool_names

        # Async delegates should share the current workspace inside docker sandboxes.
        terminal_overrides = dict(getattr(child, "_delegate_terminal_overrides", None) or {})
        if terminal_overrides.get("env_type") == "docker":
            terminal_overrides.setdefault("docker_mount_cwd_to_workspace", True)
            terminal_overrides.setdefault("cwd", "/workspace")
        child._delegate_terminal_overrides = terminal_overrides

        record = AsyncDelegateRecord(
            id=delegate_id,
            name=_display_name(delegate_id),
            owner_session_id=owner_session_id,
            goal=goal,
            context=context,
            output_file=str(output_file),
            status="running",
            started_at=_now(),
            last_activity_at=_now(),
            idle_timeout_at=_now() + int(cfg.get("idle_timeout_seconds", 900)),
            max_duration_at=_now() + int(cfg.get("max_duration_seconds", 1800)),
            profile=str(profile or ""),
            toolsets=list(toolsets or []),
            child=child,
        )

        child.tool_progress_callback = self._build_progress_callback(record)
        self._write_record_file(record)

        thread = threading.Thread(
            target=self._run_delegate_thread,
            args=(record, parent_agent, _run_single_child),
            daemon=True,
            name=f"async-delegate-{delegate_id}",
        )
        record.thread = thread

        with self._lock:
            self._records[record.id] = record
            self._ensure_monitor_thread()

        thread.start()
        return {
            "success": True,
            "mode": "async",
            "id": record.id,
            "name": record.name,
            "status": "running",
            "goal": record.goal,
            "output_file": record.output_file,
            "workspace_root": str(workspace_root),
            "max_duration_at": record.max_duration_at,
        }

    def render_turn_context(self, owner_session_id: str) -> str:
        cfg = _load_async_delegate_config()
        idle_timeout_seconds = int(cfg.get("idle_timeout_seconds", 900))
        with self._lock:
            nudges = list(self._pending_nudges.get(owner_session_id, []))
            if nudges:
                self._pending_nudges[owner_session_id].clear()
            active = [
                record for record in self._records.values()
                if record.owner_session_id == owner_session_id and record.status in _ACTIVE_STATUSES and record.is_alive()
            ]

        if not nudges and not active:
            return ""

        parts: list[str] = []
        if nudges:
            parts.append(
                "Async delegated subagent updates since your last turn:\n"
                + "\n".join(f"- {item}" for item in nudges)
            )
        if active:
            lines = []
            now_ts = _now()
            for record in sorted(active, key=lambda item: item.started_at):
                summary = record.summary(now=now_ts, idle_timeout_seconds=idle_timeout_seconds)
                lines.append(
                    f"- {summary['name']} ({summary['id']}) [{summary['status']}] "
                    f"goal={_truncate(summary['goal'], 120)} output_file={summary['output_file']}"
                )
            parts.append(
                "Open async delegate_task subagents still running in the background:\n"
                + "\n".join(lines)
            )
        return "\n\n".join(parts)

    def _build_progress_callback(self, record: AsyncDelegateRecord):
        def _callback(tool_name: str, preview: str = None, args: dict | None = None):
            text = tool_name
            if preview:
                text = f"{tool_name}: {_truncate(preview, 100)}"
            with record.lock:
                record.last_activity_at = _now()
                record.idle_timeout_at = record.last_activity_at + int(_load_async_delegate_config().get("idle_timeout_seconds", 900))
                if record.status not in _TERMINAL_STATUSES:
                    record.status = "running"
                record.activity_log.append(text)
                self._write_record_file_locked(record)
        return _callback

    def _run_delegate_thread(self, record: AsyncDelegateRecord, parent_agent, run_single_child) -> None:
        result = run_single_child(
            task_index=0,
            goal=record.goal,
            child=record.child,
            parent_agent=parent_agent,
        )
        with record.lock:
            record.final_result = result
            record.child_session_id = getattr(record.child, "session_id", "") or ""
            if record.status == "timed_out":
                record.activity_log.append("Subagent exceeded maximum runtime and was interrupted.")
            elif result.get("status") == "completed":
                record.status = "completed"
                self._queue_nudge(record.owner_session_id, f"{record.name} ({record.id}) completed. Read {record.output_file} for the summary.")
            else:
                record.status = "error"
                record.last_error = str(result.get("error") or "Subagent failed.")
                self._queue_nudge(record.owner_session_id, f"{record.name} ({record.id}) failed: {record.last_error}")
            if result.get("summary"):
                record.activity_log.append("Final summary written below.")
            self._write_record_file_locked(record)

    def _ensure_monitor_thread(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="async-delegate-monitor",
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(1.0):
            self._sweep_once()
            with self._lock:
                if not any(record.status in _ACTIVE_STATUSES and record.is_alive() for record in self._records.values()):
                    self._monitor_stop.set()

    def _sweep_once(self) -> None:
        now_ts = _now()
        with self._lock:
            records = list(self._records.values())
        for record in records:
            if record.status in _TERMINAL_STATUSES or not record.is_alive():
                continue
            with record.lock:
                if now_ts >= record.max_duration_at and not record.nudged_timeout:
                    record.status = "timed_out"
                    record.nudged_timeout = True
                    record.activity_log.append("Maximum runtime reached. Interrupt requested.")
                    child = record.child
                    if child and hasattr(child, "interrupt"):
                        try:
                            child.interrupt()
                        except Exception:
                            logger.debug("Failed to interrupt async delegate child", exc_info=True)
                    self._queue_nudge(record.owner_session_id, f"{record.name} ({record.id}) hit its maximum runtime.")
                    self._write_record_file_locked(record)

    def _queue_nudge(self, owner_session_id: str, text: str) -> None:
        with self._lock:
            self._pending_nudges.setdefault(owner_session_id, []).append(text)

    def _write_record_file(self, record: AsyncDelegateRecord) -> None:
        with record.lock:
            self._write_record_file_locked(record)

    def _write_record_file_locked(self, record: AsyncDelegateRecord) -> None:
        path = Path(record.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        body_lines = [
            f"# Async Delegate {record.name}",
            "",
            f"- ID: `{record.id}`",
            f"- Status: `{record.status}`",
            f"- Goal: {record.goal}",
            f"- Started At: `{record.started_at}`",
            f"- Last Activity At: `{record.last_activity_at}`",
            f"- Max Duration At: `{record.max_duration_at}`",
        ]
        if record.child_session_id:
            body_lines.append(f"- Child Session ID: `{record.child_session_id}`")
        if record.profile:
            body_lines.append(f"- Profile: `{record.profile}`")
        if record.toolsets:
            body_lines.append(f"- Toolsets: `{', '.join(record.toolsets)}`")

        body_lines.extend(["", "## Progress", ""])
        if record.activity_log:
            body_lines.extend(f"- {line}" for line in record.activity_log[-50:])
        else:
            body_lines.append("- Subagent started.")

        body_lines.extend(["", "## Result", ""])
        if record.final_result and record.final_result.get("summary"):
            body_lines.append(str(record.final_result["summary"]))
        elif record.last_error:
            body_lines.append(f"Error: {record.last_error}")
        else:
            body_lines.append("Still running.")

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text("\n".join(body_lines) + "\n", encoding="utf-8")
        tmp_path.replace(path)


_ASYNC_DELEGATE_MANAGER = AsyncDelegateManager()


def get_async_delegate_manager() -> AsyncDelegateManager:
    return _ASYNC_DELEGATE_MANAGER
