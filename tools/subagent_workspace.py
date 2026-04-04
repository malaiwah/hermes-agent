"""Workspace visibility policies for delegated subagents.

This module turns high-level child workspace requests into task-specific
terminal overrides that the Docker backend can honor safely.
"""

from __future__ import annotations

import os
import posixpath
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


ALLOWED_WORKSPACE_VISIBILITY = frozenset({
    "inherit",
    "full_rw",
    "full_ro",
    "temp_rw",
    "mapped",
})

MAX_WORKSPACE_MAPPINGS = 8


def resolve_parent_workspace_root() -> Path:
    """Return the parent workspace root on the host filesystem."""
    raw = os.getenv("TERMINAL_CWD") or os.getcwd()
    root = Path(raw).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(
            f"Cannot use delegated workspace visibility: parent workspace root is not a directory: {root}"
        )
    return root


def resolve_terminal_backend() -> str:
    """Resolve the configured terminal backend without importing large surfaces eagerly."""
    try:
        from tools.terminal_tool import _get_env_config

        return str(_get_env_config().get("env_type") or "local").strip() or "local"
    except Exception:
        return (os.getenv("TERMINAL_ENV") or "local").strip() or "local"


def build_workspace_overrides(
    visibility: Optional[str],
    mappings: Optional[List[Dict[str, Any]]],
    workspace_root: Path,
    child_token: str,
    backend: str,
) -> Dict[str, Any]:
    """Build terminal overrides and prompt text for a delegated child."""
    mode = str(visibility or "inherit").strip() or "inherit"
    if mode not in ALLOWED_WORKSPACE_VISIBILITY:
        allowed = ", ".join(sorted(ALLOWED_WORKSPACE_VISIBILITY))
        raise ValueError(f"Invalid workspace_visibility '{mode}'. Expected one of: {allowed}.")

    if mode != "mapped" and mappings:
        raise ValueError("workspace_mappings may only be used when workspace_visibility='mapped'.")

    if mode == "inherit":
        return {"prompt_note": "", "task_env_overrides": None}

    if backend != "docker":
        raise ValueError(
            f"workspace_visibility='{mode}' requires the docker terminal backend; current backend is '{backend}'."
        )

    workspace_root = workspace_root.resolve()
    if mode == "full_rw":
        return _build_mount_override(
            host_path=workspace_root,
            container_path="/workspace",
            read_only=False,
            prompt_note=(
                "WORKSPACE VISIBILITY:\n"
                f"- You can access the parent workspace at /workspace (host root: {workspace_root}).\n"
                "- The mount is read-write."
            ),
        )

    if mode == "full_ro":
        return _build_mount_override(
            host_path=workspace_root,
            container_path="/workspace",
            read_only=True,
            prompt_note=(
                "WORKSPACE VISIBILITY:\n"
                f"- You can access the parent workspace at /workspace (host root: {workspace_root}).\n"
                "- The mount is read-only. Copy files elsewhere before editing."
            ),
        )

    if mode == "temp_rw":
        base_dir = workspace_root / ".hermes-subagents"
        base_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix=f"{child_token[:24]}-",
                dir=str(base_dir),
            )
        ).resolve()
        rel_path = temp_dir.relative_to(workspace_root)
        return _build_mount_override(
            host_path=temp_dir,
            container_path="/workspace",
            read_only=False,
            prompt_note=(
                "WORKSPACE VISIBILITY:\n"
                f"- You only see an isolated writable subworkspace at /workspace.\n"
                f"- Host path: {rel_path}.\n"
                "- You cannot see the rest of the parent workspace."
            ),
        )

    normalized_mappings = _normalize_mappings(mappings or [], workspace_root)
    if not normalized_mappings:
        raise ValueError("workspace_visibility='mapped' requires at least one workspace_mappings entry.")

    volume_specs = []
    visible_paths = []
    for item in normalized_mappings:
        suffix = ":ro" if item["read_only"] else ""
        volume_specs.append(f"{item['host_path']}:{item['container_path']}{suffix}")
        visible_paths.append(
            f"- {item['container_path']} -> {item['host_path'].relative_to(workspace_root)}"
            + (" (read-only)" if item["read_only"] else "")
        )

    return {
        "prompt_note": "WORKSPACE VISIBILITY:\n- You only see these mapped paths:\n" + "\n".join(visible_paths),
        "task_env_overrides": {
            "cwd": "/workspace",
            "host_cwd": None,
            "docker_mount_cwd_to_workspace": False,
            "docker_volumes": volume_specs,
        },
    }


def _build_mount_override(
    *,
    host_path: Path,
    container_path: str,
    read_only: bool,
    prompt_note: str,
) -> Dict[str, Any]:
    suffix = ":ro" if read_only else ""
    return {
        "prompt_note": prompt_note,
        "task_env_overrides": {
            "cwd": container_path,
            "host_cwd": None,
            "docker_mount_cwd_to_workspace": False,
            "docker_volumes": [f"{host_path}:{container_path}{suffix}"],
        },
    }


def _normalize_mappings(
    mappings: List[Dict[str, Any]],
    workspace_root: Path,
) -> List[Dict[str, Any]]:
    if len(mappings) > MAX_WORKSPACE_MAPPINGS:
        raise ValueError(
            f"Too many workspace_mappings entries ({len(mappings)}). Maximum is {MAX_WORKSPACE_MAPPINGS}."
        )

    normalized = []
    for index, item in enumerate(mappings):
        if not isinstance(item, dict):
            raise ValueError(f"workspace_mappings[{index}] must be an object.")

        source = str(item.get("source") or item.get("host_path") or "").strip()
        if not source:
            raise ValueError(f"workspace_mappings[{index}] is missing 'source'.")

        host_path = _resolve_workspace_child_path(source, workspace_root)
        if not host_path.exists():
            raise ValueError(
                f"workspace_mappings[{index}] source does not exist within the workspace: {host_path}"
            )

        target = str(item.get("target") or item.get("container_path") or "").strip()
        container_path = _normalize_container_path(target, host_path.name)
        normalized.append(
            {
                "host_path": host_path,
                "container_path": container_path,
                "read_only": bool(item.get("read_only", False)),
            }
        )

    return normalized


def _resolve_workspace_child_path(raw_path: str, workspace_root: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (workspace_root / candidate).resolve()

    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(
            f"Workspace path escapes the parent workspace: {raw_path}"
        ) from exc

    return resolved


def _normalize_container_path(raw_path: str, fallback_name: str) -> str:
    target = raw_path or fallback_name
    if not target:
        target = f"mapped-{uuid.uuid4().hex[:8]}"

    if target.startswith("/"):
        normalized = posixpath.normpath(target)
    else:
        normalized = posixpath.normpath(posixpath.join("/workspace", target))

    if normalized != "/workspace" and not normalized.startswith("/workspace/"):
        raise ValueError(
            f"Mapped container path must stay within /workspace, got '{raw_path}'."
        )
    return normalized
