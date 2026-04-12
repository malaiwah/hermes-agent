"""Podman execution environment for sandboxed command execution.

Extends DockerEnvironment with Podman-specific options: rootful/rootless mode,
user namespace mapping, privileged mode, extra capabilities, and extra CLI args.

Security: inherits Docker's cap-drop-ALL baseline by default.  When
``privileged=True``, capabilities are unrestricted (use with care).
"""

import logging
import os
import shutil
import subprocess
from typing import Optional

from tools.environments.docker import DockerEnvironment, _SECURITY_ARGS

logger = logging.getLogger(__name__)


# Common Podman install paths checked when 'podman' is not in PATH.
_PODMAN_SEARCH_PATHS = [
    "/usr/bin/podman",
    "/usr/local/bin/podman",
    "/usr/bin/podman-remote",
    "/usr/local/bin/podman-remote",
    "/opt/homebrew/bin/podman",
    "/opt/podman/bin/podman",
    "/home/linuxbrew/.linuxbrew/bin/podman",
]

_podman_executable: Optional[str] = None  # resolved once, cached


def find_podman() -> Optional[str]:
    """Locate the podman CLI binary.

    Checks ``shutil.which`` first (respects PATH), then probes well-known
    install locations where Podman may not be in PATH.  Also checks for
    ``podman-remote`` which is the correct binary for gateway deployments
    where the hermes container talks to a host Podman socket.
    """
    global _podman_executable
    if _podman_executable is not None:
        return _podman_executable

    # Check both podman and podman-remote in PATH
    for name in ("podman", "podman-remote"):
        found = shutil.which(name)
        if found:
            _podman_executable = found
            return found

    for path in _PODMAN_SEARCH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            _podman_executable = path
            logger.info("Found podman at non-PATH location: %s", path)
            return path

    return None


def _ensure_podman_available() -> None:
    """Verify the podman CLI is available and functional."""
    podman_exe = find_podman()
    if not podman_exe:
        raise RuntimeError(
            "Podman executable not found in PATH or known install locations. "
            "Install Podman and ensure the 'podman' command is available."
        )
    try:
        result = subprocess.run(
            [podman_exe, "version"],
            capture_output=True, text=True, timeout=5,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Podman executable '{podman_exe}' could not be executed. "
            "Check your Podman installation."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("`podman version` is not responding.")
    if result.returncode != 0:
        raise RuntimeError(
            f"'podman version' failed (exit {result.returncode}): "
            f"{result.stderr.strip()[:200]}"
        )


class PodmanEnvironment(DockerEnvironment):
    """Podman container execution environment.

    Inherits all Docker workspace, volume, credential-mount, and lifecycle
    logic.  Adds Podman-specific options for user namespace mapping, rootful
    mode, privileged containers, and extra capabilities/args.
    """

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        cpu: float = 0,
        memory: int = 0,
        disk: int = 0,
        persistent_filesystem: bool = False,
        task_id: str = "default",
        volumes: list = None,
        forward_env: list[str] | None = None,
        env: dict | None = None,
        network: bool = True,
        host_cwd: str = None,
        auto_mount_cwd: bool = False,
        # Podman-specific options
        rootful: bool = False,
        privileged: bool = False,
        userns: str = "",
        extra_capabilities: list[str] | None = None,
        extra_args: list[str] | None = None,
    ):
        # Validate extra_args before anything starts
        if extra_args and not all(isinstance(x, str) for x in extra_args):
            logger.warning("podman_extra_args contains non-string entries; ignoring")
            extra_args = None
        if extra_capabilities and not all(isinstance(x, str) for x in extra_capabilities):
            logger.warning("podman_extra_capabilities contains non-string entries; ignoring")
            extra_capabilities = None

        self._rootful = rootful
        self._privileged = privileged
        self._userns = userns.strip() if userns else ""
        self._extra_capabilities = extra_capabilities or []
        self._extra_args = extra_args or []

        # DockerEnvironment.__init__ calls our overridden hook methods
        # (_resolve_cli_binary, _ensure_cli_available, _get_security_args,
        # _get_extra_run_args, _build_run_cmd) so Podman-specific config
        # must be set BEFORE calling super().__init__.
        super().__init__(
            image=image, cwd=cwd, timeout=timeout,
            cpu=cpu, memory=memory, disk=disk,
            persistent_filesystem=persistent_filesystem,
            task_id=task_id, volumes=volumes,
            forward_env=forward_env, env=env,
            network=network, host_cwd=host_cwd,
            auto_mount_cwd=auto_mount_cwd,
        )

    # ── Hook method overrides ───────────────────────────────────────

    def _ensure_cli_available(self) -> None:
        _ensure_podman_available()

    def _resolve_cli_binary(self) -> str:
        return find_podman() or "podman"

    def _get_security_args(self) -> list[str]:
        if self._privileged:
            return ["--privileged"]
        return list(_SECURITY_ARGS)

    def _get_extra_run_args(self) -> list[str]:
        args: list[str] = []
        if self._userns:
            args.extend(["--userns", self._userns])
        for cap in self._extra_capabilities:
            args.extend(["--cap-add", cap])
        args.extend(self._extra_args)
        return args

    def _build_run_cmd(
        self, container_name: str, cwd: str, all_run_args: list[str], image: str,
    ) -> list[str]:
        cmd = [
            self._docker_exe, "run", "-d",
            "--init",           # reap zombie children (same as Docker backend)
            "--name", container_name,
            "-w", cwd,
            *all_run_args,
            image,
            "sleep", "infinity",
        ]
        if self._rootful:
            cmd = ["sudo"] + cmd
        return cmd

    # ── Overrides that need rootful sudo prefix ─────────────────────

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Spawn a bash process inside the Podman container."""
        from tools.environments.base import _popen_bash

        assert self._container_id, "Container not started"
        cmd = [self._docker_exe, "exec"]

        if self._rootful:
            cmd = ["sudo"] + cmd

        if stdin_data is not None:
            cmd.append("-i")

        if login:
            cmd.extend(self._init_env_args)

        cmd.extend([self._container_id])

        if login:
            cmd.extend(["bash", "-l", "-c", cmd_string])
        else:
            cmd.extend(["bash", "-c", cmd_string])

        return _popen_bash(cmd, stdin_data)

    def cleanup(self):
        """Stop and remove the container, with sudo prefix when rootful."""
        if self._container_id:
            sudo = "sudo " if self._rootful else ""
            try:
                stop_cmd = (
                    f"(timeout 60 {sudo}{self._docker_exe} stop {self._container_id} || "
                    f"{sudo}{self._docker_exe} rm -f {self._container_id}) >/dev/null 2>&1 &"
                )
                subprocess.Popen(stop_cmd, shell=True)
            except Exception as e:
                logger.warning("Failed to stop container %s: %s", self._container_id, e)

            if not self._persistent:
                try:
                    subprocess.Popen(
                        f"sleep 3 && {sudo}{self._docker_exe} rm -f {self._container_id} >/dev/null 2>&1 &",
                        shell=True,
                    )
                except Exception:
                    pass
            self._container_id = None

        if not self._persistent:
            import shutil as _shutil
            for d in (self._workspace_dir, self._home_dir):
                if d:
                    _shutil.rmtree(d, ignore_errors=True)
