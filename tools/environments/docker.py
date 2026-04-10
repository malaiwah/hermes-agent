"""Docker execution environment for sandboxed command execution.

Security hardened (cap-drop ALL, no-new-privileges, PID limits),
configurable resource limits (CPU, memory, disk), and optional filesystem
persistence via bind mounts.
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import shlex
import uuid
from typing import Optional

from tools.environments.base import BaseEnvironment, _popen_bash
from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)


# Common Docker Desktop install paths checked when 'docker' is not in PATH.
# macOS Intel: /usr/local/bin, macOS Apple Silicon (Homebrew): /opt/homebrew/bin,
# Docker Desktop app bundle: /Applications/Docker.app/Contents/Resources/bin
_DOCKER_SEARCH_PATHS = [
    "/usr/local/bin/docker",
    "/opt/homebrew/bin/docker",
    "/Applications/Docker.app/Contents/Resources/bin/docker",
]

_docker_executable: Optional[str] = None  # resolved once, cached
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_forward_env_names(forward_env: list[str] | None) -> list[str]:
    """Return a deduplicated list of valid environment variable names."""
    normalized: list[str] = []
    seen: set[str] = set()

    for item in forward_env or []:
        if not isinstance(item, str):
            logger.warning("Ignoring non-string docker_forward_env entry: %r", item)
            continue

        key = item.strip()
        if not key:
            continue
        if not _ENV_VAR_NAME_RE.match(key):
            logger.warning("Ignoring invalid docker_forward_env entry: %r", item)
            continue
        if key in seen:
            continue

        seen.add(key)
        normalized.append(key)

    return normalized


def _normalize_env_dict(env: dict | None) -> dict[str, str]:
    """Validate and normalize a docker_env dict to {str: str}.

    Filters out entries with invalid variable names or non-string values.
    """
    if not env:
        return {}
    if not isinstance(env, dict):
        logger.warning("docker_env is not a dict: %r", env)
        return {}

    normalized: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_VAR_NAME_RE.match(key.strip()):
            logger.warning("Ignoring invalid docker_env key: %r", key)
            continue
        key = key.strip()
        if not isinstance(value, str):
            # Coerce simple scalar types (int, bool, float) to string;
            # reject complex types.
            if isinstance(value, (int, float, bool)):
                value = str(value)
            else:
                logger.warning("Ignoring non-string docker_env value for %r: %r", key, value)
                continue
        normalized[key] = value

    return normalized


def _load_hermes_env_vars() -> dict[str, str]:
    """Load ~/.hermes/.env values without failing Docker command execution."""
    try:
        from hermes_cli.config import load_env

        return load_env() or {}
    except Exception:
        return {}


def find_docker() -> Optional[str]:
    """Locate the docker CLI binary.

    Checks ``shutil.which`` first (respects PATH), then probes well-known
    install locations on macOS where Docker Desktop may not be in PATH
    (e.g. when running as a gateway service via launchd).

    Returns the absolute path, or ``None`` if docker cannot be found.
    """
    global _docker_executable
    if _docker_executable is not None:
        return _docker_executable

    found = shutil.which("docker")
    if found:
        _docker_executable = found
        return found

    for path in _DOCKER_SEARCH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            _docker_executable = path
            logger.info("Found docker at non-PATH location: %s", path)
            return path

    return None


# Security flags applied to every container.
# The container itself is the security boundary (isolated from host).
# We drop all capabilities then add back the minimum needed:
#   DAC_OVERRIDE - root can write to bind-mounted dirs owned by host user
#   CHOWN/FOWNER - package managers (pip, npm, apt) need to set file ownership
# Block privilege escalation and limit PIDs.
# /tmp is size-limited and nosuid but allows exec (needed by pip/npm builds).
#
# Configurable via env vars:
#   SANDBOX_NO_NEW_PRIVS = "true" (default) | "false" — disable to allow sudo
#   SANDBOX_PIDS_LIMIT   = integer (default "256") | "0"/"off"/"none" — disable
#
# ``--pids-limit`` is added later in the run command (see ``resource_args``)
# rather than here, so it can be auto-disabled when the ``pids`` cgroup
# controller is not delegated to this process (typical inside unprivileged
# LXCs). Hardcoding it caused every container spawn to fail with
# "controller `pids` is not available" on such hosts.
_SECURITY_ARGS = [
    "--cap-drop", "ALL",
    "--cap-add", "DAC_OVERRIDE",
    "--cap-add", "CHOWN",
    "--cap-add", "FOWNER",
    "--tmpfs", "/tmp:rw,nosuid,size=512m",
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",
    "--tmpfs", "/run:rw,noexec,nosuid,size=64m",
]

# Add no-new-privileges unless explicitly disabled
if os.getenv("SANDBOX_NO_NEW_PRIVS", "true").lower() != "false":
    _SECURITY_ARGS.extend(["--security-opt", "no-new-privileges"])
else:
    logger.warning(
        "SANDBOX_NO_NEW_PRIVS=false: containers can escalate privileges via sudo. "
        "Only disable in trusted environments."
    )


_storage_opt_ok: Optional[bool] = None  # cached result across instances

_cgroup_limits_ok: Optional[bool] = None  # cached result across instances


def _cgroup_limits_available(image: str) -> bool:
    """Probe whether cgroup resource limits (--cpus/--memory/--pids-limit) work.

    Spawns a throwaway container from *image* (the same sandbox image we are
    about to use for real, so no extra pull and no dependency on a public
    registry) with all three flags. The container runs ``sleep 0`` — sleep is
    guaranteed to be present because the sandbox itself uses ``sleep 2h`` as
    its long-lived entrypoint. On hosts without cgroup controller delegation
    (typical inside unprivileged LXCs) these flags cause container startup to
    fail; we cache the boolean result host-wide so the probe runs at most once.
    """
    global _cgroup_limits_ok
    if _cgroup_limits_ok is not None:
        return _cgroup_limits_ok

    docker_exe = find_docker()
    if not docker_exe:
        _cgroup_limits_ok = False
        return False

    try:
        result = subprocess.run(
            [docker_exe, "run", "--rm",
             "--cpus", "0.5", "--memory", "64m", "--pids-limit", "32",
             image, "sleep", "0"],
            capture_output=True, text=True, timeout=60,
        )
        _cgroup_limits_ok = result.returncode == 0
        if not _cgroup_limits_ok:
            logger.warning(
                "Cgroup resource limits (--cpus/--memory/--pids-limit) not "
                "available in this environment. Containers will run without "
                "CPU, memory or PID limits. To enable, delegate cgroup "
                "controllers to this container. Probe stderr: %s",
                (result.stderr or "").strip()[:500],
            )
    except Exception as e:
        _cgroup_limits_ok = False
        logger.warning("Cgroup limit probe failed; disabling resource limits: %s", e)

    return _cgroup_limits_ok


def _resolve_pids_limit() -> Optional[str]:
    """Return the configured ``--pids-limit`` value, or None if disabled.

    Honors ``SANDBOX_PIDS_LIMIT``:
      - unset / empty → default "256"
      - "0", "off", "none", "false", "disable", "disabled" → None (no limit)
      - any other value → that value (passed to docker as-is)
    """
    raw = os.getenv("SANDBOX_PIDS_LIMIT", "256").strip()
    if not raw or raw.lower() in {"0", "off", "none", "false", "disable", "disabled"}:
        return None
    return raw


def _ensure_docker_available() -> None:
    """Best-effort check that the docker CLI is available before use.

    Reuses ``find_docker()`` so this preflight stays consistent with the rest of
    the Docker backend, including known non-PATH Docker Desktop locations.
    """
    docker_exe = find_docker()
    if not docker_exe:
        logger.error(
            "Docker backend selected but no docker executable was found in PATH "
            "or known install locations. Install Docker Desktop and ensure the "
            "CLI is available."
        )
        raise RuntimeError(
            "Docker executable not found in PATH or known install locations. "
            "Install Docker and ensure the 'docker' command is available."
        )

    try:
        result = subprocess.run(
            [docker_exe, "version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        logger.error(
            "Docker backend selected but the resolved docker executable '%s' could "
            "not be executed.",
            docker_exe,
            exc_info=True,
        )
        raise RuntimeError(
            "Docker executable could not be executed. Check your Docker installation."
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "Docker backend selected but '%s version' timed out. "
            "The Docker daemon may not be running.",
            docker_exe,
            exc_info=True,
        )
        raise RuntimeError(
            "Docker daemon is not responding. Ensure Docker is running and try again."
        )
    except Exception:
        logger.error(
            "Unexpected error while checking Docker availability.",
            exc_info=True,
        )
        raise
    else:
        if result.returncode != 0:
            logger.error(
                "Docker backend selected but '%s version' failed "
                "(exit code %d, stderr=%s)",
                docker_exe,
                result.returncode,
                result.stderr.strip(),
            )
            raise RuntimeError(
                "Docker command is available but 'docker version' failed. "
                "Check your Docker installation."
            )


class DockerEnvironment(BaseEnvironment):
    """Hardened Docker container execution with resource limits and persistence.

    Security: all capabilities dropped, no privilege escalation, PID limits,
    size-limited tmpfs for scratch dirs. The container itself is the security
    boundary — the filesystem inside is writable so agents can install packages
    (pip, npm, apt) as needed. Writable workspace via tmpfs or bind mounts.

    Persistence: when enabled, bind mounts preserve /workspace and /root
    across container restarts.
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
        docker_network: str | None = None,
        host_cwd: str = None,
        auto_mount_cwd: bool = False,
        extra_hosts: list[str] | None = None,
        env_files: list[str] | None = None,
        docker_user: str | None = None,
    ):
        # Resolve home directory based on user
        home_dir = f"/home/{docker_user}" if docker_user and docker_user != "root" else "/root"
        if cwd in ("~", "/root") and docker_user and docker_user != "root":
            cwd = home_dir
        elif cwd == "~":
            cwd = "/root"
        self._docker_user = docker_user
        self._home_path = home_dir
        super().__init__(cwd=cwd, timeout=timeout)
        self._base_image = image
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._forward_env = _normalize_forward_env_names(forward_env)
        self._env = _normalize_env_dict(env)

        # Inject env vars from files: format "VAR_NAME:/path/to/file".
        #
        # Each entry is parsed once at __init__, the path is canonicalized
        # via Path.resolve() (no symlink swap mid-task), validated against
        # an allowlist of safe parent directories, and stored as a tuple of
        # (var_name, resolved_host_path). The exec path then re-reads the
        # file on every ``docker exec`` call (see ``_extra_env_for_exec``)
        # so that rotating credentials propagate to the next tool call
        # without having to respawn the container. The canonical case is
        # BW_SESSION from a Bitwarden-unlock sidecar: the file on the host
        # gets rewritten when the vault is unlocked / re-unlocked, and the
        # next ``docker exec`` picks up the fresh value automatically.
        #
        # The values are *not* baked into ``self._env`` (and therefore not
        # passed to ``docker run`` either) — the long-lived container's own
        # environment stays clean, and each exec gets a freshly read copy
        # for the duration of that exec'd process only.
        self._env_files: list[tuple[str, str]] = self._parse_env_files(env_files or [])

        self._container_id: Optional[str] = None
        logger.info(f"DockerEnvironment volumes: {volumes}")
        # Ensure volumes is a list (config.yaml could be malformed)
        if volumes is not None and not isinstance(volumes, list):
            logger.warning(f"docker_volumes config is not a list: {volumes!r}")
            volumes = []

        # Fail fast if Docker is not available.
        _ensure_docker_available()

        # Build resource limit args (gated by cgroup availability probe)
        resource_args = []
        if cpu > 0 and _cgroup_limits_available(self._base_image):
            resource_args.extend(["--cpus", str(cpu)])
        if memory > 0 and _cgroup_limits_available(self._base_image):
            resource_args.extend(["--memory", f"{memory}m"])
        pids_limit = _resolve_pids_limit()
        if pids_limit is not None and _cgroup_limits_available(self._base_image):
            resource_args.extend(["--pids-limit", pids_limit])
        if disk > 0 and sys.platform != "darwin":
            if self._storage_opt_supported():
                resource_args.extend(["--storage-opt", f"size={disk}m"])
            else:
                logger.warning(
                    "Docker storage driver does not support per-container disk limits "
                    "(requires overlay2 on XFS with pquota). Container will run without disk quota."
                )
        if not network:
            resource_args.append("--network=none")
        elif docker_network:
            resource_args.extend(["--network", docker_network])

        # Persistent workspace via bind mounts from a configurable host directory
        # (TERMINAL_SANDBOX_DIR, default ~/.hermes/sandboxes/). Non-persistent
        # mode uses tmpfs (ephemeral, fast, gone on cleanup).
        from tools.environments.base import get_sandbox_dir

        # User-configured volume mounts (from config.yaml docker_volumes)
        volume_args = []
        workspace_explicitly_mounted = False
        for vol in (volumes or []):
            if not isinstance(vol, str):
                logger.warning(f"Docker volume entry is not a string: {vol!r}")
                continue
            vol = vol.strip()
            if not vol:
                continue
            if ":" in vol:
                volume_args.extend(["-v", vol])
                if ":/workspace" in vol:
                    workspace_explicitly_mounted = True
            else:
                logger.warning(f"Docker volume '{vol}' missing colon, skipping")

        host_cwd_abs = os.path.abspath(os.path.expanduser(host_cwd)) if host_cwd else ""
        bind_host_cwd = (
            auto_mount_cwd
            and bool(host_cwd_abs)
            and os.path.isdir(host_cwd_abs)
            and not workspace_explicitly_mounted
        )
        if auto_mount_cwd and host_cwd and not os.path.isdir(host_cwd_abs):
            logger.debug(f"Skipping docker cwd mount: host_cwd is not a valid directory: {host_cwd}")

        self._workspace_dir: Optional[str] = None
        self._home_dir: Optional[str] = None
        writable_args = []
        if self._persistent:
            sandbox = get_sandbox_dir() / "docker" / task_id
            self._home_dir = str(sandbox / "home")
            os.makedirs(self._home_dir, exist_ok=True)
            writable_args.extend([
                "-v", f"{self._home_dir}:{self._home_path}",
            ])
            if not bind_host_cwd and not workspace_explicitly_mounted:
                self._workspace_dir = str(sandbox / "workspace")
                os.makedirs(self._workspace_dir, exist_ok=True)
                writable_args.extend([
                    "-v", f"{self._workspace_dir}:/workspace",
                ])
        else:
            if not bind_host_cwd and not workspace_explicitly_mounted:
                writable_args.extend([
                    "--tmpfs", "/workspace:rw,exec,size=10g",
                ])
            writable_args.extend([
                "--tmpfs", "/home:rw,exec,size=1g",
                "--tmpfs", "/root:rw,exec,size=1g",
            ])

        if bind_host_cwd:
            logger.info(f"Mounting configured host cwd to /workspace: {host_cwd_abs}")
            volume_args = ["-v", f"{host_cwd_abs}:/workspace", *volume_args]
        elif workspace_explicitly_mounted:
            logger.debug("Skipping docker cwd mount: /workspace already mounted by user config")

        # Mount credential files (OAuth tokens, etc.) declared by skills.
        # Read-only so the container can authenticate but not modify host creds.
        try:
            from tools.credential_files import (
                get_credential_file_mounts,
                get_skills_directory_mount,
                get_cache_directory_mounts,
            )

            for mount_entry in get_credential_file_mounts():
                volume_args.extend([
                    "-v",
                    f"{mount_entry['host_path']}:{mount_entry['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting credential %s -> %s",
                    mount_entry["host_path"],
                    mount_entry["container_path"],
                )

            # Mount skill directories (local + external) so skill
            # scripts/templates are available inside the container.
            for skills_mount in get_skills_directory_mount():
                volume_args.extend([
                    "-v",
                    f"{skills_mount['host_path']}:{skills_mount['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting skills dir %s -> %s",
                    skills_mount["host_path"],
                    skills_mount["container_path"],
                )

            # Mount host-side cache directories (documents, images, audio,
            # screenshots) so the agent can access uploaded files and other
            # cached media from inside the container.  Read-only — the
            # container reads these but the host gateway manages writes.
            for cache_mount in get_cache_directory_mounts():
                volume_args.extend([
                    "-v",
                    f"{cache_mount['host_path']}:{cache_mount['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting cache dir %s -> %s",
                    cache_mount["host_path"],
                    cache_mount["container_path"],
                )
        except Exception as e:
            logger.debug("Docker: could not load credential file mounts: %s", e)

        # Explicit environment variables (docker_env config) — set at container
        # creation so they're available to all processes (including entrypoint).
        # Override HOME when running as a non-root user so tools resolve dotfiles correctly.
        env_args = []
        if self._docker_user and self._docker_user != "root":
            env_args.extend(["-e", f"HOME={self._home_path}"])
        for key in sorted(self._env):
            env_args.extend(["-e", f"{key}={self._env[key]}"])

        host_args = []
        for entry in (extra_hosts or []):
            host_args.extend(["--add-host", entry])

        logger.info(f"Docker volume_args: {volume_args}")
        user_args = ["--user", self._docker_user] if self._docker_user else []
        all_run_args = list(_SECURITY_ARGS) + user_args + writable_args + resource_args + host_args + volume_args + env_args
        logger.info(f"Docker run_args: {all_run_args}")

        # Resolve the docker executable once so it works even when
        # /usr/local/bin is not in PATH (common on macOS gateway/service).
        self._docker_exe = find_docker() or "docker"

        # Start the container directly via `docker run -d`.
        container_name = f"hermes-{uuid.uuid4().hex[:8]}"
        run_cmd = [
            self._docker_exe, "run", "-d",
            "--name", container_name,
            "-w", cwd,
            *all_run_args,
            image,
            "sleep", "2h",
        ]
        logger.debug(f"Starting container: {' '.join(run_cmd)}")
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            timeout=120,  # image pull may take a while
            check=True,
        )
        self._container_id = result.stdout.strip()
        logger.info(f"Started container {container_name} ({self._container_id[:12]})")

    # Maximum size of a single env_files value. Linux's `execve` accepts at
    # most ARG_MAX bytes total across all argv + envp; per-entry size is
    # bounded by the same limit. We cap individual values at 64 KiB so a
    # buggy or malicious sidecar that writes a huge file fails fast with a
    # clear log line instead of a confusing E2BIG when the actual exec runs.
    _ENV_FILES_MAX_SIZE = 64 * 1024

    # Allowlist of safe parent directories for `docker_env_files` paths,
    # checked at parse time after symlink resolution. The intent is
    # defense-in-depth against a config that says "X:/etc/shadow" — for the
    # canonical sidecar use case the file lives in /run/hermes-creds, the
    # XDG runtime dir, or HERMES_HOME. Operators with unusual layouts can
    # extend this via TERMINAL_DOCKER_ENV_FILES_ALLOWED_DIRS (colon-
    # separated). Empty allowlist disables the check entirely (escape hatch
    # for tests and operators who really know what they're doing).
    @staticmethod
    def _env_files_allowed_dirs() -> "list[Path]":
        from pathlib import Path
        override = os.getenv("TERMINAL_DOCKER_ENV_FILES_ALLOWED_DIRS")
        if override is not None:
            return [Path(p).resolve() for p in override.split(":") if p.strip()]
        candidates = [
            "/run/hermes-creds",
            "/run/secrets",
            os.getenv("XDG_RUNTIME_DIR") or "",
        ]
        try:
            from hermes_constants import get_hermes_home
            candidates.append(str(get_hermes_home()))
        except Exception:
            pass
        return [Path(c).resolve() for c in candidates if c]

    @classmethod
    def _parse_env_files(cls, entries: list[str]) -> list[tuple[str, str]]:
        """Parse and validate ``docker_env_files`` config entries.

        Each entry is ``"VAR_NAME:/host/path"``. Returns a list of
        ``(var_name, resolved_path)`` tuples. Invalid entries are logged
        and skipped (non-fatal — the agent should still start even if one
        credential source is misconfigured).

        Validation:
        - Format must be ``VAR:path`` with at least one ``:``.
        - Path is resolved via ``Path.resolve()`` (follows symlinks once,
          canonicalises) so that subsequent rewrites of the symlink target
          cannot redirect reads at exec time.
        - Resolved path must be inside one of the allowed parent
          directories (see ``_env_files_allowed_dirs``). Empty allowlist
          disables the check.
        - Path does not need to exist at parse time — sidecar may not have
          written the file yet. Existence is rechecked at exec time.
        """
        from pathlib import Path
        parsed: list[tuple[str, str]] = []
        allowed = cls._env_files_allowed_dirs()
        for entry in entries:
            try:
                var_name, raw_path = entry.split(":", 1)
            except ValueError:
                logger.warning(
                    "docker_env_files: invalid entry %r, expected 'VAR:path'",
                    entry,
                )
                continue
            var_name = var_name.strip()
            if not var_name:
                logger.warning("docker_env_files: empty var name in %r", entry)
                continue
            if not raw_path.strip():
                logger.warning("docker_env_files: empty path for %s", var_name)
                continue

            # Resolve the path (follows symlinks, canonicalises). strict=False
            # so missing files don't error — the sidecar may not have written
            # the file yet at hermes-agent startup time.
            try:
                resolved = Path(raw_path).resolve(strict=False)
            except (OSError, RuntimeError) as e:
                logger.warning(
                    "docker_env_files: could not resolve %s for %s: %s",
                    raw_path, var_name, e,
                )
                continue

            # Check the resolved path is inside an allowed directory.
            if allowed:
                ok = any(
                    resolved == d or d in resolved.parents
                    for d in allowed
                )
                if not ok:
                    logger.warning(
                        "docker_env_files: rejecting %s (resolves to %s, outside "
                        "allowed dirs %s — set TERMINAL_DOCKER_ENV_FILES_ALLOWED_DIRS "
                        "to override)",
                        var_name, resolved,
                        ", ".join(str(d) for d in allowed),
                    )
                    continue

            parsed.append((var_name, str(resolved)))
            logger.info(
                "docker_env_files: registered %s ← %s", var_name, resolved,
            )
        return parsed

    def _extra_env_for_exec(self) -> dict[str, str]:
        """Return env vars to overlay onto every ``docker exec`` invocation.

        Hook for per-exec dynamic env injection. The default implementation
        re-reads the files registered via ``docker_env_files`` so that
        rotating credentials propagate to the next tool call without
        requiring the sandbox to respawn.

        Subclasses or sibling subsystems (the credential registry being the
        canonical example) can override this to inject additional values
        from any source. Failures are non-fatal: the offending entry is
        skipped with a warning, the rest still get applied.
        """
        out: dict[str, str] = {}
        # Defensive getattr: tests may construct DockerEnvironment via
        # __new__ without going through __init__, in which case _env_files
        # is unset. Treat that as "no entries" rather than crashing.
        for var_name, file_path in getattr(self, "_env_files", []) or []:
            value = self._read_env_file_value(var_name, file_path)
            if value is not None:
                out[var_name] = value
        return out

    @classmethod
    def _read_env_file_value(cls, var_name: str, file_path: str) -> Optional[str]:
        """Read one credential file with size cap + minimal newline trim.

        Returns the value, or None on any error (logged at WARNING).

        - Caps reads at ``_ENV_FILES_MAX_SIZE`` (64 KiB). Larger files are
          rejected with an explicit error rather than failing later inside
          ``execve`` with a confusing E2BIG.
        - Trims a single trailing newline (and only that — `.strip()` would
          corrupt PEM bodies, JSON blobs, or any value with significant
          leading whitespace). The trailing newline trim handles the common
          ``echo $value > file`` shell pattern.
        - Does NOT re-resolve symlinks at read time. The path was canonicalized
          at parse time and stored absolute; reads always go to the resolved
          target.
        """
        try:
            with open(file_path, "rb") as fh:
                data = fh.read(cls._ENV_FILES_MAX_SIZE + 1)
        except OSError as e:
            logger.warning(
                "docker_env_files: could not read %s for %s on exec; skipping (%s)",
                file_path, var_name, e,
            )
            return None
        if len(data) > cls._ENV_FILES_MAX_SIZE:
            logger.warning(
                "docker_env_files: %s exceeds %d byte limit (file %s); skipping",
                var_name, cls._ENV_FILES_MAX_SIZE, file_path,
            )
            return None
        try:
            value = data.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning(
                "docker_env_files: %s is not valid UTF-8 (file %s); skipping (%s)",
                var_name, file_path, e,
            )
            return None
        # Strip exactly one trailing newline (the common `echo > file` case),
        # nothing else. PEM bodies, JSON blobs, and base64 with leading
        # whitespace must round-trip unchanged.
        if value.endswith("\r\n"):
            value = value[:-2]
        elif value.endswith("\n"):
            value = value[:-1]
        return value

    @staticmethod
    def _storage_opt_supported() -> bool:
        """Check if Docker's storage driver supports --storage-opt size=.
        
        Only overlay2 on XFS with pquota supports per-container disk quotas.
        Ubuntu (and most distros) default to ext4, where this flag errors out.
        """
        global _storage_opt_ok
        if _storage_opt_ok is not None:
            return _storage_opt_ok
        try:
            docker = find_docker() or "docker"
            result = subprocess.run(
                [docker, "info", "--format", "{{.Driver}}"],
                capture_output=True, text=True, timeout=10,
            )
            driver = result.stdout.strip().lower()
            if driver != "overlay2":
                _storage_opt_ok = False
                return False
            # overlay2 only supports storage-opt on XFS with pquota.
            # Probe by attempting a dry-ish run — the fastest reliable check.
            probe = subprocess.run(
                [docker, "create", "--storage-opt", "size=1m", "hello-world"],
                capture_output=True, text=True, timeout=15,
            )
            if probe.returncode == 0:
                # Clean up the created container
                container_id = probe.stdout.strip()
                if container_id:
                    subprocess.run([docker, "rm", container_id],
                                   capture_output=True, timeout=5)
                _storage_opt_ok = True
            else:
                _storage_opt_ok = False
        except Exception:
            _storage_opt_ok = False
        logger.debug("Docker --storage-opt support: %s", _storage_opt_ok)
        return _storage_opt_ok

    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        exec_command, sudo_stdin = self._prepare_command(command)
        work_dir = cwd or self.cwd
        effective_timeout = timeout or self.timeout

        # Merge sudo password (if any) with caller-supplied stdin_data.
        if sudo_stdin is not None and stdin_data is not None:
            effective_stdin = sudo_stdin + stdin_data
        elif sudo_stdin is not None:
            effective_stdin = sudo_stdin
        else:
            effective_stdin = stdin_data

        # docker exec -w doesn't expand ~, so prepend a cd into the command.
        # Keep ~ unquoted (for shell expansion) and quote only the subpath.
        if work_dir == "~":
            exec_command = f"cd ~ && {exec_command}"
            work_dir = "/"
        elif work_dir.startswith("~/"):
            exec_command = f"cd ~/{shlex.quote(work_dir[2:])} && {exec_command}"
            work_dir = "/"

        assert self._container_id, "Container not started"
        cmd = [self._docker_exe, "exec"]
        if effective_stdin is not None:
            cmd.append("-i")
        cmd.extend(["-w", work_dir])
        # Build the per-exec environment: start with explicit docker_env values
        # (static config), then overlay docker_forward_env / skill env_passthrough
        # (dynamic from host process).  Forward values take precedence.
        exec_env: dict[str, str] = dict(self._env)

        forward_keys = set(self._forward_env)
        try:
            from tools.env_passthrough import get_all_passthrough
            forward_keys |= get_all_passthrough()
        except Exception:
            pass
        # Strip Hermes-managed secrets so they never leak into the container.
        forward_keys -= _HERMES_PROVIDER_ENV_BLOCKLIST
        hermes_env = _load_hermes_env_vars() if forward_keys else {}
        for key in sorted(forward_keys):
            value = os.getenv(key)
            if value is None:
                value = hermes_env.get(key)
            if value is not None:
                exec_env[key] = value

        # Per-exec dynamic env injection — overlay anything the
        # ``_extra_env_for_exec`` hook returns. The default implementation
        # re-reads ``docker_env_files`` so rotating credentials (e.g.
        # BW_SESSION written by a Bitwarden sidecar) propagate to the next
        # tool call without requiring the sandbox to respawn. Subclasses
        # and sibling subsystems can override the hook to plug in any
        # other dynamic source. Failures inside the hook are non-fatal:
        # offending entries are skipped with a warning, rest are applied.
        try:
            extra = self._extra_env_for_exec()
        except Exception as e:
            logger.warning("_extra_env_for_exec raised, skipping all dynamic env: %s", e)
            extra = {}
        exec_env.update(extra)
        # Track which keys came from the dynamic-injection hook so the
        # log-line masker below can redact them by *origin* rather than
        # by name-keyword heuristic. Anything coming through the hook is
        # by construction a credential (otherwise why inject it per-exec?),
        # regardless of whether its name happens to contain "TOKEN" /
        # "SECRET" / etc.
        _dynamic_keys = set(extra.keys())

        for key in sorted(exec_env):
            cmd.extend(["-e", f"{key}={exec_env[key]}"])
        cmd.extend([self._container_id, "bash", "-lc", exec_command])

        # Log the exact exec command with secret values masked. Two
        # masking rules:
        #   1. Origin: anything from `_extra_env_for_exec` is always
        #      masked (the credential registry / docker_env_files path).
        #   2. Name heuristic: env names whose UPPERCASE form contains a
        #      sensitive token are masked. The list is conservative —
        #      false positives just over-redact a log line, false
        #      negatives leak a credential. Add more aggressively than
        #      reluctantly. SESSION/AUTH/COOKIE/JWT/BEARER/SIGNATURE/PIN
        #      were missing from the original list and are now included.
        _SENSITIVE = {
            "TOKEN", "KEY", "SECRET", "PASSWORD", "PASSWD",
            "CREDENTIAL", "SESSION", "AUTH", "COOKIE",
            "JWT", "BEARER", "SIGNATURE", "PIN", "PASSPHRASE",
            "PRIVATE",
        }
        def _mask(k, v):
            if k in _dynamic_keys:
                return "***"
            if any(s in k.upper() for s in _SENSITIVE):
                return "***"
            return v
        logged_cmd = []
        i = 0
        while i < len(cmd):
            if cmd[i] == "-e" and i + 1 < len(cmd) and "=" in cmd[i + 1]:
                k, _, v = cmd[i + 1].partition("=")
                logged_cmd.extend(["-e", f"{k}={_mask(k, v)}"])
                i += 2
            else:
                logged_cmd.append(cmd[i])
                i += 1
        logger.warning("docker exec cmd: %s", " ".join(logged_cmd))

        try:
            _output_chunks = []
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if effective_stdin else subprocess.DEVNULL,
                text=True,
            )
            if effective_stdin:
                try:
                    proc.stdin.write(effective_stdin)
                    proc.stdin.close()
                except Exception:
                    pass

            def _drain():
                try:
                    for line in proc.stdout:
                        _output_chunks.append(line)
                except Exception:
                    pass

            reader = threading.Thread(target=_drain, daemon=True)
            reader.start()
            deadline = time.monotonic() + effective_timeout

            while proc.poll() is None:
                if is_interrupted():
                    proc.terminate()
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    reader.join(timeout=2)
                    return {
                        "output": "".join(_output_chunks) + "\n[Command interrupted]",
                        "returncode": 130,
                    }
                if time.monotonic() > deadline:
                    proc.kill()
                    reader.join(timeout=2)
                    return self._timeout_result(effective_timeout)
                time.sleep(0.2)

            reader.join(timeout=5)
            return {"output": "".join(_output_chunks), "returncode": proc.returncode}
        except Exception as e:
            return {"output": f"Docker execution error: {e}", "returncode": 1}

    def cleanup(self):
        """Stop and remove the container. Bind-mount dirs persist if persistent=True."""
        if self._container_id:
            try:
                # Stop in background so cleanup doesn't block
                stop_cmd = (
                    f"(timeout 60 {self._docker_exe} stop {self._container_id} || "
                    f"{self._docker_exe} rm -f {self._container_id}) >/dev/null 2>&1 &"
                )
                subprocess.Popen(stop_cmd, shell=True)
            except Exception as e:
                logger.warning("Failed to stop container %s: %s", self._container_id, e)

            if not self._persistent:
                # Also schedule removal (stop only leaves it as stopped)
                try:
                    subprocess.Popen(
                        f"sleep 3 && {self._docker_exe} rm -f {self._container_id} >/dev/null 2>&1 &",
                        shell=True,
                    )
                except Exception:
                    pass
            self._container_id = None

        if not self._persistent:
            for d in (self._workspace_dir, self._home_dir):
                if d:
                    shutil.rmtree(d, ignore_errors=True)
