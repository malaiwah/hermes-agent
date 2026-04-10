import logging
from io import StringIO
import subprocess
import sys
import types

import pytest

from tools.environments import docker as docker_env


def _mock_subprocess_run(monkeypatch):
    """Mock subprocess.run to intercept docker run -d and docker version calls.

    Returns a list of captured (cmd, kwargs) tuples for inspection.
    """
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def _make_dummy_env(**kwargs):
    """Helper to construct DockerEnvironment with minimal required args."""
    return docker_env.DockerEnvironment(
        image=kwargs.get("image", "python:3.11"),
        cwd=kwargs.get("cwd", "/root"),
        timeout=kwargs.get("timeout", 60),
        cpu=kwargs.get("cpu", 0),
        memory=kwargs.get("memory", 0),
        disk=kwargs.get("disk", 0),
        persistent_filesystem=kwargs.get("persistent_filesystem", False),
        task_id=kwargs.get("task_id", "test-task"),
        volumes=kwargs.get("volumes", []),
        network=kwargs.get("network", True),
        host_cwd=kwargs.get("host_cwd"),
        auto_mount_cwd=kwargs.get("auto_mount_cwd", False),
        env=kwargs.get("env"),
    )


def test_ensure_docker_available_logs_and_raises_when_not_found(monkeypatch, caplog):
    """When docker cannot be found, raise a clear error before container setup."""

    monkeypatch.setattr(docker_env, "find_docker", lambda: None)
    monkeypatch.setattr(
        docker_env.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess.run should not be called when docker is missing"),
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            _make_dummy_env()

    assert "Docker executable not found in PATH or known install locations" in str(excinfo.value)
    assert any(
        "no docker executable was found in PATH or known install locations"
        in record.getMessage()
        for record in caplog.records
    )


def test_ensure_docker_available_logs_and_raises_on_timeout(monkeypatch, caplog):
    """When docker version times out, surface a helpful error instead of hanging."""

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["/custom/docker", "version"], timeout=5)

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/custom/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _raise_timeout)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            _make_dummy_env()

    assert "Docker daemon is not responding" in str(excinfo.value)
    assert any(
        "/custom/docker version' timed out" in record.getMessage()
        for record in caplog.records
    )


def test_ensure_docker_available_uses_resolved_executable(monkeypatch):
    """When docker is found outside PATH, preflight should use that resolved path."""

    calls = []

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/opt/homebrew/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    docker_env._ensure_docker_available()

    assert calls == [
        (["/opt/homebrew/bin/docker", "version"], {
            "capture_output": True,
            "text": True,
            "timeout": 5,
        })
    ]


def test_auto_mount_host_cwd_adds_volume(monkeypatch, tmp_path):
    """Opt-in docker cwd mounting should bind the host cwd to /workspace."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
    )

    # Find the docker run call and check its args
    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{project_dir}:/workspace" in run_args_str


def test_auto_mount_disabled_by_default(monkeypatch, tmp_path):
    """Host cwd should not be mounted unless the caller explicitly opts in."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/root",
        host_cwd=str(project_dir),
        auto_mount_cwd=False,
    )

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{project_dir}:/workspace" not in run_args_str


def test_auto_mount_skipped_when_workspace_already_mounted(monkeypatch, tmp_path):
    """Explicit user volumes for /workspace should take precedence over cwd mount."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        volumes=[f"{other_dir}:/workspace"],
    )

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{other_dir}:/workspace" in run_args_str
    assert run_args_str.count(":/workspace") == 1


def test_explicit_read_only_workspace_mount_is_preserved(monkeypatch, tmp_path):
    """An explicit read-only /workspace bind should be passed through unchanged."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        volumes=[f"{project_dir}:/workspace:ro"],
    )

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{project_dir}:/workspace:ro" in run_args_str
    assert run_args_str.count(":/workspace") == 1


def test_auto_mount_replaces_persistent_workspace_bind(monkeypatch, tmp_path):
    """Persistent mode should still prefer the configured host cwd at /workspace."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        persistent_filesystem=True,
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        task_id="test-persistent-auto-mount",
    )

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{project_dir}:/workspace" in run_args_str
    assert "/sandboxes/docker/test-persistent-auto-mount/workspace:/workspace" not in run_args_str


def test_non_persistent_cleanup_removes_container(monkeypatch):
    """When persistent=false, cleanup() must schedule docker stop + rm."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    popen_cmds = []
    monkeypatch.setattr(
        docker_env.subprocess, "Popen",
        lambda cmd, **kw: (popen_cmds.append(cmd), type("P", (), {"poll": lambda s: 0, "wait": lambda s, **k: None, "returncode": 0, "stdout": iter([]), "stdin": None})())[1],
    )

    env = _make_dummy_env(persistent_filesystem=False, task_id="ephemeral-task")
    assert env._container_id
    container_id = env._container_id

    env.cleanup()

    # Should have stop and rm calls via Popen
    stop_cmds = [c for c in popen_cmds if container_id in str(c) and "stop" in str(c)]
    assert len(stop_cmds) >= 1, f"cleanup() should schedule docker stop for {container_id}"


class _FakePopen:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdout = StringIO("")
        self.stdin = None
        self.returncode = 0

    def poll(self):
        return self.returncode


def _make_execute_only_env(forward_env=None):
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env.cwd = "/root"
    env.timeout = 60
    env._forward_env = forward_env or []
    env._env = {}
    env._prepare_command = lambda command: (command, None)
    env._timeout_result = lambda timeout: {"output": f"timed out after {timeout}", "returncode": 124}
    env._container_id = "test-container"
    env._docker_exe = "/usr/bin/docker"
    # Base class attributes needed by unified execute()
    env._session_id = "test123"
    env._snapshot_path = "/tmp/hermes-snap-test123.sh"
    env._cwd_file = "/tmp/hermes-cwd-test123.txt"
    env._cwd_marker = "__HERMES_CWD_test123__"
    env._snapshot_ready = True
    env._last_sync_time = None
    env._init_env_args = []
    return env


def test_init_env_args_uses_hermes_dotenv_for_allowlisted_env(monkeypatch):
    """_build_init_env_args picks up forwarded env vars from .env file at init time."""
    # Use a var that is NOT in _HERMES_PROVIDER_ENV_BLOCKLIST (GITHUB_TOKEN
    # is in the copilot provider's api_key_env_vars and gets stripped).
    env = _make_execute_only_env(["DATABASE_URL"])

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {"DATABASE_URL": "value_from_dotenv"})

    args = env._build_init_env_args()
    args_str = " ".join(args)

    assert "DATABASE_URL=value_from_dotenv" in args_str


def test_init_env_args_prefers_shell_env_over_hermes_dotenv(monkeypatch):
    """Shell env vars take priority over .env file values in init env args."""
    env = _make_execute_only_env(["DATABASE_URL"])

    monkeypatch.setenv("DATABASE_URL", "value_from_shell")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {"DATABASE_URL": "value_from_dotenv"})

    args = env._build_init_env_args()
    args_str = " ".join(args)

    assert "DATABASE_URL=value_from_shell" in args_str
    assert "value_from_dotenv" not in args_str


# ── docker_env tests ──────────────────────────────────────────────


def test_docker_env_appears_in_run_command(monkeypatch):
    """Explicit docker_env values should be passed via -e at docker run time."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(env={"SSH_AUTH_SOCK": "/run/user/1000/ssh-agent.sock", "GNUPGHOME": "/root/.gnupg"})

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args = run_calls[0][0]
    run_args_str = " ".join(run_args)
    assert "SSH_AUTH_SOCK=/run/user/1000/ssh-agent.sock" in run_args_str
    assert "GNUPGHOME=/root/.gnupg" in run_args_str


def test_docker_env_appears_in_init_env_args(monkeypatch):
    """Explicit docker_env values should appear in _build_init_env_args."""
    env = _make_execute_only_env()
    env._env = {"MY_VAR": "my_value"}

    args = env._build_init_env_args()
    args_str = " ".join(args)

    assert "MY_VAR=my_value" in args_str


def test_forward_env_overrides_docker_env_in_init_args(monkeypatch):
    """docker_forward_env should override docker_env for the same key."""
    env = _make_execute_only_env(forward_env=["MY_KEY"])
    env._env = {"MY_KEY": "static_value"}

    monkeypatch.setenv("MY_KEY", "dynamic_value")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})

    args = env._build_init_env_args()
    args_str = " ".join(args)

    assert "MY_KEY=dynamic_value" in args_str
    assert "MY_KEY=static_value" not in args_str


def test_docker_env_and_forward_env_merge_in_init_args(monkeypatch):
    """docker_env and docker_forward_env with different keys should both appear."""
    env = _make_execute_only_env(forward_env=["TOKEN"])
    env._env = {"SSH_AUTH_SOCK": "/run/user/1000/agent.sock"}

    monkeypatch.setenv("TOKEN", "secret123")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})

    args = env._build_init_env_args()
    args_str = " ".join(args)

    assert "SSH_AUTH_SOCK=/run/user/1000/agent.sock" in args_str
    assert "TOKEN=secret123" in args_str



def test_normalize_env_dict_filters_invalid_keys():
    """_normalize_env_dict should reject invalid variable names."""
    result = docker_env._normalize_env_dict({
        "VALID_KEY": "ok",
        "123bad": "rejected",
        "": "rejected",
        "also valid": "rejected",  # spaces invalid
        "GOOD": "ok",
    })
    assert result == {"VALID_KEY": "ok", "GOOD": "ok"}


def test_normalize_env_dict_coerces_scalars():
    """_normalize_env_dict should coerce int/float/bool to str."""
    result = docker_env._normalize_env_dict({
        "PORT": 8080,
        "DEBUG": True,
        "RATIO": 0.5,
    })
    assert result == {"PORT": "8080", "DEBUG": "True", "RATIO": "0.5"}


def test_normalize_env_dict_rejects_non_dict():
    """_normalize_env_dict should return empty dict for non-dict input."""
    assert docker_env._normalize_env_dict("not a dict") == {}
    assert docker_env._normalize_env_dict(None) == {}
    assert docker_env._normalize_env_dict([]) == {}


def test_normalize_env_dict_rejects_complex_values():
    """_normalize_env_dict should reject list/dict values."""
    result = docker_env._normalize_env_dict({
        "GOOD": "string",
        "BAD_LIST": [1, 2, 3],
        "BAD_DICT": {"nested": True},
    })
    assert result == {"GOOD": "string"}


# ---------------------------------------------------------------------------
# docker_env_files: per-exec re-read with allowlist + size cap
# ---------------------------------------------------------------------------

def _allow_anywhere(monkeypatch):
    """Disable the path allowlist for tests by emptying it via env var."""
    monkeypatch.setenv("TERMINAL_DOCKER_ENV_FILES_ALLOWED_DIRS", "")


def test_parse_env_files_valid_entry(monkeypatch, tmp_path):
    _allow_anywhere(monkeypatch)
    f = tmp_path / "session"
    f.write_text("hello")
    parsed = docker_env.DockerEnvironment._parse_env_files(
        [f"BW_SESSION:{f}"]
    )
    assert len(parsed) == 1
    assert parsed[0][0] == "BW_SESSION"
    assert parsed[0][1] == str(f.resolve())


def test_parse_env_files_invalid_format_skipped(monkeypatch, caplog):
    _allow_anywhere(monkeypatch)
    with caplog.at_level(logging.WARNING):
        parsed = docker_env.DockerEnvironment._parse_env_files(["NO_COLON"])
    assert parsed == []
    assert any("invalid entry" in r.getMessage() for r in caplog.records)


def test_parse_env_files_empty_var_name_skipped(monkeypatch, caplog, tmp_path):
    _allow_anywhere(monkeypatch)
    with caplog.at_level(logging.WARNING):
        parsed = docker_env.DockerEnvironment._parse_env_files([f":{tmp_path / 'x'}"])
    assert parsed == []


def test_parse_env_files_resolves_symlink(monkeypatch, tmp_path):
    """A symlink at parse time is followed; subsequent symlink swaps don't redirect reads."""
    _allow_anywhere(monkeypatch)
    target = tmp_path / "real"
    target.write_text("real-value")
    link = tmp_path / "link"
    link.symlink_to(target)
    parsed = docker_env.DockerEnvironment._parse_env_files([f"X:{link}"])
    assert len(parsed) == 1
    assert parsed[0][1] == str(target.resolve())  # canonicalized to real path


def test_parse_env_files_path_does_not_have_to_exist(monkeypatch, tmp_path):
    """Sidecar may not have written the file yet at parse time — must not error."""
    _allow_anywhere(monkeypatch)
    missing = tmp_path / "not-yet-written"
    parsed = docker_env.DockerEnvironment._parse_env_files([f"X:{missing}"])
    assert len(parsed) == 1


def test_parse_env_files_allowlist_rejects_outside_paths(monkeypatch, tmp_path, caplog):
    """With an allowlist set, paths outside it are rejected."""
    safe_dir = tmp_path / "allowed"
    safe_dir.mkdir()
    bad = tmp_path / "elsewhere" / "secret"
    bad.parent.mkdir()
    bad.write_text("nope")
    monkeypatch.setenv("TERMINAL_DOCKER_ENV_FILES_ALLOWED_DIRS", str(safe_dir))
    with caplog.at_level(logging.WARNING):
        parsed = docker_env.DockerEnvironment._parse_env_files([f"X:{bad}"])
    assert parsed == []
    assert any("outside allowed dirs" in r.getMessage() for r in caplog.records)


def test_parse_env_files_allowlist_accepts_inside_paths(monkeypatch, tmp_path):
    """With an allowlist set, paths inside it are accepted."""
    safe_dir = tmp_path / "allowed"
    safe_dir.mkdir()
    good = safe_dir / "session"
    good.write_text("ok")
    monkeypatch.setenv("TERMINAL_DOCKER_ENV_FILES_ALLOWED_DIRS", str(safe_dir))
    parsed = docker_env.DockerEnvironment._parse_env_files([f"X:{good}"])
    assert len(parsed) == 1


def test_read_env_file_value_strips_one_trailing_newline(monkeypatch, tmp_path):
    f = tmp_path / "session"
    f.write_text("abc\n")  # `echo abc > file` shape
    assert docker_env.DockerEnvironment._read_env_file_value("X", str(f)) == "abc"


def test_read_env_file_value_strips_crlf(monkeypatch, tmp_path):
    f = tmp_path / "session"
    f.write_bytes(b"abc\r\n")
    assert docker_env.DockerEnvironment._read_env_file_value("X", str(f)) == "abc"


def test_read_env_file_value_preserves_internal_whitespace(monkeypatch, tmp_path):
    """`.strip()` would corrupt PEM bodies; we only trim one trailing newline."""
    pem = "-----BEGIN PRIVATE KEY-----\n  base64body\n-----END PRIVATE KEY-----\n"
    f = tmp_path / "key"
    f.write_text(pem)
    got = docker_env.DockerEnvironment._read_env_file_value("KEY", str(f))
    # Trailing \n stripped, internal whitespace preserved exactly
    assert got == pem[:-1]
    assert "  base64body" in got
    assert got.endswith("-----END PRIVATE KEY-----")


def test_read_env_file_value_preserves_leading_whitespace(monkeypatch, tmp_path):
    """JSON blobs with leading spaces must round-trip unchanged."""
    f = tmp_path / "json"
    f.write_text("  {\"key\": \"value\"}")  # no trailing newline
    got = docker_env.DockerEnvironment._read_env_file_value("J", str(f))
    assert got == "  {\"key\": \"value\"}"


def test_read_env_file_value_size_limit(monkeypatch, tmp_path, caplog):
    """Files larger than _ENV_FILES_MAX_SIZE are rejected with a clear log line."""
    f = tmp_path / "huge"
    f.write_bytes(b"A" * (docker_env.DockerEnvironment._ENV_FILES_MAX_SIZE + 100))
    with caplog.at_level(logging.WARNING):
        got = docker_env.DockerEnvironment._read_env_file_value("X", str(f))
    assert got is None
    assert any("exceeds" in r.getMessage() and "limit" in r.getMessage() for r in caplog.records)


def test_read_env_file_value_size_limit_at_boundary(monkeypatch, tmp_path):
    """A file at exactly the size limit is accepted."""
    payload = b"A" * docker_env.DockerEnvironment._ENV_FILES_MAX_SIZE
    f = tmp_path / "boundary"
    f.write_bytes(payload)
    got = docker_env.DockerEnvironment._read_env_file_value("X", str(f))
    assert got == payload.decode()


def test_read_env_file_value_missing_file_returns_none(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        got = docker_env.DockerEnvironment._read_env_file_value("X", str(tmp_path / "nope"))
    assert got is None
    assert any("could not read" in r.getMessage() for r in caplog.records)


def test_read_env_file_value_non_utf8_returns_none(monkeypatch, tmp_path, caplog):
    f = tmp_path / "binary"
    f.write_bytes(b"\xff\xfe\xfd")
    with caplog.at_level(logging.WARNING):
        got = docker_env.DockerEnvironment._read_env_file_value("X", str(f))
    assert got is None
    assert any("not valid UTF-8" in r.getMessage() for r in caplog.records)


def test_extra_env_for_exec_re_reads_file(monkeypatch, tmp_path):
    """The exec hook re-reads the file each call so rotated values propagate."""
    _allow_anywhere(monkeypatch)
    f = tmp_path / "session"
    f.write_text("session-A")
    _mock_subprocess_run(monkeypatch)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    env = _make_dummy_env()
    # Inject the parsed entry directly (don't go through __init__'s env_files arg
    # to avoid coupling this test to constructor wiring).
    env._env_files = [("BW_SESSION", str(f.resolve()))]

    first = env._extra_env_for_exec()
    assert first == {"BW_SESSION": "session-A"}

    f.write_text("session-B")  # rotation
    second = env._extra_env_for_exec()
    assert second == {"BW_SESSION": "session-B"}


def test_extra_env_for_exec_skips_failed_entries(monkeypatch, tmp_path, caplog):
    """A bad entry is skipped; good entries still get applied."""
    _allow_anywhere(monkeypatch)
    good = tmp_path / "good"
    good.write_text("ok")
    _mock_subprocess_run(monkeypatch)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    env = _make_dummy_env()
    env._env_files = [
        ("GOOD", str(good.resolve())),
        ("MISSING", str(tmp_path / "nope")),
    ]
    with caplog.at_level(logging.WARNING):
        out = env._extra_env_for_exec()
    assert out == {"GOOD": "ok"}
    assert any("MISSING" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# docker exec cmd masking — both name-heuristic and origin-based
# ---------------------------------------------------------------------------

def _capture_exec_log(monkeypatch, env, env_overrides=None):
    """Run env.execute() with mocked Popen and return the captured log line."""
    popen_calls = []

    class _FakePopen2:
        def __init__(self, cmd, **kwargs):
            popen_calls.append(cmd)
            self.stdin = None
            self.stdout = StringIO("")
            self.stderr = None
            self.returncode = 0
        def wait(self, timeout=None):
            return 0
        def communicate(self, *a, **kw):
            return ("", "")
        def poll(self):
            return 0

    monkeypatch.setattr(docker_env.subprocess, "Popen", _FakePopen2)
    captured_logs = []
    real_warning = docker_env.logger.warning
    def _capture_warning(msg, *args, **kwargs):
        if args:
            try:
                captured_logs.append(msg % args)
            except Exception:
                captured_logs.append(str(msg))
        else:
            captured_logs.append(str(msg))
        real_warning(msg, *args, **kwargs)
    monkeypatch.setattr(docker_env.logger, "warning", _capture_warning)

    if env_overrides:
        for k, v in env_overrides.items():
            monkeypatch.setenv(k, v)

    env.execute("echo hi")
    exec_lines = [l for l in captured_logs if "docker exec cmd:" in l]
    return exec_lines[0] if exec_lines else ""


def test_exec_log_masks_session_in_name(monkeypatch):
    """SESSION-named env vars are masked even though the original heuristic missed them."""
    _allow_anywhere(monkeypatch)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    env = _make_execute_only_env()
    env._env = {"BW_SESSION": "VERYSECRETSESSIONVALUE12345"}
    log = _capture_exec_log(monkeypatch, env)
    assert "VERYSECRETSESSIONVALUE12345" not in log, f"session leaked in log: {log}"
    assert "BW_SESSION=***" in log


def test_exec_log_masks_auth_cookie_jwt_bearer(monkeypatch):
    """The expanded sensitive-name list catches more credential-like names."""
    _allow_anywhere(monkeypatch)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    env = _make_execute_only_env()
    env._env = {
        "AUTH_COOKIE": "auth-cookie-value-zzz",
        "MY_JWT": "jwt.value.zzz",
        "X_BEARER": "bearer-zzz",
        "PASSPHRASE": "passphrase-zzz",
    }
    log = _capture_exec_log(monkeypatch, env)
    for v in ("auth-cookie-value-zzz", "jwt.value.zzz", "bearer-zzz", "passphrase-zzz"):
        assert v not in log, f"value {v} leaked in log: {log}"


def test_exec_log_masks_dynamic_origin_regardless_of_name(monkeypatch, tmp_path):
    """Anything from _extra_env_for_exec is masked, even with an innocuous name."""
    _allow_anywhere(monkeypatch)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    f = tmp_path / "innocuous"
    f.write_text("DYNAMICALLY_INJECTED_VALUE")
    env = _make_execute_only_env()
    env._env_files = [("INNOCENT_VAR", str(f.resolve()))]  # NOT a sensitive-looking name
    log = _capture_exec_log(monkeypatch, env)
    assert "DYNAMICALLY_INJECTED_VALUE" not in log, f"dynamic value leaked: {log}"
    assert "INNOCENT_VAR=***" in log


def test_exec_log_does_not_mask_innocent_static_values(monkeypatch):
    """Plain static env (not credential-like) is not over-masked — readability check."""
    _allow_anywhere(monkeypatch)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    env = _make_execute_only_env()
    env._env = {"PORT": "8080", "DEBUG": "true", "HOME": "/root"}
    log = _capture_exec_log(monkeypatch, env)
    assert "PORT=8080" in log
    assert "DEBUG=true" in log
    assert "HOME=/root" in log
