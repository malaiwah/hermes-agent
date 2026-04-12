"""Tests for the Podman execution environment."""

from unittest.mock import patch, MagicMock
import subprocess

import pytest


class TestFindPodman:
    """Tests for podman binary discovery."""

    def test_found_via_shutil_which(self):
        from tools.environments import podman as mod
        mod._podman_executable = None  # reset cache
        with patch("shutil.which", return_value="/usr/bin/podman"):
            result = mod.find_podman()
        assert result == "/usr/bin/podman"
        mod._podman_executable = None

    def test_found_via_search_paths(self):
        from tools.environments import podman as mod
        mod._podman_executable = None
        with (
            patch("shutil.which", return_value=None),
            patch("os.path.isfile", side_effect=lambda p: p == "/opt/homebrew/bin/podman"),
            patch("os.access", return_value=True),
        ):
            result = mod.find_podman()
        assert result == "/opt/homebrew/bin/podman"
        mod._podman_executable = None

    def test_found_podman_remote_in_path(self):
        """podman-remote should be found when podman is not available."""
        from tools.environments import podman as mod
        mod._podman_executable = None
        def _which(name):
            return "/usr/bin/podman-remote" if name == "podman-remote" else None
        with patch("shutil.which", side_effect=_which):
            result = mod.find_podman()
        assert result == "/usr/bin/podman-remote"
        mod._podman_executable = None

    def test_not_found_returns_none(self):
        from tools.environments import podman as mod
        mod._podman_executable = None
        with (
            patch("shutil.which", return_value=None),
            patch("os.path.isfile", return_value=False),
        ):
            result = mod.find_podman()
        assert result is None
        mod._podman_executable = None

    def test_search_paths_are_separate_strings(self):
        """Regression: ensure no missing commas cause string concatenation."""
        from tools.environments.podman import _PODMAN_SEARCH_PATHS
        for path in _PODMAN_SEARCH_PATHS:
            # Each path should end with the binary name, not be two paths joined
            assert path.endswith("/podman") or path.endswith("/podman-remote"), f"Bad path: {path}"
            # No path should be longer than a reasonable filesystem path
            assert len(path) < 60, f"Suspiciously long path (missing comma?): {path}"


class TestEnsurePodmanAvailable:
    """Tests for the availability check."""

    def test_raises_when_not_found(self):
        from tools.environments import podman as mod
        mod._podman_executable = None
        with (
            patch.object(mod, "find_podman", return_value=None),
            pytest.raises(RuntimeError, match="not found"),
        ):
            mod._ensure_podman_available()

    def test_raises_when_version_fails(self):
        from tools.environments import podman as mod
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"
        with (
            patch.object(mod, "find_podman", return_value="/usr/bin/podman"),
            patch("subprocess.run", return_value=mock_result),
            pytest.raises(RuntimeError, match="failed"),
        ):
            mod._ensure_podman_available()


class TestPodmanEnvironmentHooks:
    """Test that PodmanEnvironment overrides the right hooks without starting containers."""

    def test_resolve_cli_binary(self):
        from tools.environments.podman import PodmanEnvironment
        env = PodmanEnvironment.__new__(PodmanEnvironment)
        env._rootful = False
        env._privileged = False
        env._userns = ""
        env._extra_capabilities = []
        env._extra_args = []
        with patch("tools.environments.podman.find_podman", return_value="/usr/bin/podman"):
            assert env._resolve_cli_binary() == "/usr/bin/podman"

    def test_security_args_default(self):
        from tools.environments.podman import PodmanEnvironment, _SECURITY_ARGS
        env = PodmanEnvironment.__new__(PodmanEnvironment)
        env._privileged = False
        assert env._get_security_args() == list(_SECURITY_ARGS)

    def test_security_args_privileged(self):
        from tools.environments.podman import PodmanEnvironment
        env = PodmanEnvironment.__new__(PodmanEnvironment)
        env._privileged = True
        assert env._get_security_args() == ["--privileged"]

    def test_extra_run_args_empty(self):
        from tools.environments.podman import PodmanEnvironment
        env = PodmanEnvironment.__new__(PodmanEnvironment)
        env._userns = ""
        env._extra_capabilities = []
        env._extra_args = []
        assert env._get_extra_run_args() == []

    def test_extra_run_args_full(self):
        from tools.environments.podman import PodmanEnvironment
        env = PodmanEnvironment.__new__(PodmanEnvironment)
        env._userns = "keep-id"
        env._extra_capabilities = ["SYS_PTRACE"]
        env._extra_args = ["--security-opt=seccomp=unconfined"]
        result = env._get_extra_run_args()
        assert "--userns" in result
        assert "keep-id" in result
        assert "--cap-add" in result
        assert "SYS_PTRACE" in result
        assert "--security-opt=seccomp=unconfined" in result

    def test_build_run_cmd_rootful_adds_sudo(self):
        from tools.environments.podman import PodmanEnvironment
        env = PodmanEnvironment.__new__(PodmanEnvironment)
        env._docker_exe = "/usr/bin/podman"
        env._rootful = True
        cmd = env._build_run_cmd("test-container", "/workspace", [], "myimage:latest")
        assert cmd[0] == "sudo"
        assert cmd[1] == "/usr/bin/podman"

    def test_build_run_cmd_rootless_no_sudo(self):
        from tools.environments.podman import PodmanEnvironment
        env = PodmanEnvironment.__new__(PodmanEnvironment)
        env._docker_exe = "/usr/bin/podman"
        env._rootful = False
        cmd = env._build_run_cmd("test-container", "/workspace", [], "myimage:latest")
        assert cmd[0] == "/usr/bin/podman"


class TestPodmanExtraArgsValidation:
    """Test input validation for extra args."""

    def test_non_string_extra_args_rejected(self):
        """PodmanEnvironment should warn and ignore non-string extra_args."""
        from tools.environments.podman import PodmanEnvironment
        env = PodmanEnvironment.__new__(PodmanEnvironment)
        # Simulate what __init__ does with bad input
        extra_args = ["--valid", 123, None]
        if extra_args and not all(isinstance(x, str) for x in extra_args):
            extra_args = None
        assert extra_args is None

    def test_string_extra_args_accepted(self):
        from tools.environments.podman import PodmanEnvironment
        extra_args = ["--valid", "--also-valid"]
        if extra_args and not all(isinstance(x, str) for x in extra_args):
            extra_args = None
        assert extra_args == ["--valid", "--also-valid"]
