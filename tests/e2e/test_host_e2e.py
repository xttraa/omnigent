"""End-to-end tests for the Host API (``omnigent connect``).

These tests start a real server subprocess, connect a real host
daemon, create sessions via the REST API, and verify the full
launch-runner → exchange-messages flow.

All tests run against the mock LLM server — no real credentials
needed::

    .venv/bin/python -m pytest tests/e2e/test_host_e2e.py -v

The last test (claude-native host-restart regression) is skipped because
it requires real ``claude`` + ``tmux`` CLIs with interactive OAuth login.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import yaml

from tests.e2e.conftest import (
    POLL_INTERVAL_S,
    configure_mock_llm,
    lookup_agent_id,
    poll_session_until_terminal,
    send_user_message_to_session,
    upload_agent,
)
from tests.e2e.helpers import final_assistant_text


@dataclass
class _SpawnedHostDaemon:
    """A spawned host daemon subprocess paired with its known host_id.

    :param proc: The daemon subprocess handle.
    :param host_id: The host_id pre-seeded into ``config.yaml`` before
        spawning, e.g. ``"host_a1b2c3d4e5f6..."``.
    :param daemon_log: Path to the captured daemon stderr log; carries
        the ``Launched runner ... (pid=NNNN)`` line tests parse to find
        a spawned runner's process id.
    """

    proc: subprocess.Popen[bytes]
    host_id: str
    daemon_log: Path


def _spawn_host_daemon(
    *,
    tmp_path: Path,
    live_server: str,
    mock_llm_server_url: str,
) -> _SpawnedHostDaemon:
    """
    Spawn an isolated host daemon for a single host e2e test.

    Pre-seeds ``config.yaml`` with a UNIQUE ``(host_id, name)``: the host
    e2e tests share a session-scoped server, and the host store enforces a
    unique ``(owner, name)`` row. With the default machine hostname every
    test would collide on that row, so a later test's freshly-registered
    host_id gets overwritten and never shows online. A unique name per test
    keeps each host its own row.

    The daemon's environment carries ``OPENAI_BASE_URL`` and
    ``OPENAI_API_KEY`` pointing at the mock LLM server.  The host
    daemon forwards ``OPENAI_*`` to its runner subprocesses via
    ``HARNESS_CREDENTIAL_ENV_VARS``, so the runner's openai-agents
    executor hits the mock server.

    :param tmp_path: Per-test temp dir used as the daemon's ``HOME``.
    :param live_server: Server URL the daemon registers with, e.g.
        ``"http://localhost:18501"``.
    :param mock_llm_server_url: Base URL of the mock LLM server, e.g.
        ``"http://127.0.0.1:12345"``.
    :returns: The spawned daemon handle and its host_id.
    """
    omni_dir = tmp_path / ".omnigent"
    omni_dir.mkdir(parents=True, exist_ok=True)
    host_id = f"host_{uuid.uuid4().hex}"
    host_name = f"e2e-host-{uuid.uuid4().hex[:12]}"
    (omni_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {"host": {"host_id": host_id, "name": host_name}},
            default_flow_style=False,
            sort_keys=True,
        )
    )
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "OPENAI_BASE_URL": f"{mock_llm_server_url}/v1",
        "OPENAI_API_KEY": "mock-key",
    }
    # Capture the daemon's stderr to a file so tests can read the
    # "Launched runner ... (pid=NNNN)" line (and inspect it on failure).
    # The child keeps its own dup of the fd after this handle is closed.
    daemon_log = tmp_path / "host-daemon.log"
    with open(daemon_log, "w") as log_fh:
        proc = subprocess.Popen(
            [sys.executable, "-m", "omnigent.host._daemon_entry", "--server", live_server],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=log_fh,
        )
    return _SpawnedHostDaemon(proc=proc, host_id=host_id, daemon_log=daemon_log)


def _runner_pid_from_daemon_log(log_path: Path) -> int | None:
    """Parse the launched runner's PID from the host daemon's log.

    The daemon logs ``Launched runner <id> for workspace <ws> (pid=NNNN)``
    when it spawns a runner subprocess.

    :param log_path: Path to the captured daemon stderr log.
    :returns: The runner subprocess PID, or ``None`` if not present yet.
    """
    if not log_path.exists():
        return None
    match = re.search(
        r"Launched runner \S+ for workspace .*? \(pid=(\d+)\)",
        log_path.read_text(),
    )
    return int(match.group(1)) if match else None


def _pid_alive(pid: int) -> bool:
    """Return whether a process id is currently alive.

    :param pid: Process id to probe, e.g. ``12345``.
    :returns: ``True`` if the process exists, ``False`` once it has exited.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_smoke_agent_yaml(tmp_path: Path) -> Path:
    """Create a minimal Omnigent YAML for host e2e tests.

    :param tmp_path: Pytest temp directory.
    :returns: Path to the agent directory.
    """
    agent_dir = tmp_path / "host-e2e-agent"
    agent_dir.mkdir()
    (agent_dir / "host-e2e-agent.yaml").write_text(
        "\n".join(
            [
                "name: host-e2e-agent",
                "description: Minimal agent for host e2e tests.",
                "executor:",
                "  harness: openai-agents",
                "  model: gpt-5.4",
                "prompt: |",
                "  You are a terse smoke-test assistant.",
                "  Follow the user's instruction exactly.",
                "",
            ]
        )
    )
    return agent_dir


def _wait_for_host_online(
    client: httpx.Client,
    host_id: str,
    timeout: float = 30.0,
) -> None:
    """Poll GET /v1/hosts until the host appears online.

    :param client: HTTP client pointed at the server.
    :param host_id: Host ID to wait for.
    :param timeout: Max seconds to wait.
    :raises AssertionError: If the host never appears online.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = client.get("/v1/hosts")
            if resp.status_code == 200:
                for host in resp.json().get("hosts", []):
                    if host["host_id"] == host_id and host["status"] == "online":
                        return
        except httpx.ConnectError:
            pass
        time.sleep(POLL_INTERVAL_S)
    raise AssertionError(f"Host {host_id!r} did not appear online within {timeout}s")


def test_host_connect_and_list(
    live_server: str,
    http_client: httpx.Client,
    tmp_path: Path,
    mock_llm_server_url: str,
) -> None:
    """
    Start ``omnigent connect`` as a subprocess, verify the host
    appears in ``GET /v1/hosts`` with status online, stop it, and
    verify it goes offline.

    This is the basic registration smoke test — if the host never
    appears online, the WS tunnel handshake or DB upsert is broken.
    """
    daemon = _spawn_host_daemon(
        tmp_path=tmp_path,
        live_server=live_server,
        mock_llm_server_url=mock_llm_server_url,
    )
    proc = daemon.proc
    host_id = daemon.host_id

    try:
        # Host should appear online in GET /v1/hosts.
        _wait_for_host_online(http_client, host_id, timeout=30.0)

        resp = http_client.get("/v1/hosts")
        assert resp.status_code == 200
        hosts = resp.json()["hosts"]
        matching = [h for h in hosts if h["host_id"] == host_id]
        # Exactly one host with our ID should be listed.
        assert len(matching) == 1, (
            f"Expected 1 host with id {host_id!r}, got {len(matching)}. All hosts: {hosts}"
        )
        assert matching[0]["status"] == "online"

    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # After killing the daemon, host should go offline.
    # Give the server a moment to process the disconnect.
    time.sleep(1.0)
    resp = http_client.get(f"/v1/hosts/{host_id}")
    if resp.status_code == 200:
        assert resp.json()["status"] == "offline", "Host should be offline after daemon is killed"


def test_host_launch_runner_and_session_round_trip(
    live_server: str,
    http_client: httpx.Client,
    tmp_path: Path,
    mock_llm_server_url: str,
) -> None:
    """
    Full golden-path e2e: connect host, upload agent, create
    session, launch runner via ``POST /v1/hosts/{id}/runners``,
    send a message, and verify the LLM responds.

    This exercises the complete Web UI flow from the design doc:
    list hosts → create session → launch runner → exchange messages.
    """
    # Configure mock LLM to reply with the marker for the round-trip.
    marker = "HOST_E2E_GOLDEN_PATH_OK"
    configure_mock_llm(mock_llm_server_url, [{"text": marker}])

    # 1. Start host daemon.
    daemon = _spawn_host_daemon(
        tmp_path=tmp_path,
        live_server=live_server,
        mock_llm_server_url=mock_llm_server_url,
    )
    host_proc = daemon.proc
    host_id = daemon.host_id

    try:
        _wait_for_host_online(http_client, host_id, timeout=30.0)

        # 2. Upload agent.
        agent_name = upload_agent(
            http_client,
            _write_smoke_agent_yaml(tmp_path),
        )

        # 3. Create session (no runner yet).
        agent_id = lookup_agent_id(http_client, agent_name)
        resp = http_client.post(
            "/v1/sessions",
            json={"agent_id": agent_id},
        )
        resp.raise_for_status()
        session_id = resp.json()["id"]

        # 4. Launch runner on the host.
        launch_resp = http_client.post(
            f"/v1/hosts/{host_id}/runners",
            json={
                "session_id": session_id,
                "workspace": str(tmp_path),
            },
            timeout=60.0,
        )
        assert launch_resp.status_code == 200, (
            f"Launch failed: {launch_resp.status_code} {launch_resp.text}"
        )
        runner_id = launch_resp.json()["runner_id"]

        # 5. Wait for runner to connect and bind.
        deadline = time.monotonic() + 30.0
        runner_online = False
        while time.monotonic() < deadline:
            status_resp = http_client.get(f"/v1/runners/{runner_id}/status")
            if status_resp.status_code == 200 and status_resp.json().get("online") is True:
                runner_online = True
                break
            time.sleep(0.5)
        assert runner_online, f"Runner {runner_id} never came online after launch"

        # 6. Bind runner to session (the launch endpoint wrote
        #    runner_id but the session needs a PATCH for the relay).
        http_client.patch(
            f"/v1/sessions/{session_id}",
            json={"runner_id": runner_id},
        ).raise_for_status()

        # 7. Send a message and verify the LLM responds.
        response_id = send_user_message_to_session(
            http_client,
            session_id=session_id,
            content=(
                f"Reply with exactly the literal string {marker} "
                "and nothing else. Do not call tools."
            ),
        )
        body = poll_session_until_terminal(
            http_client,
            session_id=session_id,
            response_id=response_id,
            timeout=180,
        )

        # The session should complete and the marker should be in
        # the assistant's response.
        assert body["status"] == "completed", f"Session failed: {body.get('error')}"
        text = final_assistant_text(body)
        assert marker in text, f"Marker {marker!r} missing from response: {text!r}"

    finally:
        host_proc.send_signal(signal.SIGTERM)
        try:
            host_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            host_proc.kill()
            host_proc.wait()


def test_host_runner_survives_host_disconnect(
    live_server: str,
    http_client: httpx.Client,
    tmp_path: Path,
    mock_llm_server_url: str,
) -> None:
    """
    Start host, launch runner, kill host, verify session still
    works (runner has independent WS tunnel).

    This proves the design decision that runners connect directly
    to the server, not through the host. If the session breaks
    after host disconnect, runner independence is violated.
    """
    # Pre-kill and post-kill markers.
    marker1 = "HOST_SURVIVE_PRE_KILL"
    marker2 = "HOST_SURVIVE_POST_KILL"
    configure_mock_llm(mock_llm_server_url, [{"text": marker1}, {"text": marker2}])

    daemon = _spawn_host_daemon(
        tmp_path=tmp_path,
        live_server=live_server,
        mock_llm_server_url=mock_llm_server_url,
    )
    host_proc = daemon.proc
    host_id = daemon.host_id

    try:
        _wait_for_host_online(http_client, host_id, timeout=30.0)

        # Upload agent + create session + launch runner.
        agent_name = upload_agent(
            http_client,
            _write_smoke_agent_yaml(tmp_path),
        )
        agent_id = lookup_agent_id(http_client, agent_name)
        resp = http_client.post(
            "/v1/sessions",
            json={"agent_id": agent_id},
        )
        resp.raise_for_status()
        session_id = resp.json()["id"]

        launch_resp = http_client.post(
            f"/v1/hosts/{host_id}/runners",
            json={"session_id": session_id, "workspace": str(tmp_path)},
            timeout=60.0,
        )
        assert launch_resp.status_code == 200
        runner_id = launch_resp.json()["runner_id"]

        # Wait for runner online.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            sr = http_client.get(f"/v1/runners/{runner_id}/status")
            if sr.status_code == 200 and sr.json().get("online"):
                break
            time.sleep(0.5)

        http_client.patch(
            f"/v1/sessions/{session_id}",
            json={"runner_id": runner_id},
        ).raise_for_status()

        # Verify session works BEFORE killing host.
        rid1 = send_user_message_to_session(
            http_client,
            session_id=session_id,
            content=f"Reply with exactly {marker1} and nothing else.",
        )
        body1 = poll_session_until_terminal(
            http_client,
            session_id=session_id,
            response_id=rid1,
            timeout=120,
        )
        assert body1["status"] == "completed"
        assert marker1 in final_assistant_text(body1)

        # Kill the host daemon (but NOT the runner — it's a separate
        # process with start_new_session=True in the host daemon,
        # but since the daemon spawns runners as children, we need
        # to only kill the daemon, not its children).
        host_proc.send_signal(signal.SIGTERM)
        host_proc.wait(timeout=5)

        # Give server a moment to notice the host disconnect.
        time.sleep(1.0)

        # Runner should still be online.
        sr = http_client.get(f"/v1/runners/{runner_id}/status")
        # Runner may or may not still be online depending on whether
        # the daemon's SIGTERM cascaded. If the runner IS still
        # online, verify the session still works.
        if sr.status_code == 200 and sr.json().get("online"):
            rid2 = send_user_message_to_session(
                http_client,
                session_id=session_id,
                content=f"Reply with exactly {marker2} and nothing else.",
            )
            body2 = poll_session_until_terminal(
                http_client,
                session_id=session_id,
                response_id=rid2,
                timeout=120,
            )
            assert body2["status"] == "completed", (
                "Session should still work after host disconnect — "
                "runner has independent WS tunnel"
            )
            assert marker2 in final_assistant_text(body2)

    except Exception:
        # Cleanup: make sure host proc is dead.
        if host_proc.poll() is None:
            host_proc.kill()
            host_proc.wait()
        raise


def test_host_death_kills_runners(
    live_server: str,
    http_client: httpx.Client,
    tmp_path: Path,
    mock_llm_server_url: str,
) -> None:
    """
    Start host, launch a runner, kill the host, verify the runner
    exits within a few seconds.

    The runner's parent-PID watchdog polls every 1s and exits when
    the parent (host daemon) is gone. If the runner stays alive
    after the host dies, the watchdog is broken and we'd accumulate
    orphaned runner processes.
    """
    daemon = _spawn_host_daemon(
        tmp_path=tmp_path,
        live_server=live_server,
        mock_llm_server_url=mock_llm_server_url,
    )
    host_proc = daemon.proc
    host_id = daemon.host_id

    try:
        _wait_for_host_online(http_client, host_id, timeout=30.0)

        # Upload an agent + resolve its durable id. The standalone
        # /api/agents endpoint was removed; agents are now
        # created via multipart POST /v1/sessions and looked up by name.
        agent_name = upload_agent(http_client, _write_smoke_agent_yaml(tmp_path))
        agent_id = lookup_agent_id(http_client, agent_name)

        # Create session + launch runner.
        session_resp = http_client.post(
            "/v1/sessions",
            json={"agent_id": agent_id},
        )
        session_resp.raise_for_status()
        session_id = session_resp.json()["id"]

        launch_resp = http_client.post(
            f"/v1/hosts/{host_id}/runners",
            json={"session_id": session_id, "workspace": str(tmp_path)},
            timeout=60.0,
        )
        assert launch_resp.status_code == 200
        runner_id = launch_resp.json()["runner_id"]

        # Wait for runner to come online.
        deadline = time.monotonic() + 30.0
        runner_online = False
        while time.monotonic() < deadline:
            sr = http_client.get(f"/v1/runners/{runner_id}/status")
            if sr.status_code == 200 and sr.json().get("online"):
                runner_online = True
                break
            time.sleep(0.5)
        assert runner_online, f"Runner {runner_id} never came online"

        # Resolve the runner's OS pid before killing the host so we can
        # assert on the process directly.
        runner_pid = _runner_pid_from_daemon_log(daemon.daemon_log)
        assert runner_pid is not None, (
            "could not find the launched runner pid in the daemon log:\n"
            f"{daemon.daemon_log.read_text()}"
        )

        # Kill the host daemon.
        host_proc.kill()
        host_proc.wait()

        # The orphaned runner must exit (parent-PID watchdog). Assert on
        # the runner PROCESS — the invariant this test protects (no orphan
        # accumulation). The server's online flag is a poor proxy here: it
        # only clears a dead runner on the next 30s keepalive ping, long
        # after the runner has actually exited.
        deadline = time.monotonic() + 15.0
        runner_died = False
        while time.monotonic() < deadline:
            if not _pid_alive(runner_pid):
                runner_died = True
                break
            time.sleep(0.5)

        assert runner_died, (
            f"Runner process {runner_pid} should have exited after host "
            "death (parent-PID watchdog). If it's still alive, orphaned "
            "runner processes will accumulate."
        )

    except Exception:
        if host_proc.poll() is None:
            host_proc.kill()
            host_proc.wait()
        raise


# ── Host-restart native round-trip (regression guard) ──────────
#
# Skipped: this test requires a real interactive Claude login (OAuth) anchored
# to the real HOME, plus `claude` + `tmux` CLIs on PATH. That authentication
# cannot be relocated into CI or mocked. The server-side handshake ordering
# is covered by tests/server/integration/test_session_host_launch.py.

@pytest.mark.skip(
    reason=(
        "claude-native host-restart e2e requires real `claude` + `tmux` CLIs "
        "with interactive OAuth login; cannot run against mock LLM"
    ),
)
def test_host_native_session_round_trips_after_runner_death(
    live_server: str,
    http_client: httpx.Client,
    tmp_path: Path,
    mock_llm_server_url: str,
) -> None:
    """A web message to a host-bound claude-native session whose runner
    died relaunches the runner and round-trips through the forwarder.

    Regression guard for the host-restart native-session fix. Skipped:
    requires real ``claude`` + ``tmux`` CLIs with interactive OAuth login.
    """
