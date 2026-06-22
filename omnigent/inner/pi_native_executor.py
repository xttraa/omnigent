"""Executor that bridges Omnigent messages into a native Pi TUI."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import secrets
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from omnigent.inner.executor import (
    Executor,
    ExecutorConfig,
    ExecutorError,
    ExecutorEvent,
    Message,
    ToolSpec,
    TurnComplete,
)
from omnigent.inner.native_attachments import materialize_attachment
from omnigent.pi_native_bridge import (
    PI_NATIVE_BRIDGE_DIR_ENV_VAR,
    PI_NATIVE_REQUEST_SESSION_ID_ENV_VAR,
    clear_policy_server_config,
    enqueue_user_message,
    write_policy_server_config,
)

logger = logging.getLogger(__name__)


class _PolicyServer:
    """Minimal TCP server for TOOL_CALL policy evaluation from the Pi native extension."""

    def __init__(self) -> None:
        self._server: asyncio.Server | None = None
        self.port: int = 0
        self.token: str = secrets.token_urlsafe(32)
        self._policy_gate: Any = None

    async def start(self) -> int:
        self._server = await asyncio.start_server(self._handle_client, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            raw = await reader.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                return
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                return
            token = request.get("token")
            if not isinstance(token, str) or not hmac.compare_digest(token, self.token):
                writer.write(
                    json.dumps({"id": request.get("id"), "error": "unauthorized"}).encode() + b"\n"
                )
                await writer.drain()
                return
            req_id = request.get("id")
            tool_name = request.get("tool")
            if not isinstance(req_id, str) or not isinstance(tool_name, str):
                return
            tool_args = request.get("args", {})
            verdict = await self._evaluate_policy(tool_name, tool_args)
            writer.write((json.dumps({"id": req_id, "verdict": verdict}) + "\n").encode())
            await writer.drain()
        except (asyncio.CancelledError, ConnectionError):
            pass
        finally:
            writer.close()

    async def _evaluate_policy(self, name: str, args: dict) -> dict:
        if self._policy_gate is None:
            return {"block": False, "reason": ""}
        try:
            raw = self._policy_gate(name, args)
            resolved = await raw if asyncio.iscoroutine(raw) or asyncio.isfuture(raw) else raw
            if isinstance(resolved, dict):
                return {
                    "block": bool(resolved.get("block")),
                    "reason": str(resolved.get("reason") or ""),
                }
            return {"block": False, "reason": ""}
        except Exception as exc:  # noqa: BLE001 — fail-open; the verdict path must never wedge Pi
            logger.warning("Pi native policy eval failed for %r: %s", name, exc)
            return {"block": False, "reason": ""}


class PiNativeExecutor(Executor):
    """
    Harness-side executor for ``omnigent pi`` web UI turns.

    The native Pi process is already running in the session terminal with
    the Omnigent Pi extension loaded. Each turn queues the latest user
    message into the bridge inbox; the extension consumes it and calls
    ``pi.sendUserMessage`` inside the TUI process.
    """

    def __init__(self, bridge_dir: Path | None = None) -> None:
        self._bridge_dir = bridge_dir or _bridge_dir_from_env()
        self._request_session_id = _request_session_id_from_env()
        self._policy_server: _PolicyServer | None = None

    def supports_streaming(self) -> bool:
        """:returns: ``False`` because output is emitted by the Pi extension."""
        return False

    def supports_live_message_queue(self) -> bool:
        """:returns: ``True`` because messages can be queued for the extension."""
        return True

    async def enqueue_session_message(self, session_key: str, content: Any) -> bool:
        """
        Queue a live steering message for the resident Pi extension.

        :param session_key: Adapter session key. Unused; the bridge is
            per conversation.
        :param content: User-supplied content.
        :returns: ``True`` when the message was queued.
        """
        del session_key
        text = _content_to_text(content, self._bridge_dir)
        if not text:
            return False
        enqueue_user_message(self._bridge_dir, text)
        return True

    async def _gate_native_tool(self, name: str, args: dict) -> dict:
        evaluator = getattr(self, "_policy_evaluator", None)
        if evaluator is None:
            return {"block": False, "reason": ""}
        verdict = await evaluator("PHASE_TOOL_CALL", {"name": name, "arguments": args})
        if verdict.action == "POLICY_ACTION_DENY":
            return {"block": True, "reason": verdict.reason or "blocked by policy"}
        return {"block": False, "reason": ""}

    async def _ensure_policy_server(self) -> None:
        if self._policy_server is not None:
            return
        server = _PolicyServer()
        await server.start()
        server._policy_gate = self._gate_native_tool
        self._policy_server = server
        write_policy_server_config(self._bridge_dir, server.port, server.token)

    async def close_session(self, session_key: str) -> None:
        """
        Stop the policy server and remove its config file.

        :param session_key: Adapter session key. Unused.
        """
        del session_key
        if self._policy_server is not None:
            await self._policy_server.stop()
            self._policy_server = None
        clear_policy_server_config(self._bridge_dir)

    async def close(self) -> None:
        """Stop the policy server and remove its config file."""
        if self._policy_server is not None:
            await self._policy_server.stop()
            self._policy_server = None
        clear_policy_server_config(self._bridge_dir)

    async def run_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[ExecutorEvent]:
        """
        Queue the latest user message for Pi.

        :param messages: Conversation history in executor message shape.
        :param tools: Tool schemas from Omnigent. Ignored for now; native
            Pi owns its configured tool surface.
        :param system_prompt: System prompt from the agent spec. Ignored
            because the native Pi terminal controls its own prompt/settings.
        :param config: Per-turn executor config. Unused.
        :yields: :class:`TurnComplete` after the input was queued, or an
            :class:`ExecutorError` when no user text can be sent.
        """
        del tools, system_prompt, config
        await self._ensure_policy_server()
        text = _latest_user_text(messages, self._bridge_dir)
        if not text:
            yield ExecutorError(message="Pi native turn had no user text to send")
            return
        enqueue_user_message(self._bridge_dir, text)
        yield TurnComplete(response=None)


def _bridge_dir_from_env() -> Path:
    """
    Resolve the native Pi bridge directory from harness spawn env.

    :returns: Bridge directory path.
    :raises RuntimeError: If the env var is missing.
    """
    raw = os.environ.get(PI_NATIVE_BRIDGE_DIR_ENV_VAR, "").strip()
    if not raw:
        raise RuntimeError(f"{PI_NATIVE_BRIDGE_DIR_ENV_VAR} is required for pi-native harness")
    return Path(raw)


def _request_session_id_from_env() -> str | None:
    """
    Resolve the Omnigent session id that requested this harness process.

    :returns: Omnigent session id, e.g. ``"conv_abc123"``, or ``None``.
    """
    raw = os.environ.get(PI_NATIVE_REQUEST_SESSION_ID_ENV_VAR, "").strip()
    return raw or None


def _latest_user_text(messages: list[Message], bridge_dir: Path) -> str:
    """
    Return the latest user text from executor messages.

    :param messages: Conversation history in executor message shape.
    :param bridge_dir: Bridge directory for materializing attachments.
    :returns: Plain text content, or ``""`` when no user text is present.
    """
    for message in reversed(messages):
        if message.get("role") == "user":
            return _content_to_text(message.get("content"), bridge_dir)
    return ""


def _content_to_text(content: Any, bridge_dir: Path) -> str:
    """
    Normalize executor content into plain text for Pi.

    Text blocks are extracted directly. Image/file blocks are materialized
    to the bridge directory and referenced by path so Pi can inspect them
    with its native filesystem tools.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        attachment_lines: list[str] = []
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type == "input_text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif block_type in ("input_image", "input_file"):
                path = materialize_attachment(block, bridge_dir)
                if path is not None:
                    attachment_lines.append(f"[Attached: {path}]")
        return "\n\n".join([*attachment_lines, *text_parts])
    if content is None:
        return ""
    return str(content)
