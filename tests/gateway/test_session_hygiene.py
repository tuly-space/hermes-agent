"""Gateway hygiene boundary: disabled compression keeps a bounded model input.

Enabled compression passes complete history to AIAgent's assembled-request
preflight (test_unified_compaction_boundary.py); it does not run a detached
Gateway compressor. Shared runner scaffolding below uses a real SessionDB.
"""

import importlib
import sys
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from agent.model_metadata import estimate_messages_tokens_rough
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _make_history(n_messages: int, content_size: int = 100) -> list:
    """Build a transcript with alternating user/assistant messages."""
    content = "x" * content_size
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": content,
         "timestamp": f"t{i}"}
        for i in range(n_messages)
    ]


class HygieneCaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content,
                          "reply_to": reply_to, "metadata": metadata})
        return SendResult(success=True, message_id="hygiene-1")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _make_cooldown_runner(monkeypatch, tmp_path, agent_cls, session_db, session_id):
    """Fresh GatewayRunner with a real AsyncSessionDB facade and fake transport.

    The AIAgent-only module placeholder detects accidental resurrection of a
    Gateway detached compressor. Tests running the real agent restore the real
    module before entering run_conversation.
    """
    from hermes_state import AsyncSessionDB

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    (tmp_path / "config.yaml").write_text("compression:\n  enabled: true\n")
    gateway_run = importlib.import_module("gateway.run")
    adapter = HygieneCaptureAdapter()
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm:12345", session_id=session_id,
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = _make_history(6, content_size=400)
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = AsyncSessionDB(session_db)
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(return_value={
        "final_response": "ok", "messages": [], "tools": [],
        "history_offset": 0, "last_prompt_tokens": 0,
    })
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length",
                        lambda *_args, **_kwargs: 100)
    event = MessageEvent(
        text="hello",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345",
                             chat_type="dm", user_id="12345"),
        message_id="1",
    )
    return runner, adapter, event


def _make_bound_probe_transcript(total: int = 60) -> list:
    """Setup prefix + old tool group straddling the tail cut + newest tail."""
    rows = [{"role": "session_meta", "tools": [], "model": "m",
             "platform": "telegram", "timestamp": "t0"}]
    while len(rows) < total - 20:
        role = "user" if len(rows) % 2 else "assistant"
        rows.append({"role": role, "content": f"old-{len(rows)}", "timestamp": f"t{len(rows)}"})
    rows.append({"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c1", "function": {"name": "t"}}]})
    rows.append({"role": "tool", "tool_call_id": "c1", "content": "old tool output 1"})
    rows.append({"role": "tool", "tool_call_id": "c2", "content": "old tool output 2"})
    rows.append({"role": "user", "content": "newest ask", "timestamp": "t-ask"})
    rows.append({"role": "assistant", "content": "newest reply", "timestamp": "t-reply"})
    while len(rows) < total:
        role = "user" if len(rows) % 2 else "assistant"
        rows.append({"role": role, "content": f"tail-{len(rows)}", "timestamp": f"t{len(rows)}"})
    if total == 60:
        assert rows[41]["role"] == "tool", "fixture must cut through a tool group"
    return rows


def test_rough_token_estimation_scales_with_content():
    assert estimate_messages_tokens_rough(_make_history(10, content_size=10_000)) > (
        estimate_messages_tokens_rough(_make_history(10, content_size=100))
    )


def test_bound_model_input_without_hygiene_is_deterministic_and_fail_closed():
    """Disabled compression keeps setup + newest tail without mutating storage."""
    from gateway.run_turn import bound_model_input_without_hygiene

    rows = _make_bound_probe_transcript()
    snapshot = [dict(r) for r in rows]
    limit = 20
    bounded = bound_model_input_without_hygiene(rows, limit)
    assert len(bounded) <= limit < len(rows)
    assert bounded[0] is rows[0] and bounded[-1] is rows[-1]
    assert bounded[1].get("role") != "tool"
    assert rows == snapshot
    assert bound_model_input_without_hygiene(rows, limit) == bounded
    assert bound_model_input_without_hygiene(rows, len(rows)) is rows
    assert bound_model_input_without_hygiene(rows, len(rows) + 5) is rows
