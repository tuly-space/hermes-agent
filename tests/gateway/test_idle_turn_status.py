"""Cold next-turn idle compaction uses the normal gateway status rail."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.mark.parametrize("old", [True, False])
def test_cold_turn_status_before_provider_and_short_gap_skips(tmp_path, monkeypatch, old):
    from gateway import run
    from tests.gateway.test_warning_notifications import RecordingAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Exercise the real adapter status coroutine rather than bypassing gateway filtering.
    monkeypatch.setattr(run, "safe_schedule_threadsafe", lambda coro, *a, **kw: asyncio.run(coro))
    calls = []
    class StatusAdapter(RecordingAdapter):
        async def send(self, chat_id, content, **kwargs):
            calls.append("status")
            return await super().send(chat_id, content, **kwargs)

    adapter = StatusAdapter()
    source = SessionSource(platform=Platform.DISCORD, chat_id="chat", user_id="user", thread_id="thread")
    ctx = TurnContext(source=source, user_config={}, _run_still_current=lambda: True,
                      _status_adapter=adapter, _status_chat_id="chat",
                      _status_thread_metadata={"thread_id": "thread"})
    runner = TurnRunner(object.__new__(GatewayRunner), ctx)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("cold", source="discord")
    then = time.time() - (7200 if old else 30)
    db.append_messages_batch("cold", [
        {"role": "user", "content": "old question", "timestamp": then},
        {"role": "assistant", "content": "old answer", "timestamp": then},
    ])
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                    model="test-model", provider="openrouter", session_db=db,
                    session_id="cold", quiet_mode=True, skip_memory=True,
                    skip_context_files=True, enabled_toolsets=[])
    agent._last_activity_ts = time.time()  # a newly restored agent has a fresh clock
    agent.compression_idle_compact_after_seconds = 3600
    agent.context_compressor.threshold_tokens = 100_000
    agent.context_compressor.summary_target_ratio = .2
    agent.context_compressor.awaiting_real_usage_after_compression = False
    agent.context_compressor.get_active_compression_failure_cooldown = lambda: None
    agent.status_callback = runner._status_callback_sync
    def compact(messages, prompt, **kwargs):
        calls.append("compact")
        return list(messages), prompt
    monkeypatch.setattr(agent, "_compress_context", compact)
    agent.client = MagicMock()
    def provider(**kwargs):
        calls.append("provider")
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Reply", tool_calls=None, reasoning_content=None, reasoning=None),
            finish_reason="stop")], usage=None, model="test-model")
    agent.client.chat.completions.create.side_effect = provider
    from agent import turn_context
    monkeypatch.setattr(turn_context, "estimate_request_tokens_rough", lambda *a, **kw: 50_000)
    try:
        result = agent.run_conversation("new question", conversation_history=db.get_messages_as_conversation("cold"))
        assert result["completed"]
        assert calls[-1] == "provider"
        if old:
            assert calls == ["status", "compact", "provider"]
            assert len(adapter.sent) == 1
            assert "Resumed after" in adapter.sent[0][1]
            assert adapter.sent[0][2]["metadata"]["thread_id"] == "thread"
        else:
            assert calls == ["provider"]
            assert adapter.sent == []
    finally:
        agent.close()
        db.close()


def test_native_cold_turn_status_precedes_auxiliary_request(tmp_path, monkeypatch):
    from gateway import run
    from tests.gateway.test_warning_notifications import RecordingAdapter
    from agent import turn_context

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compression:\n  codex_gpt55_autoraise: false\n  idle_compact_after_seconds: 3600\n"
        "auxiliary:\n  compression:\n    provider: openai-codex\n"
        "    model: gpt-6-sol\n    native: true\n    preserve_reasoning: true\n"
    )
    events = []
    class StatusAdapter(RecordingAdapter):
        async def send(self, chat_id, content, **kwargs):
            events.append("status")
            return await super().send(chat_id, content, **kwargs)

    monkeypatch.setattr(run, "safe_schedule_threadsafe", lambda coro, *a, **kw: asyncio.run(coro))
    adapter = StatusAdapter()
    source = SessionSource(platform=Platform.DISCORD, chat_id="chat", user_id="user", thread_id="thread")
    ctx = TurnContext(source=source, user_config={}, _run_still_current=lambda: True,
                      _status_adapter=adapter, _status_chat_id="chat",
                      _status_thread_metadata={"thread_id": "thread"})
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("cold-native", source="discord", model="gpt-6-astra")
    then = time.time() - 7200
    db.append_messages_batch("cold-native", [
        {"role": "user", "content": "old question", "timestamp": then},
        {"role": "assistant", "content": "old answer", "timestamp": then},
    ])
    agent = AIAgent(api_key="test-key", base_url="https://chatgpt.com/backend-api/codex",
                    api_mode="codex_responses", model="gpt-6-astra", provider="openai-codex",
                    session_db=db, session_id="cold-native", quiet_mode=True,
                    skip_memory=True, skip_context_files=True, enabled_toolsets=[])
    agent._last_activity_ts = time.time()
    agent.compression_idle_compact_after_seconds = 3600
    agent.context_compressor.threshold_tokens = 100_000
    agent.context_compressor.summary_target_ratio = .2
    agent.status_callback = TurnRunner(object.__new__(GatewayRunner), ctx)._status_callback_sync
    from agent import native_maintenance
    assert native_maintenance.eligible(agent)
    assert db.get_messages_as_conversation("cold-native")[-1]["timestamp"] < time.time() - 3600
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: (
        events.append("aux") or iter([
            {"type": "response.output_item.done", "item": {"type": "compaction", "encrypted_content": "sealed"}},
            {"type": "response.completed", "response": {"status": "completed", "output": []}},
        ]))))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **kw: None)
    response = SimpleNamespace(output=[SimpleNamespace(
        type="message", status="completed", content=[SimpleNamespace(type="output_text", text="Reply")]
    )], usage=SimpleNamespace(input_tokens=3000, output_tokens=5, total_tokens=3005),
        status="completed", incomplete_details=None, model="gpt-6-astra")
    def main_response(api_kwargs, **kwargs):
        events.append("provider")
        assert api_kwargs["model"] == "gpt-6-astra"
        assert "context_management" not in api_kwargs
        assert any(item.get("type") == "compaction" and item.get("encrypted_content") == "sealed"
                   for item in api_kwargs["input"])
        return response
    monkeypatch.setattr(agent, "_run_codex_stream", main_response)
    # Codex uses the native wire estimator, not the generic fallback estimator.
    # Stay below the ordinary threshold so only idle admission triggers this pass.
    monkeypatch.setattr(turn_context, "_preflight_request_tokens", lambda *a, **kw: 50_000)
    try:
        result = agent.run_conversation("new question", conversation_history=db.get_messages_as_conversation("cold-native"))
        assert result["completed"]
        assert adapter.sent and "Resumed after" in adapter.sent[0][1], (events, adapter.sent)
        assert events == ["status", "aux", "provider"]
        assert adapter.sent[0][2]["metadata"]["thread_id"] == "thread"
        saved = db.get_messages_as_conversation("cold-native")
        carrier = next(m for m in saved if m.get("content") == "old answer")
        assert any(i.get("type") == "compaction" for i in carrier.get("codex_reasoning_items", []))
        assert [m["content"] for m in saved if m["role"] == "user"] == ["old question", "new question"]
    finally:
        agent.close()
        db.close()
