"""Native-first maintenance: raw SSE and durable checkpoint CAS."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import native_maintenance as maintenance
from agent.codex_responses_adapter import has_replayable_native_compaction_checkpoint
from agent.native_compaction import resolve_native_compaction_capabilities
from agent.transports.codex import ResponsesApiTransport
from hermes_state import SessionDB


@pytest.fixture
def session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("same-session", source="cli", model="gpt-6-astra")
    db.append_messages_batch("same-session", [
        {"role": "user", "content": "An old fact"},
        {"role": "assistant", "content": "Acknowledged"},
    ])
    try:
        yield db
    finally:
        db.close()


def _agent(db, stream):
    model, provider, base_url = "gpt-6-astra", "openai-codex", "https://chatgpt.com/backend-api/codex"
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=iter(stream))))
    agent = SimpleNamespace(
        model=model, provider=provider, base_url=base_url, api_mode="codex_responses",
        runtime_capabilities=resolve_native_compaction_capabilities(
            model=model, provider=provider, base_url=base_url, is_codex_backend=True),
        codex_responses_native_compaction=True, compression_native_first=True,
        compression_enabled=True, compression_checkpoint_required=False,
        context_compressor=SimpleNamespace(threshold_tokens=100_000, context_length=272_000,
                                           note_native_compaction_checkpoint=Mock()),
        _session_db=db, session_id="same-session", tools=[{"type": "function", "function": {
            "name": "read_only", "parameters": {"type": "object", "properties": {}}}}],
        _cached_system_prompt="Frozen instruction bytes", _get_transport=lambda: ResponsesApiTransport(),
        _create_request_openai_client=Mock(return_value=client),
        _close_request_openai_client=Mock(),
        _abort_request_openai_client=Mock(),
    )
    return agent, client


def _events(item=None, status="completed"):
    result = []
    if item is not None:
        result.append({"type": "response.output_item.done", "item": item})
    result.append({"type": "response.completed", "response": {
        "status": status, "output": [], "usage": {"input_tokens": 172_000,
        "input_tokens_details": {"cached_tokens": 40_000}, "output_tokens": 450}}})
    return result








def _run(monkeypatch, agent, db):
    from agent import chat_completion_helpers, turn_request_assembly
    # The minimal stand-in has no AIAgent message-copy/sanitizer methods.
    monkeypatch.setattr(turn_request_assembly, "maintenance_api_prefix",
                        lambda a, rows, prompt: [{"role": "system", "content": prompt}] + [
                            {"role": m["role"], "content": m.get("content", "")} for m in rows])
    # The ordinary request builder is exercised in this test without a credential
    # resolver; only the actual HTTP client is replaced by the event fixture.
    monkeypatch.setattr(chat_completion_helpers, "build_api_kwargs", lambda a, messages: {
        "model": a.model, "instructions": messages[0]["content"],
        "input": [{"role": m["role"], "content": m["content"]} for m in messages[1:]],
        "tools": [{"type": "function", "name": "read_only", "parameters": {"type": "object", "properties": {}}}],
        "context_management": [{"type": "compaction", "compact_threshold": 90_000}],
        "store": False,
    })
    messages = db.get_messages_as_conversation("same-session", include_row_ids=True)
    return maintenance.attempt(agent, messages, agent._cached_system_prompt, 175_000,
                               phase="threshold", expected_watermark=db.get_active_message_watermark("same-session")), messages


def test_native_checkpoint_from_output_item_done_is_durable_without_chat_delivery(monkeypatch, session):
    agent, client = _agent(session, _events({"type": "compaction", "encrypted_content": "sealed"}))
    succeeded, messages = _run(monkeypatch, agent, session)
    assert succeeded
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["instructions"] == agent._cached_system_prompt
    assert kwargs["context_management"] == [{"type": "compaction", "compact_threshold": 1024}]
    assert kwargs["extra_body"]["tools"][0]["name"] == "read_only"
    assert kwargs["tool_choice"] == "none"
    assert kwargs["extra_body"]["input"][-1]["role"] == "user"
    assert agent.context_compressor.note_native_compaction_checkpoint.call_count == 1
    assert [m["content"] for m in session.get_messages_as_conversation("same-session")] == ["An old fact", "Acknowledged"]
    restarted = session.get_messages_as_conversation("same-session", include_row_ids=True)
    assert has_replayable_native_compaction_checkpoint(agent, restarted)
    assert messages[-1]["codex_reasoning_items"][-1]["encrypted_content"] == "sealed"


def test_real_builder_replays_big_tool_checkpoint_at_covered_boundary(monkeypatch, session):
    from run_agent import AIAgent
    from agent.chat_completion_helpers import build_api_kwargs

    session.append_messages_batch("same-session", [
        {"role": "user", "content": "Fetch the large payload"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_big", "call_id": "call_big", "type": "function",
            "function": {"name": "read_only", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_big", "content": "valuable tool data" * 100},
    ])
    agent = AIAgent(api_key="test-key", base_url="https://api.openai.com/v1",
                    api_mode="codex_responses", model="gpt-5.6", provider="openai-api",
                    session_db=session, session_id="same-session", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    agent.codex_responses_native_compaction = True
    agent.compression_native_first = True
    agent._cached_system_prompt = "Frozen instruction bytes"
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=iter(
        _events({"type": "compaction", "encrypted_content": "sealed-big"})))))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **kw: None)
    try:
        original = session.get_messages_as_conversation("same-session", include_row_ids=True)
        watermark = session.get_active_message_watermark("same-session")
        assert maintenance.attempt(agent, original, agent._cached_system_prompt, 100_000,
                                   phase="threshold", expected_watermark=watermark)
        wire = client.responses.create.call_args.kwargs
        assert wire["tool_choice"] == "none"
        assert wire["extra_body"]["input"][-1]["role"] == "user"  # inert maintenance tail
        assert any(item.get("type") == "function_call_output" for item in wire["extra_body"]["input"])
        restored = session.get_messages_as_conversation("same-session", include_row_ids=True)
        assert [m["content"] for m in restored] == [m["content"] for m in original]
        replay = build_api_kwargs(agent, [{"role": "system", "content": agent._cached_system_prompt},
                                         *restored, {"role": "user", "content": "Next real request"}])
        items = replay["input"]
        checkpoint_at = next(i for i, item in enumerate(items) if item.get("type") == "compaction")
        assert items[checkpoint_at]["encrypted_content"] == "sealed-big"
        assert all(item.get("type") != "function_call_output" for item in items[checkpoint_at + 1:])
        assert items[-1]["role"] == "user"
        assert items[-1]["content"] == "Next real request"
        # The gateway's canonical transcript reader omits row ids.
        without_ids = session.get_messages_as_conversation("same-session")
        no_ids_wire = build_api_kwargs(agent, [{"role": "system", "content": agent._cached_system_prompt},
                                               *without_ids, {"role": "user", "content": "Next real request"}])
        assert any(item.get("type") == "compaction" for item in no_ids_wire["input"])
        without_ids[0]["content"] = "edited old fact"
        edited_wire = build_api_kwargs(agent, [{"role": "system", "content": agent._cached_system_prompt},
                                               *without_ids, {"role": "user", "content": "Next real request"}])
        assert all(item.get("type") != "compaction" for item in edited_wire["input"])
        assert any(item.get("type") == "function_call_output" for item in edited_wire["input"])
    finally:
        agent.close()










@pytest.mark.parametrize("stream", [_events(), _events({"type": "compaction", "encrypted_content": ""}),
                                    _events({"type": "compaction", "encrypted_content": "sealed"}, status="failed")])
def test_missing_or_failed_checkpoint_never_mutates_history(monkeypatch, session, stream):
    agent, _ = _agent(session, stream)
    succeeded, messages = _run(monkeypatch, agent, session)
    assert not succeeded
    assert "codex_reasoning_items" not in messages[-1]
    assert not has_replayable_native_compaction_checkpoint(agent, session.get_messages_as_conversation("same-session"))


def test_stale_watermark_or_active_turn_rejects_checkpoint(session):
    row = session.get_messages_as_conversation("same-session", include_row_ids=True)[-1]
    checkpoint = {"type": "compaction", "encrypted_content": "sealed"}
    watermark = session.get_active_message_watermark("same-session")
    session.append_message("same-session", "user", "new message")
    assert not session.attach_native_checkpoint("same-session", watermark, row["_row_id"], checkpoint)
    owner = "pid=1:turn=active"
    session.try_acquire_session_turn_lease("same-session", owner)
    try:
        with pytest.raises(Exception):
            session.attach_native_checkpoint("same-session", session.get_active_message_watermark("same-session"),
                                             row["_row_id"], checkpoint)
    finally:
        session.release_session_turn_lease("same-session", owner)






def test_delayed_provider_output_cannot_commit_after_new_message(monkeypatch, session):
    agent, client = _agent(session, [])
    class Stream:
        def __iter__(self):
            session.append_message("same-session", "user", "arrived while streaming")
            return iter(_events({"type": "compaction", "encrypted_content": "stale"}))
        def close(self):
            pass
    client.responses.create.return_value = Stream()
    succeeded, _ = _run(monkeypatch, agent, session)
    assert not succeeded
    assert not has_replayable_native_compaction_checkpoint(agent, session.get_messages_as_conversation("same-session"))




def test_truncated_stream_with_compaction_item_is_not_committed(monkeypatch, session):
    agent, _ = _agent(session, [{"type": "response.output_item.done",
                               "item": {"type": "compaction", "encrypted_content": "partial"}}])
    succeeded, _ = _run(monkeypatch, agent, session)
    assert not succeeded
    assert not has_replayable_native_compaction_checkpoint(
        agent, session.get_messages_as_conversation("same-session"))


def test_fence_revoked_during_native_stream_rejects_checkpoint_and_summary(tmp_path, monkeypatch):
    from run_agent import AIAgent
    from agent.conversation_compression import CompressionCommitFence
    from agent import conversation_compression
    agent = AIAgent(api_key="test-key", base_url="https://api.openai.com/v1",
                    api_mode="codex_responses", model="gpt-5.6", provider="openai-api",
                    quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    agent.compression_native_first = agent.codex_responses_native_compaction = True
    fence = CompressionCommitFence()
    summaries = Mock()
    monkeypatch.setattr(conversation_compression, "compress_context", summaries)
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock()))
    class Stream:
        def __iter__(self):
            assert agent._active_compression_commit_fence is fence
            fence.revoke_commit_admission()
            return iter(_events({"type": "compaction", "encrypted_content": "stale"}))
        def close(self):
            pass
    client.responses.create.return_value = Stream()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **kw: None)
    try:
        rows = [{"role": "user", "content": "fact"}, {"role": "assistant", "content": "ack"}]
        result, _ = agent._compress_context(rows, "frozen", approx_tokens=agent.context_compressor.threshold_tokens,
                                            commit_fence=fence)
        assert result is rows
        summaries.assert_not_called()
        assert not agent._native_maintenance_committed
        assert not hasattr(agent, "_active_compression_commit_fence")
    finally:
        agent.close()


def test_maintenance_uses_send_transforms_for_digest_and_tool_boundary(session, monkeypatch):
    from run_agent import AIAgent
    from agent.turn_request_assembly import maintenance_api_prefix
    from agent.chat_completion_helpers import build_api_kwargs
    session.append_messages_batch("same-session", [
        {"role": "user", "content": "  raw  ", "api_content": "  injected fact  "},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_big", "type": "function",
          "function": {"name": "read_only", "arguments": "{ \"z\": 1 }"}}]},
        {"role": "tool", "tool_call_id": "call_big", "content": "  tool evidence  "},
    ])
    agent = AIAgent(api_key="test-key", base_url="https://api.openai.com/v1",
                    api_mode="codex_responses", model="gpt-5.6", provider="openai-api",
                    session_db=session, session_id="same-session", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    agent.codex_responses_native_compaction = agent.compression_native_first = True
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=iter(
        _events({"type": "compaction", "encrypted_content": "sealed-transform"})))))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **kw: None)
    try:
        rows = session.get_messages_as_conversation("same-session", include_row_ids=True)
        # Freshly flushed in-memory text keeps whitespace that DB replay strips.
        # It is still the same durable prefix and must get a native attempt.
        rows[0]["content"] = "  " + rows[0]["content"] + "  "
        assert maintenance.attempt(agent, rows, "frozen", 100_000, phase="threshold",
                                   expected_watermark=session.get_active_message_watermark("same-session"))
        wire = client.responses.create.call_args.kwargs["extra_body"]["input"]
        assert any(i.get("content") == "injected fact" for i in wire)
        assert any(i.get("type") == "function_call_output" for i in wire)
        restored = session.get_messages_as_conversation("same-session")
        replay = maintenance_api_prefix(agent, restored + [{"role": "user", "content": "next"}], "frozen")
        input_items = build_api_kwargs(agent, replay)["input"]
        at = next(i for i, item in enumerate(input_items) if item.get("type") == "compaction")
        assert input_items[at]["encrypted_content"] == "sealed-transform"
        assert input_items[-1]["content"] == "next"
        assert all(i.get("type") != "function_call_output" for i in input_items[at+1:])
        restored[2]["api_content"] = "altered fact"
        assert all(i.get("type") != "compaction" for i in build_api_kwargs(
            agent, maintenance_api_prefix(agent, restored + [{"role": "user", "content": "next"}], "frozen"))["input"])
    finally:
        agent.close()








@pytest.mark.parametrize("native_success", [True, False])
def test_auto_compress_attempts_native_before_summary_including_tool_tail(tmp_path, monkeypatch, native_success):
    from run_agent import AIAgent
    from agent import conversation_compression

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "compression:\n  codex_responses_native: true\n"
        "  codex_responses_native_first: true\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent = AIAgent(api_key="test-key", base_url="https://api.openai.com/v1",
                    api_mode="codex_responses", model="gpt-5.6", provider="openai-api",
                    quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    calls = []
    monkeypatch.setattr(maintenance, "attempt", lambda *a, **kw: calls.append("native") or native_success)
    monkeypatch.setattr(conversation_compression, "resolve_context_compression_timeouts", lambda: (0, 0))
    monkeypatch.setattr(conversation_compression, "compress_context",
                        lambda *a, **kw: (calls.append("summary") or [{"role": "assistant", "content": "summary"}], "frozen"))
    messages = [
        {"role": "system", "content": "frozen"},
        {"role": "user", "content": "read"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call", "function": {"name": "read_only", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call", "content": "large output"},
    ]
    try:
        result, _ = agent._compress_context(messages, "frozen", approx_tokens=agent.context_compressor.threshold_tokens)
        assert calls == (["native"] if native_success else ["native", "summary"])
        assert (result is not messages) if native_success else result[0]["content"] == "summary"
    finally:
        agent.close()
