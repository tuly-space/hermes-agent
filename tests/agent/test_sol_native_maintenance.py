"""The official Astra/Sol maintenance route and durable replay contract."""

import copy
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import native_maintenance
from agent.chat_completion_helpers import build_api_kwargs
from agent.turn_retry_state import TurnRetryState
from agent.turn_recovery import _recover_stale_codex_reasoning
from hermes_state import SessionDB
from run_agent import AIAgent


_CODEX_URL = "https://chatgpt.com/backend-api/codex"


def _agent(home, monkeypatch, db=None, *, native=True, preserve=True,
           provider="openai-codex", write_config=True):
    home.mkdir(exist_ok=True)
    if write_config:
        (home / "config.yaml").write_text(
            "auxiliary:\n  compression:\n"
            f"    provider: {provider}\n    model: gpt-6-sol\n"
            + (f"    native: {str(native).lower()}\n" if native is not None else "")
            + f"    preserve_reasoning: {str(preserve).lower()}\n"
        )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return AIAgent(
        api_key="session-credential", base_url=_CODEX_URL, api_mode="codex_responses",
        model="gpt-6-astra", provider="openai-codex", session_db=db,
        session_id="sol-cycle" if db else None, quiet_mode=True,
        skip_context_files=True, skip_memory=True, enabled_toolsets=[],
    )


def _messages():
    return [
        {"role": "user", "content": "Keep this fact"},
        {"role": "assistant", "content": "", "codex_reasoning_items": [{
            "type": "reasoning", "encrypted_content": "opaque-astra-bytes",
            "_issuer_kind": "codex_backend", "_issuer_model": "gpt-6-astra",
        }], "tool_calls": [{"id": "call_one", "type": "function", "function": {
            "name": "read_only", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_one", "content": "tool evidence"},
        {"role": "assistant", "content": "Acknowledged"},
    ]


def _events():
    return iter([
        {"type": "response.output_item.done", "item": {
            "type": "compaction", "encrypted_content": "opaque-sol-checkpoint"}},
        {"type": "response.completed", "response": {"status": "completed", "output": []}},
    ])


@pytest.mark.parametrize("preserve", [True, False])
def test_sol_maintenance_uses_session_route_and_replays_checkpoint_after_resume(tmp_path, monkeypatch, preserve):
    db = SessionDB(db_path=tmp_path / "session.db")
    db.create_session("sol-cycle", source="cli", model="gpt-6-astra")
    db.append_messages_batch("sol-cycle", _messages())
    agent = _agent(tmp_path / "profile", monkeypatch, db, preserve=preserve)
    calls = []
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs) or _events()))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **kw: None)
    try:
        rows = db.get_messages_as_conversation("sol-cycle", include_row_ids=True)
        assert native_maintenance.attempt(
            agent, rows, "Frozen instruction", 100_000, phase="threshold",
            expected_watermark=db.get_active_message_watermark("sol-cycle"),
        )
        assert agent.model == "gpt-6-astra"
        wire = calls[0]
        assert wire["model"] == "gpt-6-sol"
        assert wire["instructions"] == "Frozen instruction"
        assert wire["context_management"] == [{"type": "compaction", "compact_threshold": 1024}]
        assert wire["tool_choice"] == "none"
        items = wire["extra_body"]["input"]
        assert any(i.get("type") == "function_call" for i in items)
        assert any(i.get("type") == "function_call_output" for i in items)
        assert ("opaque-astra-bytes" in str(items)) is preserve
        assert all("opaque-astra-bytes" not in str(i) for i in items if i.get("role") == "user")
        saved = db.get_messages_as_conversation("sol-cycle")
        assert [m["content"] for m in saved] == [m["content"] for m in _messages()]
        assert saved[-1]["codex_reasoning_items"][-1]["_issuer_model"] == "gpt-6-sol"
    finally:
        agent.close()

    fresh = _agent(tmp_path / "profile", monkeypatch, db, preserve=preserve)
    try:
        kwargs = build_api_kwargs(fresh, [{"role": "system", "content": "Frozen instruction"}]
                                  + db.get_messages_as_conversation("sol-cycle")
                                  + [{"role": "user", "content": "Continue"}])
        assert "context_management" not in kwargs
        assert any(i.get("type") == "compaction" and i.get("encrypted_content") == "opaque-sol-checkpoint"
                   for i in kwargs["input"])
        assert kwargs["model"] == "gpt-6-astra"
        foreign = copy.deepcopy(db.get_messages_as_conversation("sol-cycle"))
        foreign[-1]["codex_reasoning_items"][-1]["_issuer_model"] = "gpt-6-other"
        assert not any(i.get("type") == "compaction" for i in build_api_kwargs(fresh, [
            {"role": "system", "content": "Frozen instruction"}] + foreign)["input"])
        for gate in ("compression_enabled", "compression_checkpoint_required"):
            setattr(fresh, gate, gate == "compression_checkpoint_required")
            assert not native_maintenance.eligible(fresh)
            assert not any(i.get("type") == "compaction" for i in build_api_kwargs(fresh, [
                {"role": "system", "content": "Frozen instruction"}]
                + db.get_messages_as_conversation("sol-cycle"))["input"])
            setattr(fresh, gate, gate == "compression_enabled")
        db.append_messages_batch("sol-cycle", [
            {"role": "user", "content": "Next fact"},
            {"role": "assistant", "content": "Another answer", "codex_reasoning_items": [{
                "type": "reasoning", "encrypted_content": "second-astra-bytes",
                "_issuer_kind": "codex_backend", "_issuer_model": "gpt-6-astra",
            }]},
        ])
        second_calls = []
        second_client = SimpleNamespace(responses=SimpleNamespace(
            create=lambda **wire: second_calls.append(wire) or _events()))
        monkeypatch.setattr(fresh, "_create_request_openai_client", lambda **kw: second_client)
        monkeypatch.setattr(fresh, "_close_request_openai_client", lambda *a, **kw: None)
        assert native_maintenance.attempt(
            fresh, db.get_messages_as_conversation("sol-cycle", include_row_ids=True),
            "Frozen instruction", 110_000, phase="threshold",
            expected_watermark=db.get_active_message_watermark("sol-cycle"),
        )
        assert "opaque-sol-checkpoint" in str(second_calls[0]["extra_body"]["input"])
        assert ("second-astra-bytes" in str(second_calls[0]["extra_body"]["input"])) is preserve
    finally:
        fresh.close()
        db.close()


def test_opt_in_controls_and_rejected_checkpoint_survive_new_agent(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    agent = _agent(home, monkeypatch)
    try:
        assert native_maintenance.eligible(agent)
        assert "context_management" not in build_api_kwargs(agent, [
            {"role": "system", "content": "Frozen"}, {"role": "user", "content": "Hello"}])
    finally:
        agent.close()
    other_home = tmp_path / "other-profile"
    unsupported = _agent(other_home, monkeypatch, provider="openrouter")
    try:
        assert not native_maintenance.eligible(unsupported)
    finally:
        unsupported.close()
    from agent.transports.codex import ResponsesApiTransport
    checkpoint_message = {"role": "assistant", "content": "Answer", "codex_reasoning_items": [{
        "type": "compaction", "encrypted_content": "foreign-route-capsule",
        "_issuer_kind": "codex_backend", "_issuer_model": "gpt-6-sol"}]}
    custom = ResponsesApiTransport().build_kwargs(
        model="gpt-6-astra", messages=[{"role": "user", "content": "Question"}, checkpoint_message],
        provider="openai-codex", base_url="https://proxy.example/v1", is_codex_backend=True,
        sol_checkpoint_replay=True,
    )
    assert not any(i.get("type") == "compaction" for i in custom["input"])
    ordinary = _agent(other_home, monkeypatch, native=False)
    try:
        assert not native_maintenance.eligible(ordinary)
    finally:
        ordinary.close()
    back_to_first = _agent(home, monkeypatch, write_config=False)
    try:
        assert native_maintenance.eligible(back_to_first)
    finally:
        back_to_first.close()
    legacy = _agent(tmp_path / "legacy-profile", monkeypatch, native=None)
    try:
        assert legacy.compression_aux_native is None
    finally:
        legacy.close()

    db = SessionDB(db_path=tmp_path / "session.db")
    db.create_session("sol-cycle", source="cli", model="gpt-6-astra")
    db.append_messages_batch("sol-cycle", _messages())
    persisted = db.get_messages_as_conversation("sol-cycle", include_row_ids=True)
    assert db.attach_native_checkpoint(
        "sol-cycle", db.get_active_message_watermark("sol-cycle"), persisted[-1]["_row_id"],
        {"type": "compaction", "encrypted_content": "rejected-sol-checkpoint",
         "_issuer_kind": "codex_backend", "_issuer_model": "gpt-6-sol",
         "_checkpoint_count": len(persisted),
         "_checkpoint_prefix_digest": native_maintenance._prefix_digest(persisted)},
    )
    active = _agent(home, monkeypatch, db)
    try:
        rows = db.get_messages_as_conversation("sol-cycle")
        assert _recover_stale_codex_reasoning(active, TurnRetryState(), rows)
        assert db.get_session_model_config_value("sol-cycle", "sol_native_replay_disabled") is True
    finally:
        active.close()
    fresh = _agent(home, monkeypatch, db)
    try:
        assert fresh._codex_reasoning_replay_enabled is False
        assert not native_maintenance.eligible(fresh)
        assert all(i.get("type") != "reasoning" for i in build_api_kwargs(fresh, [
            {"role": "system", "content": "Frozen"}] + db.get_messages_as_conversation("sol-cycle"))["input"])
        assert any("rejected-sol-checkpoint" in str(m.get("codex_reasoning_items"))
                   for m in db.get_messages_as_conversation("sol-cycle"))
    finally:
        fresh.close()
        db.close()


def test_pending_tool_and_blocked_keepalive_cannot_commit(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "session.db")
    db.create_session("sol-cycle", source="cli", model="gpt-6-astra")
    db.append_messages_batch("sol-cycle", _messages())
    agent = _agent(tmp_path / "profile", monkeypatch, db)
    try:
        rows = db.get_messages_as_conversation("sol-cycle", include_row_ids=True)
        pending = rows + [{"role": "assistant", "content": "", "tool_calls": [{
            "id": "unanswered", "function": {"name": "read_only", "arguments": "{}"}}]}]
        assert not native_maintenance.attempt(agent, pending, "Frozen", 100_000, phase="threshold")
        assert agent._native_maintenance_abort_fallback is True
        from agent import conversation_compression
        summaries = Mock()
        monkeypatch.setattr(conversation_compression, "compress_context", summaries)
        unchanged, _ = agent._compress_context(
            pending, "Frozen", approx_tokens=agent.context_compressor.threshold_tokens)
        assert unchanged is pending
        summaries.assert_not_called()
        agent.compression_aux_native = False
        unchanged, _ = agent._compress_context(
            pending, "Frozen", approx_tokens=agent.context_compressor.threshold_tokens)
        assert unchanged is pending
        summaries.assert_not_called()
        agent.compression_aux_native = True

        stopped = threading.Event()
        class Stream:
            def __iter__(self):
                while not stopped.wait(0.002):
                    yield {"type": "response.in_progress"}
            def close(self):
                stopped.set()
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: Stream()))
        monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
        monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **kw: None)
        monkeypatch.setattr(agent, "_abort_request_openai_client", lambda c, **kw: stopped.set())
        monkeypatch.setattr(agent, "_resolved_api_call_timeout", lambda: 0.05)
        started = time.monotonic()
        assert not native_maintenance.attempt(
            agent, rows, "Frozen", 100_000, phase="threshold",
            expected_watermark=db.get_active_message_watermark("sol-cycle"),
        )
        assert time.monotonic() - started < 1.0
        assert stopped.is_set()
        assert agent._native_maintenance_abort_fallback is False
        assert not any(m.get("codex_reasoning_items", [{}])[-1].get("type") == "compaction"
                       for m in db.get_messages_as_conversation("sol-cycle"))
    finally:
        agent.close()
        db.close()


def test_stale_commit_suppresses_summary_fallback(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "session.db")
    db.create_session("sol-cycle", source="cli", model="gpt-6-astra")
    db.append_messages_batch("sol-cycle", _messages())
    agent = _agent(tmp_path / "profile", monkeypatch, db)
    class Stream:
        def __iter__(self):
            db.append_message("sol-cycle", "user", "New foreground input")
            return _events()
        def close(self):
            pass
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: Stream()))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **kw: None)
    from agent import conversation_compression
    summaries = Mock()
    monkeypatch.setattr(conversation_compression, "compress_context", summaries)
    try:
        rows = db.get_messages_as_conversation("sol-cycle", include_row_ids=True)
        unchanged, _ = agent._compress_context(
            rows, "Frozen", approx_tokens=agent.context_compressor.threshold_tokens)
        assert unchanged is rows
        summaries.assert_not_called()
        assert agent._native_maintenance_abort_fallback is True
        assert not any(i.get("type") == "compaction" for m in db.get_messages_as_conversation("sol-cycle")
                       for i in m.get("codex_reasoning_items", []))
    finally:
        agent.close()
        db.close()


def test_native_response_without_checkpoint_uses_ordinary_summary(tmp_path, monkeypatch):
    agent = _agent(tmp_path / "profile", monkeypatch)
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: iter([
        {"type": "response.completed", "response": {"status": "completed", "output": []}}
    ])))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **kw: None)
    from agent import conversation_compression
    summaries = Mock(return_value=([{"role": "assistant", "content": "ordinary summary"}], "Frozen"))
    monkeypatch.setattr(conversation_compression, "compress_context", summaries)
    try:
        rows = [{"role": "user", "content": "Fact"}, {"role": "assistant", "content": "Acknowledged"}]
        compressed, _ = agent._compress_context(
            rows, "Frozen", approx_tokens=agent.context_compressor.threshold_tokens)
        assert compressed[0]["content"] == "ordinary summary"
        summaries.assert_called_once()
    finally:
        agent.close()
