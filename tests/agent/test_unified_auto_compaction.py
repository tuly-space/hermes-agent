"""Automatic admission through the real AIAgent turn and SessionDB boundary."""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent


def _reply():
    message = SimpleNamespace(content="done", reasoning_content=None, reasoning=None, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)


@pytest.fixture
def agent(monkeypatch, tmp_path):
    from hermes_cli import config
    settings = {"compression": {"enabled": True, "threshold": 0.5, "hygiene_hard_message_limit": 5,
                                "idle_compact_after_seconds": 1}, "sessions": {}, "bedrock": {}}
    monkeypatch.setattr(config, "load_config_readonly", lambda: settings)
    monkeypatch.setattr(config, "load_config", lambda: settings)
    db = SessionDB(db_path=tmp_path / "state.db")
    with patch("model_tools.get_tool_definitions", return_value=[]), patch(
        "model_tools.check_toolset_requirements", return_value={}
    ), patch("agent.process_bootstrap.OpenAI"):
        instance = AIAgent(api_key="dummy", base_url="https://openrouter.ai/api/v1", model="test/model",
                           session_db=db, session_id="unified", quiet_mode=True, skip_memory=True,
                           skip_context_files=True, enabled_toolsets=[])
    instance.client.chat.completions.create.return_value = _reply()
    instance._cached_system_prompt = "Stable prompt"
    instance._use_prompt_caching = False
    instance._disable_streaming = True
    instance.save_trajectories = False
    instance.context_compressor.threshold_tokens = 100
    instance.context_compressor.note_usage_less_response()
    yield instance, db
    instance.close()
    db.close()


def test_combined_idle_tokens_and_count_one_chain(agent, monkeypatch, caplog):
    instance, db = agent
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "long " * 120,
                "timestamp": 1} for i in range(6)]
    calls = []

    def compress(messages, system_message, **kwargs):
        calls.append(kwargs["approx_tokens"])
        instance._last_compaction_in_place = True
        # A bounded ordinary result, with the current user retained.
        return [{"role": "assistant", "content": "summary"}, messages[-1]], system_message

    monkeypatch.setattr(instance, "_compress_context", compress)
    outcome = instance.run_conversation("next", conversation_history=history)
    assert outcome["final_response"] == "done"
    assert len(calls) == 1
    assert "reasons=idle,tokens,message_count" in caplog.text
    assert db.get_messages_as_conversation("unified")


def test_checkpoint_message_count_is_effective(agent, monkeypatch):
    from agent.codex_responses_adapter import effective_native_responses_message_count
    instance, _ = agent
    instance.api_mode = "codex_responses"
    instance.provider = "openai-codex"
    instance.base_url = "https://chatgpt.com/backend-api/codex"
    instance.model = "gpt-6-astra"
    instance.codex_responses_native_compaction = True
    from agent.native_compaction import resolve_native_compaction_capabilities
    instance.runtime_capabilities = resolve_native_compaction_capabilities(
        model=instance.model, provider=instance.provider, base_url=instance.base_url, is_codex_backend=True)
    history = [{"role": "user", "content": str(i)} if i % 2 == 0 else {"role": "assistant", "content": str(i)}
               for i in range(12)]
    # The checkpoint replay converter is the same one the request sends.
    from agent.native_maintenance import _prefix_digest
    checkpoint = {"type": "compaction", "encrypted_content": "opaque", "_issuer_kind": "codex_backend",
                  "_issuer_model": instance.model, "_checkpoint_count": len(history),
                  "_checkpoint_prefix_digest": _prefix_digest(history)}
    history[-1]["codex_reasoning_items"] = [checkpoint]
    count = effective_native_responses_message_count(instance, history)
    assert count is not None
    assert count < len(history)


@pytest.mark.parametrize("reason", ["idle", "tokens", "message_count", "engine"])
def test_individual_reasons_use_one_assembled_request(agent, monkeypatch, caplog, reason):
    instance, _ = agent
    instance.compression_idle_compact_after_seconds = 1 if reason == "idle" else 0
    instance.compression_hard_message_limit = 5 if reason == "message_count" else 5000
    instance.context_compressor.threshold_tokens = 100 if reason in {"tokens", "idle"} else 50_000
    instance.context_compressor.should_compress_preflight = (
        lambda messages: reason == "engine" and len(messages) > 2
    )
    history = [{"role": "user" if i % 2 == 0 else "assistant",
                "content": "long " * 120 if reason == "tokens" else "small",
                "timestamp": 1} for i in range(6 if reason == "message_count" else 4)]
    calls = []

    def compress(messages, system_message, **kwargs):
        calls.append(list(messages))
        instance._last_compaction_in_place = True
        return [{"role": "assistant", "content": "summary"}, messages[-1]], system_message

    monkeypatch.setattr(instance, "_compress_context", compress)
    outcome = instance.run_conversation("next", conversation_history=history)
    assert outcome["final_response"] == "done"
    assert len(calls) == 1
    assert f"reasons={reason}" in caplog.text


def test_count_blocked_fails_closed_before_provider(agent, monkeypatch):
    instance, db = agent
    instance.compression_idle_compact_after_seconds = 0
    instance.context_compressor.threshold_tokens = 50_000
    instance.context_compressor._automatic_compression_blocked = lambda: True
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
               for i in range(6)]
    result = instance.run_conversation("next", conversation_history=history)
    assert result["final_response"] != "done"
    instance.client.chat.completions.create.assert_not_called()
    assert db.get_messages_as_conversation("unified")


def test_native_replay_checkpoint_avoids_archived_count_and_second_compression(agent, monkeypatch):
    from agent.native_maintenance import _prefix_digest
    from agent.native_compaction import resolve_native_compaction_capabilities
    from agent.codex_responses_adapter import has_replayable_native_compaction_checkpoint
    instance, _ = agent
    instance.api_mode = "codex_responses"
    instance.provider = "openai-codex"
    instance.base_url = "https://chatgpt.com/backend-api/codex"
    instance.model = "gpt-6-astra"
    instance.codex_responses_native_compaction = True
    instance.runtime_capabilities = resolve_native_compaction_capabilities(
        model=instance.model, provider=instance.provider, base_url=instance.base_url, is_codex_backend=True)
    instance.compression_idle_compact_after_seconds = 0
    instance.context_compressor.threshold_tokens = 50_000
    instance.compression_hard_message_limit = 10
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
               for i in range(12)]
    history[-1]["codex_reasoning_items"] = [{
        "type": "compaction", "encrypted_content": "opaque", "_issuer_kind": "codex_backend",
        "_issuer_model": instance.model, "_checkpoint_count": len(history),
        "_checkpoint_prefix_digest": _prefix_digest(history),
    }]
    assert has_replayable_native_compaction_checkpoint(instance, history)
    instance.context_compressor.awaiting_real_usage_after_compression = True
    instance._compress_context = lambda *a, **k: pytest.fail("duplicate compaction before usage")
    # Simulate the Responses transport only; the assembled request and preflight
    # still run through AIAgent and the real temporary SessionDB.
    instance._create_request_openai_client = lambda **_kwargs: instance.client
    instance._close_request_openai_client = lambda *_args, **_kwargs: None
    instance.client.responses.create.return_value = iter([
        {"type": "response.created", "response": {"id": "fixture", "status": "in_progress"}},
        {"type": "response.output_text.delta", "delta": "done"},
        {"type": "response.completed", "response": {"status": "completed", "output": [],
            "usage": {"input_tokens": 42, "output_tokens": 2}}},
    ])
    result = instance.run_conversation("next", conversation_history=history)
    assert result["final_response"] == "done"


def test_large_tool_result_compacts_only_at_next_request(agent, monkeypatch, caplog):
    instance, db = agent
    instance.compression_idle_compact_after_seconds = 0
    instance.compression_hard_message_limit = 5000
    instance.context_compressor.threshold_tokens = 100
    instance.tools = [{"type": "function", "function": {"name": "web_search",
        "description": "test", "parameters": {"type": "object", "properties": {}}}}]
    instance.tool_delay = 0
    instance.client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None, reasoning_content=None, reasoning=None, tool_calls=[
                SimpleNamespace(id="call_1", type="function", function=SimpleNamespace(
                    name="web_search", arguments='{"query":"x"}'))]),
            finish_reason="tool_calls")], usage=None),
        _reply(),
    ]
    instance.context_compressor.prune_tool_results_only = lambda msgs, **kw: (msgs, 0)
    compress_calls = []

    def compress(messages, system, **kwargs):
        compress_calls.append(list(messages))
        instance._last_compaction_in_place = True
        return [{"role": "assistant", "content": "summary"}, messages[-1]], system

    monkeypatch.setattr(instance, "_compress_context", compress)
    with patch("model_tools.handle_function_call", return_value="x" * 12_000):
        result = instance.run_conversation("fetch")
    assert result["final_response"] == "done"
    assert len(compress_calls) == 1
    assert any(m.get("role") == "tool" for m in compress_calls[0])
    assert "reasons=tokens" in caplog.text
    assert db.get_messages_as_conversation("unified")


@pytest.mark.parametrize("native_result", ["checkpoint", "missing", "cancelled"])
def test_assembled_gate_runs_one_native_chain(agent, monkeypatch, native_result):
    """Real gate/facade/native commit with fake HTTP, not a mocked _compress_context."""
    from agent import conversation_compression
    from agent.native_compaction import resolve_native_compaction_capabilities
    from agent.codex_responses_adapter import has_replayable_native_compaction_checkpoint
    instance, db = agent
    instance.api_mode = "codex_responses"
    instance.provider = "openai-codex"
    instance.base_url = "https://chatgpt.com/backend-api/codex"
    instance.model = "gpt-6-astra"
    instance.compression_aux_native = True
    instance.compression_aux_provider = "openai-codex"
    instance.compression_aux_model = "gpt-6-sol"
    instance.compression_aux_base_url = None
    instance.runtime_capabilities = resolve_native_compaction_capabilities(
        model=instance.model, provider=instance.provider, base_url=instance.base_url,
        is_codex_backend=True)
    instance.compression_idle_compact_after_seconds = 0
    instance.context_compressor.threshold_tokens = 50_000
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
               for i in range(6)]
    db.create_session("unified", "cli", model=instance.model)
    db.append_messages_batch("unified", history)
    from agent import title_generator
    monkeypatch.setattr(title_generator, "_auto_title_enabled", lambda: False)
    calls, summaries, statuses = [], [], []
    instance.status_callback = lambda _event, text: statuses.append(text)

    def create(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "gpt-6-sol":
            if native_result == "cancelled":
                instance.hard_interrupt()
            events = [] if native_result == "missing" else [{
                "type": "response.output_item.done", "item": {
                    "type": "compaction", "encrypted_content": "test-checkpoint"}}]
            return iter(events + [{"type": "response.completed", "response": {
                "status": "completed", "output": []}}])
        return iter([
            {"type": "response.output_text.delta", "delta": "done"},
            {"type": "response.completed", "response": {"status": "completed", "output": [],
                "usage": {"input_tokens": 42, "output_tokens": 2}}},
        ])

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    instance._create_request_openai_client = lambda **_kwargs: client
    instance._close_request_openai_client = lambda *_args, **_kwargs: None
    instance._abort_request_openai_client = lambda *_args, **_kwargs: None

    def summary(_agent, messages, system, **_kwargs):
        summaries.append(messages)
        instance._last_compaction_in_place = True
        return [{"role": "assistant", "content": "summary"}, messages[-1]], system

    monkeypatch.setattr(conversation_compression, "compress_context", summary)
    result = instance.run_conversation("next", conversation_history=history)
    from agent.auxiliary_client import _effective_aux_timeout
    from agent.conversation_compression import resolve_context_compression_timeouts
    assert calls[0]["timeout"] == _effective_aux_timeout("compression", None)
    assert resolve_context_compression_timeouts({"context_timeout_seconds": 2,
                                                 "context_total_ceiling_seconds": 3})[0] >= calls[0]["timeout"]
    assert sum(c["model"] == "gpt-6-sol" for c in calls) == 1
    assert len(summaries) == (1 if native_result == "missing" else 0)
    assert len([s for s in statuses if "compression:" in s]) == 1
    assert not any("127 tokens >=" in s for s in statuses)
    saved = db.get_messages_as_conversation("unified")
    assert has_replayable_native_compaction_checkpoint(instance, saved) is (native_result == "checkpoint")
    if native_result == "cancelled":
        assert result.get("interrupted") or result.get("compression_deferred")
        assert len(calls) == 1
    else:
        assert result["final_response"] == "done"
        assert len(calls) == 2
        assert "context_management" not in calls[-1]


def _native_fixture(agent, monkeypatch):
    from agent.native_compaction import resolve_native_compaction_capabilities
    instance, db = agent
    instance.api_mode = "codex_responses"
    instance.provider = "openai-codex"
    instance.base_url = "https://chatgpt.com/backend-api/codex"
    instance.model = "gpt-6-astra"
    instance.compression_aux_native = True
    instance.compression_aux_provider = "openai-codex"
    instance.compression_aux_model = "gpt-6-sol"
    instance.runtime_capabilities = resolve_native_compaction_capabilities(
        model=instance.model, provider=instance.provider, base_url=instance.base_url,
        is_codex_backend=True,
    )
    instance.compression_idle_compact_after_seconds = 0
    instance.context_compressor.threshold_tokens = 50_000
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
               for i in range(6)]
    db.create_session("unified", "cli", model=instance.model)
    db.append_messages_batch("unified", history)
    from agent import title_generator
    monkeypatch.setattr(title_generator, "_auto_title_enabled", lambda: False)
    instance._create_request_openai_client = lambda **_kwargs: instance.client
    instance._close_request_openai_client = lambda *_args, **_kwargs: None
    instance._abort_request_openai_client = lambda *_args, **_kwargs: None
    return instance, db, history


def _checkpoint_stream():
    return iter([
        {"type": "response.output_item.done", "item": {
            "type": "compaction", "encrypted_content": "checkpoint"}},
        {"type": "response.completed", "response": {"status": "completed", "output": []}},
    ])


@pytest.mark.parametrize("failure", ["cas_false", "cas_exception", "stale_prefix"])
def test_native_refused_admission_never_sends_or_summarizes_stale_input(agent, monkeypatch, failure):
    from agent import conversation_compression
    from agent.codex_responses_adapter import has_replayable_native_compaction_checkpoint
    instance, db, history = _native_fixture(agent, monkeypatch)
    native_calls = []
    instance.client.responses.create.side_effect = lambda **kw: native_calls.append(kw) or _checkpoint_stream()
    if failure == "cas_false":
        monkeypatch.setattr(db, "attach_native_checkpoint", lambda *a, **kw: False)
    elif failure == "cas_exception":
        def conflict(*a, **kw):
            raise RuntimeError("lease lost")
        monkeypatch.setattr(db, "attach_native_checkpoint", conflict)
    else:
        history[0]["content"] = "different from durable prefix"
    monkeypatch.setattr(conversation_compression, "compress_context",
                        lambda *a, **kw: pytest.fail("stale fallback summary"))
    result = instance.run_conversation("next", conversation_history=history)
    assert result.get("compression_deferred")
    assert result["compression_deferred_reason"] == {
        "cas_false": "stale_transcript_or_lease",
        "cas_exception": "checkpoint_commit_error",
        "stale_prefix": "transcript_content_mismatch",
    }[failure]
    assert len(native_calls) == (0 if failure == "stale_prefix" else 1)
    assert not has_replayable_native_compaction_checkpoint(instance, db.get_messages_as_conversation("unified"))


@pytest.mark.parametrize("abort", ["timeout", "interrupt", "disabled_watchdog"])
def test_native_wait_cancel_rejects_late_checkpoint_on_real_turn(agent, monkeypatch, abort):
    from agent import conversation_compression
    from agent.codex_responses_adapter import has_replayable_native_compaction_checkpoint
    instance, db, history = _native_fixture(agent, monkeypatch)
    if abort == "disabled_watchdog":
        from agent import auxiliary_client
        monkeypatch.setattr(auxiliary_client, "_effective_aux_timeout", lambda *_args: 0.15)
        monkeypatch.setattr(conversation_compression, "resolve_context_compression_timeouts",
                            lambda: (0, 0.2))
    else:
        monkeypatch.setattr(conversation_compression, "resolve_context_compression_timeouts",
                            lambda: (0.15, 0.2))
    # Real preflight/facade/native request; only the network stream is held.
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def blocked_stream(**kwargs):
        calls.append(kwargs)
        def events():
            started.set()
            try:
                release.wait(3)
                yield from _checkpoint_stream()
            finally:
                finished.set()
        return events()

    instance.client.responses.create.side_effect = blocked_stream
    monkeypatch.setattr(conversation_compression, "compress_context",
                        lambda *a, **kw: pytest.fail("fallback after native cancellation"))
    canceller = None
    if abort == "interrupt":
        def cancel_after_start():
            if started.wait(3):
                instance.hard_interrupt()
        canceller = threading.Thread(target=cancel_after_start)
        canceller.start()
    try:
        result = instance.run_conversation("next", conversation_history=history)
        assert started.is_set() and len(calls) == 1
        assert result.get("compression_deferred") or result.get("interrupted") or result.get("failed")
        assert not has_replayable_native_compaction_checkpoint(instance, db.get_messages_as_conversation("unified"))
    finally:
        release.set()
        if canceller is not None:
            canceller.join(3)
    assert finished.wait(3)
    assert not has_replayable_native_compaction_checkpoint(instance, db.get_messages_as_conversation("unified"))
    assert not any(m.get("codex_reasoning_items") for m in history)
    assert not any(m.get("codex_reasoning_items") for m in (instance._session_messages or []))


def test_review_fork_first_request_over_count_still_reaches_provider(agent):
    instance, _ = agent
    instance._review_defer_compaction_before_first_response = True
    instance.compression_idle_compact_after_seconds = 0
    instance.context_compressor.threshold_tokens = 50_000
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
               for i in range(6)]
    instance._compress_context = lambda *a, **kw: pytest.fail("first review request compacted")
    result = instance.run_conversation("next", conversation_history=history)
    assert result["final_response"] == "done"
    instance.client.chat.completions.create.assert_called_once()


def test_ordinary_summary_idle_latch_and_remeasured_token_retry(agent, monkeypatch):
    instance, _ = agent
    instance.compression_idle_compact_after_seconds = 1
    instance.compression_hard_message_limit = 5000
    instance.context_compressor.threshold_tokens = 100
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "long " * 120,
                "timestamp": 1} for i in range(6)]
    calls = []

    def summary(messages, system, **kwargs):
        calls.append(kwargs["approx_tokens"])
        instance._last_compaction_in_place = True
        instance.context_compressor.awaiting_real_usage_after_compression = True
        if len(calls) == 1:
            # Still above threshold but smaller: permit the measured ordinary retry.
            return [{"role": "assistant", "content": "long " * 160}, messages[-1]], system
        return [{"role": "assistant", "content": "short"}, messages[-1]], system

    monkeypatch.setattr(instance, "_compress_context", summary)
    result = instance.run_conversation("next", conversation_history=history)
    assert result["final_response"] == "done"
    assert len(calls) == 2
    assert calls[1] < calls[0]


def test_native_timeout_before_request_does_not_send_queued_stale_work(agent, monkeypatch):
    from agent import conversation_compression
    instance, db, history = _native_fixture(agent, monkeypatch)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(conversation_compression, "resolve_context_compression_timeouts",
                        lambda: (0.12, 0.15))
    transport = instance._get_transport()
    original = transport.preflight_kwargs

    def slow_preflight(*args, **kwargs):
        entered.set()
        try:
            release.wait(3)
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(transport, "preflight_kwargs", slow_preflight)
    try:
        result = instance.run_conversation("next", conversation_history=history)
        assert entered.is_set()
        assert result.get("failed") or result.get("compression_deferred")
    finally:
        release.set()
    assert finished.wait(3)
    instance.client.responses.create.assert_not_called()
    assert not any(m.get("codex_reasoning_items") for m in db.get_messages_as_conversation("unified"))


def test_idle_does_not_repeat_ordinary_summary_awaiting_usage(agent):
    instance, _ = agent
    instance.compression_hard_message_limit = 5000
    instance.compression_idle_compact_after_seconds = 1
    instance.context_compressor.threshold_tokens = 100
    instance.context_compressor.last_compression_rough_tokens = 0
    instance.context_compressor.awaiting_real_usage_after_compression = True
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "small",
                "timestamp": 1} for i in range(4)]
    instance._compress_context = lambda *a, **kw: pytest.fail("idle retry before provider usage")
    result = instance.run_conversation("next", conversation_history=history)
    assert result["final_response"] == "done"
    instance.client.chat.completions.create.assert_called_once()

