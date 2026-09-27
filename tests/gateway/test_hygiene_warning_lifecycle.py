"""Gateway cold resumes use AIAgent's durable compression cooldown and warning sink."""
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent
from tests.gateway.test_session_hygiene import _make_cooldown_runner, _make_history


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False, True])
async def test_failed_preflight_retains_history_and_cooldown_across_fresh_agents(
    tmp_path, monkeypatch, setting,
):
    from hermes_cli import config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    settings = {"compression": {"enabled": True, "hygiene_hard_message_limit": 5},
                "sessions": {}, "bedrock": {}}
    if setting is not None:
        settings["display"] = {"suppress_warning_notifications": setting}
    monkeypatch.setattr(config, "load_config_readonly", lambda: settings)
    monkeypatch.setattr(config, "load_config", lambda: settings)
    def write_config():
        policy = ("" if setting is None else
                  f"display: {{suppress_warning_notifications: {str(setting).lower()}}}\n")
        (tmp_path / "config.yaml").write_text(
            policy + "compression: {enabled: true, hygiene_hard_message_limit: 5}\n"
        )

    write_config()
    sid = "hygiene-policy"
    history = _make_history(6, content_size=40)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(sid, "telegram")
    db.append_messages_batch(sid, history)
    real_module = importlib.import_module("run_agent")
    try:
        attempts, notices, provider_calls = [], [], []
        recovering = False

        async def real_turn(message, context_prompt, history, source, session_id, **_kwargs):
            # The fixture's fake run_agent module belongs to the retired detached worker.
            monkeypatch.setitem(sys.modules, "run_agent", real_module)
            with patch("model_tools.get_tool_definitions", return_value=[]), patch(
                "model_tools.check_toolset_requirements", return_value={}
            ), patch("agent.process_bootstrap.OpenAI"):
                agent = AIAgent(api_key="dummy", base_url="https://openrouter.ai/api/v1",
                                model="test/model", session_db=db, session_id=session_id,
                                quiet_mode=True, skip_memory=True, skip_context_files=True,
                                enabled_toolsets=[])
            agent._cached_system_prompt = "stable"
            agent._disable_streaming = True
            agent._use_prompt_caching = False
            agent.save_trajectories = False
            agent.compression_idle_compact_after_seconds = 0
            agent.context_compressor.threshold_tokens = 50_000
            agent.status_callback = lambda kind, text: notices.append((kind, text))
            agent.client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content="ok", reasoning_content=None, reasoning=None, tool_calls=None),
                    finish_reason="stop")], usage=None)

            def compressor_edge(messages, **_kwargs):
                attempts.append(session_id)
                if not recovering:
                    agent.context_compressor._last_compress_aborted = True
                    agent.context_compressor._last_summary_error = "fixture auxiliary failure"
                    agent.context_compressor._record_compression_failure_cooldown(
                        300, "fixture auxiliary failure"
                    )
                    return messages
                agent.context_compressor._last_compress_aborted = False
                agent.context_compressor._last_summary_error = None
                agent.context_compressor._last_aux_model_failure_model = "fixture-aux"
                agent.context_compressor._last_aux_model_failure_error = "fixture auxiliary failure"
                return [{"role": "assistant", "content": "retained summary"}, messages[-1]]

            agent.context_compressor.compress = compressor_edge
            try:
                result = agent.run_conversation(message, conversation_history=history)
                provider_calls.append(agent.client.chat.completions.create.call_count)
                return {**result, "tools": [], "history_offset": 0,
                        "last_prompt_tokens": 0}
            finally:
                agent.close()

        for iteration in range(2):
            runner, adapter, event = _make_cooldown_runner(
                monkeypatch, tmp_path, AIAgent, db, sid
            )
            monkeypatch.setitem(sys.modules, "run_agent", real_module)
            runner.session_store.load_transcript.return_value = history
            runner._run_agent = real_turn
            # The helper writes legacy config; the actual agent reads the explicit
            # settings above and warning transport reads this file.
            write_config()
            await runner._handle_message(event)
            assert runner.session_store.rewrite_transcript.call_count == 0
            assert attempts == [sid]
            cooldown = db.get_compression_failure_cooldown(sid)
            assert cooldown and cooldown["remaining_seconds"] > 0
            assert any(row["content"] == history[0]["content"] for row in db.get_messages(sid))
            if iteration == 0:
                failure = [text for kind, text in notices if kind == "warn" and "Compression aborted" in text]
                assert len(failure) == 1
                # Exercise the same adapter policy sink as the gateway's status delivery.
                await adapter.emit_warning(event.source.chat_id, failure[0],
                                           logical_platform=event.source.platform)
                assert len(adapter.sent) == (0 if setting is True else 1)
            assert provider_calls[-1] == 0  # over-count failure cannot send unbounded input
        assert len([n for kind, n in notices if kind == "warn" and "Compression aborted" in n]) == 1

        db.clear_compression_failure_cooldown(sid)
        recovering = True
        runner, adapter, event = _make_cooldown_runner(monkeypatch, tmp_path, AIAgent, db, sid)
        monkeypatch.setitem(sys.modules, "run_agent", real_module)
        runner.session_store.load_transcript.return_value = history
        runner._run_agent = real_turn
        write_config()
        assert await runner._handle_message(event) == "ok"
        assert attempts == [sid, sid]
        assert db.get_compression_failure_cooldown(sid) is None
        assert any(row["content"] == "retained summary" for row in db.get_messages(sid))
        aux = [text for kind, text in notices if kind == "warn" and "Configured compression model" in text]
        assert len(aux) == 1
        await adapter.emit_warning(event.source.chat_id, aux[0], logical_platform=event.source.platform)
        assert len(adapter.sent) == (0 if setting is True else 1)
        assert provider_calls[-1] == 1
        runner.session_store.rewrite_transcript.assert_not_called()
    finally:
        db.close()
