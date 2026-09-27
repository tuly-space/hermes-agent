"""Gateway passes the complete transcript to the assembled AIAgent gate."""
import importlib
import sys
from unittest.mock import patch

import pytest

from hermes_state import SessionDB
from tests.gateway.test_session_hygiene import _make_cooldown_runner, _make_history


@pytest.mark.asyncio
async def test_gateway_passes_full_history_then_agent_compacts_once(monkeypatch, tmp_path, caplog):
    real_module = importlib.import_module("run_agent")
    from run_agent import AIAgent
    from hermes_cli import config

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("gateway-unified", "telegram")
    runner, _adapter, event = _make_cooldown_runner(
        monkeypatch, tmp_path, AIAgent, db, "gateway-unified")
    # The legacy fixture substitutes an AIAgent-only module for its now-retired
    # detached worker. Keep the actual module for the real turn loop's imports.
    monkeypatch.setitem(sys.modules, "run_agent", real_module)
    history = _make_history(8, content_size=40)
    runner.session_store.load_transcript.return_value = history
    (tmp_path / "config.yaml").write_text(
        "compression:\n  enabled: true\n  hygiene_hard_message_limit: 5\n")
    settings = {"compression": {"enabled": True, "hygiene_hard_message_limit": 5},
                "sessions": {}, "bedrock": {}}
    monkeypatch.setattr(config, "load_config_readonly", lambda: settings)
    monkeypatch.setattr(config, "load_config", lambda: settings)
    calls = []
    histories = []

    async def real_turn(message, context_prompt, history, source, session_id, **_kwargs):
        histories.append(history)
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
        from types import SimpleNamespace
        agent.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="ok", reasoning_content=None, reasoning=None, tool_calls=None),
                finish_reason="stop")], usage=None)

        def compress(messages, system, **_kwargs):
            calls.append(list(messages))
            agent._last_compaction_in_place = True
            return [{"role": "assistant", "content": "summary"}, messages[-1]], system

        agent._compress_context = compress
        try:
            result = agent.run_conversation(message, conversation_history=history)
            return {**result, "tools": [], "history_offset": 0, "last_prompt_tokens": 0}
        finally:
            agent.close()

    runner._run_agent = real_turn
    try:
        assert await runner._handle_message(event) == "ok"
        assert histories and len(histories[0]) == len(history)
        assert len(calls) == 1
        assert "reasons=message_count" in caplog.text
        runner.session_store.rewrite_transcript.assert_not_called()
        assert db.get_messages_as_conversation("gateway-unified")
    finally:
        db.close()
