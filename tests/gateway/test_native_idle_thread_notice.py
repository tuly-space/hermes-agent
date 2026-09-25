"""The durable idle timer notifies only the originating Discord thread via its adapter."""
import asyncio
import threading
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

from agent import native_maintenance as maintenance
from gateway.config import Platform, PlatformConfig
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from hermes_state import SessionDB
from plugins.platforms.discord.adapter import DiscordAdapter
from run_agent import AIAgent


def _adapter(thread_id):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    sent = asyncio.Event()
    async def send(**kw):
        sent.set()
        return NS(id=f"message-in-{thread_id}")
    channel = NS(send=AsyncMock(side_effect=send))
    resolved = []

    async def resolve(target):
        resolved.append(target)
        return channel if target == thread_id else None

    adapter._client = object()  # adapter-level fake transport; no Discord connection
    adapter._resolve_channel = resolve
    return adapter, sent, channel, resolved


def _wire(agent, adapter, loop, *, thread_id="thread-1", platform=Platform.DISCORD,
          muted=False):
    source = SessionSource(platform=platform, chat_id="parent", chat_type="thread",
                           thread_id=thread_id)
    ctx = TurnContext(source=source, session_key=None, user_config={},
                      _loop_for_step=loop, _status_adapter=adapter, _status_chat_id="parent",
                      _status_thread_metadata={"thread_id": thread_id},
                      _hooks_ref=NS(loaded_hooks=[]), mute_notification_reply=muted)
    runner = NS(_service_tier=None, _consume_pending_turn_sidecar_notes=lambda _: [])
    turn = TurnRunner(runner, ctx)
    turn._attach_session_title_callback = lambda *args: None
    turn._make_bg_review_callbacks = lambda: (lambda _: None, lambda: None)
    turn._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
    return agent._native_idle_start_callback


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("idle-thread", source="discord", model="gpt-5.6")
    db.append_messages_batch("idle-thread", [
        {"role": "user", "content": "Saved fact"},
        {"role": "assistant", "content": "Acknowledged"},
    ])
    agent = AIAgent(api_key="test-key", base_url="https://api.openai.com/v1",
                    api_mode="codex_responses", model="gpt-5.6", provider="openai-api",
                    session_db=db, session_id="idle-thread", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    agent.codex_responses_native_compaction = agent.compression_native_first = True
    agent.compression_native_idle_after_seconds = 1
    agent.compression_native_idle_min_tokens = 80_000
    agent.compression_in_place = True
    agent._cached_system_prompt = "Frozen"
    monkeypatch.setattr(maintenance, "_idle_pressure", lambda *args: 80_000)
    from agent import periodic_scheduler
    timers = []
    monkeypatch.setattr(periodic_scheduler, "schedule", lambda fn, seconds: timers.append(fn) or Mock())
    yield agent, db, timers
    agent.close()
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome,notices,fallbacks", [
    ("native", 1, 0),
    ("failed_response", 1, 1),
    ("failed_request", 1, 1),
    ("pre_request", 1, 1),
    ("cancel_before_request", 0, 0),
    ("below_floor", 0, 0),
    ("stale_generation", 0, 0),
])
async def test_real_idle_timer_routes_one_start_to_thread_even_after_turn_done(
        prepared, monkeypatch, outcome, notices, fallbacks):
    agent, db, timers = prepared
    adapter, sent, channel, resolved = _adapter("thread-1")
    callback = _wire(agent, adapter, asyncio.get_running_loop())
    assert callable(callback)
    client = NS(responses=NS(create=Mock()))
    if outcome == "native":
        events = [{"type": "response.output_item.done", "item": {
            "type": "compaction", "encrypted_content": "sealed"}}]
    else:
        events = []
    events.append({"type": "response.completed", "response": {
        "status": "completed", "output": [], "usage": {"input_tokens": 81_000}}})
    client.responses.create.return_value = iter(events)
    if outcome == "failed_request":
        client.responses.create.side_effect = RuntimeError("provider unavailable")
    def create(**kwargs):
        if outcome == "pre_request":
            raise RuntimeError("cannot open client")
        if outcome == "cancel_before_request":
            agent._native_idle_cancel.set()
        return client
    monkeypatch.setattr(agent, "_create_request_openai_client", create)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **kw: None)
    summary = Mock(side_effect=RuntimeError("summary unavailable"))
    agent._compress_context = summary
    maintenance.arm_idle(agent, {"completed": True})
    assert len(timers) == 1
    # Completed turn has gone away; status callbacks are generation-scoped and not usable here.
    if outcome == "below_floor":
        monkeypatch.setattr(maintenance, "_idle_pressure", lambda *args: 79_999)
    if outcome == "stale_generation":
        maintenance.cancel_idle(agent)
    await asyncio.to_thread(timers[0])
    if notices:
        await asyncio.wait_for(sent.wait(), timeout=2)
    await asyncio.sleep(0)
    assert channel.send.await_count == notices
    assert resolved == (["thread-1"] if notices else [])
    assert summary.call_count == fallbacks
    if notices:
        text = channel.send.await_args.kwargs["content"]
        assert "正在后台压缩" in text and "原始记录会保留" in text
        assert "成功" not in text
        assert adapter._last_self_message_id.get("thread-1") is None
    assert [row["content"] for row in db.get_messages_as_conversation("idle-thread")] == [
        "Saved fact", "Acknowledged"]
    if outcome == "cancel_before_request":
        client.responses.create.assert_not_called()


@pytest.mark.asyncio
async def test_reused_agent_rebinds_exact_origin_and_closed_loop_is_safe(prepared):
    agent, _, _ = prepared
    loop = asyncio.get_running_loop()
    first, first_sent, first_channel, first_routes = _adapter("thread-1")
    old = _wire(agent, first, loop)
    second, second_sent, second_channel, second_routes = _adapter("thread-2")
    current = _wire(agent, second, loop, thread_id="thread-2")
    assert old is not current
    await asyncio.to_thread(old)
    await asyncio.to_thread(current)
    await asyncio.wait_for(asyncio.gather(first_sent.wait(), second_sent.wait()), timeout=2)
    assert first_routes == ["thread-1"] and second_routes == ["thread-2"]
    assert first_channel.send.await_count == second_channel.send.await_count == 1
    # Direct adapter routing + non-conversational metadata; never the parent or Home.
    assert first._last_self_message_id.get("thread-1") is None
    no_thread = _wire(agent, second, loop, thread_id="")
    assert no_thread is None
    assert _wire(agent, second, loop, platform=Platform.TELEGRAM) is None
    assert _wire(agent, second, loop, thread_id="thread-2", muted=True) is None
    stopped = asyncio.new_event_loop()
    stopped.close()
    late = _wire(agent, second, stopped, thread_id="thread-2")
    await asyncio.to_thread(late)
    assert second_channel.send.await_count == 1


@pytest.mark.asyncio
async def test_failed_or_slow_send_never_blocks_native_maintenance(prepared, monkeypatch):
    agent, _, timers = prepared
    adapter, _, channel, _ = _adapter("thread-1")
    async def stuck(**kwargs):
        await asyncio.Event().wait()
    channel.send.side_effect = stuck
    _wire(agent, adapter, asyncio.get_running_loop())
    calls = []
    def native(*args, **kwargs):
        kwargs["on_idle_start"]()
        calls.append("native")
        return True
    monkeypatch.setattr(maintenance, "attempt", native)
    maintenance.arm_idle(agent, {"completed": True})
    await asyncio.wait_for(asyncio.to_thread(timers[0]), timeout=2)
    assert calls == ["native"]
    # Cancel this test's pending fake transport; the production send has a 10s cap.
    await asyncio.sleep(0)
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task() and task.get_coro().__name__ == "send_start":
            task.cancel()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_native_failure_real_summary_commit_still_only_one_thread_notice(prepared, monkeypatch):
    agent, db, timers = prepared
    adapter, sent, channel, routes = _adapter("thread-1")
    _wire(agent, adapter, asyncio.get_running_loop())
    client = NS(responses=NS(create=Mock(return_value=iter([{
        "type": "response.completed", "response": {"status": "completed", "output": []}}]))))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **kw: None)
    agent.context_compressor.compress = lambda *a, **kw: [
        {"role": "user", "content": "Summary of saved fact"},
        {"role": "assistant", "content": "Acknowledged summary"},
    ]
    maintenance.arm_idle(agent, {"completed": True})
    await asyncio.to_thread(timers[0])
    await asyncio.wait_for(sent.wait(), timeout=2)
    await asyncio.sleep(0)
    assert channel.send.await_count == 1 and routes == ["thread-1"]
    assert [row["content"] for row in db.get_messages_as_conversation("idle-thread")] == [
        "Summary of saved fact", "Acknowledged summary"]
    assert len(db.get_messages("idle-thread", include_inactive=True)) > 2  # raw rows retained


def test_callback_failure_does_not_block_native_or_repeat_on_summary(prepared, monkeypatch):
    agent, _, timers = prepared
    notice = Mock(side_effect=RuntimeError("adapter unavailable"))
    agent._native_idle_start_callback = notice
    def native(*args, **kwargs):
        kwargs["on_idle_start"]()
        return False
    monkeypatch.setattr(maintenance, "attempt", native)
    agent._compress_context = Mock(side_effect=RuntimeError("summary unavailable"))
    maintenance.arm_idle(agent, {"completed": True})
    timers[0]()
    notice.assert_called_once()
    agent._compress_context.assert_called_once()


@pytest.mark.asyncio
async def test_deferred_notice_restores_origin_profile_a_b_a(prepared, monkeypatch, tmp_path):
    from hermes_constants import get_hermes_home
    agent, _, _ = prepared
    loop = asyncio.get_running_loop()
    homes = [tmp_path / "a", tmp_path / "b", tmp_path / "a"]
    callbacks = []
    seen = []
    for index, home in enumerate(homes):
        home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        adapter, _, channel, _ = _adapter(f"thread-{index}")
        async def record(*, content, reference=None, target=index):
            seen.append((target, get_hermes_home()))
            return NS(id=f"message-{target}")
        channel.send.side_effect = record
        callbacks.append(_wire(agent, adapter, loop, thread_id=f"thread-{index}"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient"))
    for callback in callbacks:
        await asyncio.to_thread(callback)
    async def all_arrived():
        while len(seen) < len(callbacks):
            await asyncio.sleep(0.01)
    await asyncio.wait_for(all_arrived(), timeout=2)
    assert sorted(seen) == sorted(enumerate(homes))
    assert get_hermes_home() == tmp_path / "ambient"
