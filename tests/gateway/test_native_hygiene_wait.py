"""Detached Gateway hygiene uses the auxiliary native request's effective wait budget."""

import asyncio
import importlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.run_turn import GatewayTurnMixin
from hermes_state import SessionDB
from run_agent import AIAgent

_CODEX_URL = "https://chatgpt.com/backend-api/codex"


def _events():
    return iter([
        {"type": "response.output_item.done", "item": {
            "type": "compaction", "encrypted_content": "fake-sol-checkpoint"}},
        {"type": "response.completed", "response": {"status": "completed", "output": []}},
    ])


def _setup(tmp_path, monkeypatch, *, timeout=420, native=True, model="gpt-6-astra", profile_home=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    home = profile_home or tmp_path / "profile"
    home.mkdir(parents=True, exist_ok=True)
    if not (home / "config.yaml").exists():
        (home / "config.yaml").write_text(
            "auxiliary:\n  compression:\n    provider: openai-codex\n"
            f"    model: gpt-6-sol\n    native: {str(native).lower()}\n    timeout: {timeout}\n"
        )
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(db_path=tmp_path / "session.db")
    db.create_session("hygiene-sid", source="telegram", model=model)
    db.append_messages_batch("hygiene-sid", [
        {"role": "user", "content": "First fact"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second fact"},
        {"role": "assistant", "content": "Second answer"},
    ])
    history = db.get_messages_as_conversation("hygiene-sid", include_row_ids=True)
    agent = AIAgent(
        api_key="fake-session-key", base_url=_CODEX_URL, api_mode="codex_responses",
        model=model, provider="openai-codex", session_db=db, session_id="hygiene-sid",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
    )
    agent._cached_system_prompt = "Frozen instruction"
    runner = GatewayTurnMixin.__new__(GatewayTurnMixin)
    runner._hmwa_hygiene_build_agent = AsyncMock(return_value=(agent, db))
    runner._hmwa_hygiene_apply_result = AsyncMock()
    runner._cleanup_agent_resources_off_loop = AsyncMock()
    runner._evict_cached_agent = MagicMock()
    runner._defer_agent_cleanup_until_future_done = lambda future, _agent, **kw: None
    runner._hmwa_hygiene_record_failure_cooldown = AsyncMock()
    runner._hmwa_hygiene_stamp = MagicMock()
    runner._hmwa_hygiene_notify = AsyncMock()
    hs = SimpleNamespace(
        timeout_seconds=30.0, total_ceiling_seconds=600.0,
        max_turn_hold_seconds=10.0, failure_cooldown_seconds=-1,
    )
    plan = SimpleNamespace(approx_tokens=100_000, msg_count=len(history), warn_token_threshold=999_999)
    entry = SimpleNamespace(session_id="hygiene-sid")
    attempt = SimpleNamespace(agent=None, meta={}, history=history, cleanup_deferred=False)
    async def run():
        await runner._hmwa_hygiene_detached_attempt(
            attempt, hs, plan, history, history, model, {}, None, entry,
            "telegram:dm:hygiene", None, 1,
        )
    return run, runner, agent, db, history, hs, attempt


def _accelerate_clock(monkeypatch, factor=100):
    """The host and fence see accelerated elapsed time; asyncio still schedules normally."""
    real_monotonic = time.monotonic
    offset = [0.0]
    clock = SimpleNamespace(
        monotonic=lambda: real_monotonic() * factor + offset[0],
        advance=lambda seconds: offset.__setitem__(0, offset[0] + seconds),
    )
    monkeypatch.setattr(importlib.import_module("gateway.run_turn"), "time", clock)
    monkeypatch.setattr(importlib.import_module("agent.conversation_compression"), "time", clock)
    return clock


@pytest.mark.asyncio
async def test_native_detached_wait_accepts_34_seconds_without_tokens(tmp_path, monkeypatch):
    run, runner, agent, db, history, hs, attempt = _setup(tmp_path, monkeypatch)
    _accelerate_clock(monkeypatch)
    calls = []
    def provider(**kwargs):
        calls.append(kwargs)
        time.sleep(0.34)  # 34 simulated seconds; no token/progress callbacks.
        return _events()
    client = SimpleNamespace(responses=SimpleNamespace(create=provider))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *args, **kw: None)
    try:
        await asyncio.wait_for(run(), 3)
        assert len(calls) == 1 and calls[0]["timeout"] == 420
        assert calls[0]["model"] == "gpt-6-sol"
        assert agent._native_maintenance_committed
        assert not attempt.commit_fence.is_cancelled
        assert hs.timeout_seconds == hs.total_ceiling_seconds == hs.max_turn_hold_seconds == 420
        runner._hmwa_hygiene_apply_result.assert_awaited_once()
        assert db.get_messages_as_conversation("hygiene-sid")[-1]["codex_reasoning_items"][-1]["encrypted_content"] == "fake-sol-checkpoint"
    finally:
        agent.close()
        db.close()


@pytest.mark.asyncio
async def test_native_deadline_revokes_late_commit(tmp_path, monkeypatch):
    run, runner, agent, db, history, hs, attempt = _setup(tmp_path, monkeypatch, timeout=350)
    clock = _accelerate_clock(monkeypatch, factor=1)
    started, release = threading.Event(), threading.Event()
    def provider(**kwargs):
        started.set()
        release.wait(3)
        return _events()
    client = SimpleNamespace(responses=SimpleNamespace(create=provider))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *args, **kw: None)
    try:
        task = asyncio.create_task(run())
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 2), 3)
        clock.advance(351)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, 3)
        assert hs.total_ceiling_seconds == hs.timeout_seconds == hs.max_turn_hold_seconds == 350
        assert attempt.commit_fence.is_cancelled
        runner._hmwa_hygiene_apply_result.assert_not_awaited()
        release.set()
        await asyncio.wait_for(attempt.future, 3)
        assert not agent._native_maintenance_committed
        assert "codex_reasoning_items" not in db.get_messages_as_conversation("hygiene-sid")[-1]
    finally:
        release.set()
        if hasattr(attempt, "future"):
            await asyncio.wait_for(attempt.future, 3)
        agent.close()
        db.close()


@pytest.mark.asyncio
async def test_native_user_cancel_keeps_unwind_fence(tmp_path, monkeypatch):
    run, runner, agent, db, history, hs, attempt = _setup(tmp_path, monkeypatch)
    started, release = threading.Event(), threading.Event()
    def provider(**kwargs):
        started.set()
        release.wait(3)
        return _events()
    client = SimpleNamespace(responses=SimpleNamespace(create=provider))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *args, **kw: None)
    task = asyncio.create_task(run())
    try:
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 2), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert attempt.commit_fence.is_cancelled
        release.set()
        await asyncio.wait_for(attempt.future, 3)
        runner._hmwa_hygiene_apply_result.assert_not_awaited()
        assert "codex_reasoning_items" not in db.get_messages_as_conversation("hygiene-sid")[-1]
    finally:
        release.set()
        if hasattr(attempt, "future"):
            await asyncio.wait_for(attempt.future, 3)
        agent.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("native,model", [(False, "gpt-6-astra"), (True, "gpt-5.6")])
async def test_ordinary_or_unsupported_route_keeps_short_turn_hold(tmp_path, monkeypatch, native, model):
    run, runner, agent, db, history, hs, attempt = _setup(
        tmp_path, monkeypatch, native=native, model=model,
    )
    started, release = threading.Event(), threading.Event()
    def compressor(messages, _prompt, **kwargs):
        started.set()
        release.wait(3)
        return messages, ""
    monkeypatch.setattr(agent, "_compress_context", compressor)
    assert hs.max_turn_hold_seconds == 10
    hs.max_turn_hold_seconds = 0.05  # accelerated old 10-second cap, no native budget override.
    try:
        with pytest.raises(importlib.import_module("gateway.run").HygieneTurnHoldExceeded):
            await asyncio.wait_for(run(), 3)
        assert started.is_set()
        assert hs.max_turn_hold_seconds == 0.05
        assert hs.timeout_seconds == 30 and hs.total_ceiling_seconds == 600
        release.set()
        await asyncio.wait_for(attempt.future, 3)
    finally:
        release.set()
        if hasattr(attempt, "future"):
            await asyncio.wait_for(attempt.future, 3)
        agent.close()
        db.close()


@pytest.mark.asyncio
async def test_native_timeout_resolution_is_profile_scoped_a_b_a(tmp_path, monkeypatch):
    for index, (label, configured) in enumerate((("A", 420), ("B", 480), ("A", 420))):
        run, runner, agent, db, history, hs, attempt = _setup(
            tmp_path / f"turn-{index}", monkeypatch, timeout=configured,
            profile_home=tmp_path / label,
        )
        calls = []
        client = SimpleNamespace(responses=SimpleNamespace(
            create=lambda **kwargs: calls.append(kwargs) or _events(),
        ))
        monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
        monkeypatch.setattr(agent, "_close_request_openai_client", lambda *args, **kw: None)
        try:
            await asyncio.wait_for(run(), 3)
            assert len(calls) == 1 and calls[0]["timeout"] == configured
            assert hs.timeout_seconds == hs.total_ceiling_seconds == hs.max_turn_hold_seconds == configured
            assert agent._native_maintenance_committed
        finally:
            agent.close()
            db.close()
