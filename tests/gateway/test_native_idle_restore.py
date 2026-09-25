"""Restored gateway idle job uses the normal cold surface without a stored prompt."""
import asyncio
import json
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from agent.idle_timetable import IdleTimetable
from agent.native_maintenance import idle_delegation_blocked
from agent.usage_anchor import capture_usage_anchor
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway import native_idle
from gateway.run_agent_cache import GatewayAgentCacheMixin
from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["success", "fallback_success", "persist_failure", "fresh_activity",
                                  "tick_false", "tick_raise", "completion_send_failure",
                                  "completion_schedule_failure", "postcommit_cancel",
                                  "delegation_running", "delegation_finalizing", "completion_pending",
                                  "completion_delivered", "completion_dropped", "other_parent",
                                  "ledger_unavailable"])
async def test_restored_missing_prompt_reaches_fake_provider_and_retires(tmp_path, monkeypatch, case):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "compression:\n  codex_responses_native: true\n"
        "  codex_responses_native_first: true\n"
        "  codex_responses_native_idle_after_seconds: 1\n"
        "  codex_responses_native_idle_min_tokens: 80000\n"
        "title_generation:\n  enabled: false\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread-007",
                           chat_type="thread", thread_id="thread-007", user_id="user-1")
    entry = store.get_or_create_session(source)
    db = store._db_for_key(entry.session_key)
    assert isinstance(db, SessionDB)
    db.append_messages_batch(entry.session_id, [
        {"role": "user", "content": "Saved fact", "timestamp": time.time() - 10},
        {"role": "assistant", "content": "Acknowledged", "timestamp": time.time() - 10}])
    history = db.get_messages_as_conversation(entry.session_id)
    db.patch_session_model_config(entry.session_id, {
        "gateway_runtime": {"provider": "openai-api", "base_url": "https://api.openai.com/v1",
                            "api_mode": "codex_responses"},
        "_usage_anchor": capture_usage_anchor(100_000, 100, history),
    })
    db._write_sql("UPDATE sessions SET model = ?, last_activity_at = ? WHERE id = ?",
                  ("gpt-5.6", time.time() - 10, entry.session_id))
    project = tmp_path / "saved-workspace"
    project.mkdir()
    (project / "AGENTS.md").write_text("Saved workspace context marker")
    db.update_session_cwd(entry.session_id, str(project))
    assert db.get_session(entry.session_id)["system_prompt"] is None
    sends = []
    class Adapter:
        async def send(self, chat_id, text, metadata=None):
            committed = db.get_messages_as_conversation(entry.session_id)[-1]
            sends.append((chat_id, text, metadata,
                          bool(committed.get("codex_reasoning_items") or committed["content"] == "Acknowledged summary")))
            if case == "completion_send_failure" and text.startswith("上下文压缩已完成"):
                raise RuntimeError("Discord send failed")
            return NS(success=True)
    adapter = Adapter()
    runner = NS(session_store=store, config=GatewayConfig(), _running=True,
                _restored_source=lambda e: e.origin,
                _profile_scope_for_source=lambda s: nullcontext(),
                _resolve_turn_toolsets=lambda *a: ([], None),
                _pinned_session_context_prompt=lambda *a: "Gateway context for this thread",
                _get_system_prompt_for_channel=lambda *a, **kw: "Channel policy",
                _delivery_adapter_for=lambda s: adapter)
    original = None
    monkeypatch.setattr(native_idle, "_profile_homes", lambda r: [home])
    from gateway import run
    monkeypatch.setattr(run, "_resolve_runtime_agent_kwargs_for_provider", lambda *a, **kw: {
        "api_key": "fake-key", "base_url": "https://api.openai.com/v1",
        "api_mode": "codex_responses", "provider": "openai-api"})
    fake_wire = []
    followup_wire = []
    def fake_client(self, *, reason, api_kwargs):
        assert reason == "native_maintenance", f"unexpected outbound request: {reason}"
        client = NS(responses=NS(create=lambda **kwargs: _events(kwargs, fake_wire)))
        return client
    def _events(kwargs, collected):
        collected.append(kwargs)
        events = [] if case == "fallback_success" else [{"type": "response.output_item.done", "item": {
            "type": "compaction", "encrypted_content": "opaque-test-checkpoint"}}]
        events.append({"type": "response.completed", "response": {
            "status": "completed", "output": [], "usage": {"input_tokens": 100_000}}})
        return iter(events)
    monkeypatch.setattr(AIAgent, "_create_request_openai_client", fake_client)
    monkeypatch.setattr(AIAgent, "_close_request_openai_client", lambda *a, **kw: None)
    from agent import idle_timetable
    table = IdleTimetable(home / "native_idle_timetable.json")
    try:
        assert table.backfill_once(lambda: native_idle._candidates(runner, home)) == 1
        assert table.backfill_once(lambda: pytest.fail("rescan")) == 0
        # New scheduler instance has no original agent or in-memory callback.
        restored = IdleTimetable(table.path)
        restored.load()
        assert restored._entries[entry.session_id]["idle_at"] < time.time() - 2
        monkeypatch.setattr(idle_timetable, "for_home", lambda *_: restored)
        monkeypatch.setattr(native_idle, "for_home", lambda *_: restored)
        original = native_idle._build_agent(runner, db, entry, source, db.get_session(entry.session_id))
        if case == "fallback_success":
            monkeypatch.setattr(type(original.context_compressor), "compress", lambda *a, **kw: [
                {"role": "user", "content": "Summary of saved fact"},
                {"role": "assistant", "content": "Acknowledged summary"},
            ])
        original._session_messages = db.get_messages_as_conversation(entry.session_id)
        runner._agent_cache = {entry.session_key: (original, "signature", 2, entry.session_id)}
        runner._agent_cache_lock = threading.Lock()
        runner._running_agent_ids = lambda: set()
        runner._release_evicted_agent_soft = lambda agent: GatewayAgentCacheMixin._release_evicted_agent_soft(runner, agent)
        runner._spawn_release_thread = lambda target, args, *a, **kw: target(*args)

        if case == "fresh_activity":
            # A foreground turn in another process wrote dialogue while its
            # timetable cancellation failed; the stale row must not call out.
            db.append_messages_batch(entry.session_id, [
                {"role": "user", "content": "Fresh question"},
                {"role": "assistant", "content": "Fresh answer"}])
        if case == "persist_failure":
            def refuse(*args, **kwargs):
                raise OSError("checkpoint storage unavailable")
            monkeypatch.setattr(SessionDB, "attach_native_checkpoint", refuse)
        if case in ("tick_false", "tick_raise"):
            def declined(agent, sid, cancellation):
                agent._native_idle_start_callback()
                if case == "tick_raise":
                    raise RuntimeError("maintenance failed")
                return False
            monkeypatch.setattr(native_idle, "_idle_tick", declined)
        if case == "postcommit_cancel":
            real_tick = native_idle._idle_tick
            def cancel_after_commit(agent, sid, cancellation):
                committed = real_tick(agent, sid, cancellation)
                assert committed
                cancellation.set()
                return committed
            monkeypatch.setattr(native_idle, "_idle_tick", cancel_after_commit)
        if case == "completion_schedule_failure":
            make_notice = native_idle._native_idle_thread_notice_callback
            def cannot_schedule(source, adapter, loop, home, text, *, muted=False):
                if text.startswith("上下文压缩已完成"):
                    raise RuntimeError("loop cannot schedule")
                return make_notice(source, adapter, loop, home, text, muted=muted)
            monkeypatch.setattr(native_idle, "_native_idle_thread_notice_callback", cannot_schedule)
        if case in {"delegation_running", "delegation_finalizing", "completion_pending",
                    "completion_delivered", "completion_dropped", "other_parent"}:
            state = {"delegation_running": "running", "delegation_finalizing": "finalizing",
                     "completion_pending": "completed", "completion_delivered": "completed",
                     "completion_dropped": "completed", "other_parent": "running"}[case]
            delivery = {"completion_delivered": "delivered", "completion_dropped": "dropped"}.get(case, "pending")
            now = time.time()
            db._write_sql("""INSERT INTO async_delegations
                (delegation_id, origin_session, parent_session_id, state, dispatched_at,
                 updated_at, delivery_state) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("blocked-unit", entry.session_key,
                 "replaced-session" if case == "other_parent" else entry.session_id,
                 state, now, now, delivery))
            assert idle_delegation_blocked(db, entry.session_id) is (case not in {
                "completion_delivered", "completion_dropped", "other_parent"})
        if case == "ledger_unavailable":
            original_read = SessionDB._read_one
            def unavailable(self, sql, params=()):
                if "FROM async_delegations" in sql:
                    raise OSError("ledger read failed")
                return original_read(self, sql, params)
            monkeypatch.setattr(SessionDB, "_read_one", unavailable)
        cold_resume = Mock(wraps=native_idle._build_agent)
        monkeypatch.setattr(native_idle, "_build_agent", cold_resume)
        loop = asyncio.get_running_loop()
        restored.start(lambda sid, item, cancelled: native_idle._execute(runner, home, loop, sid, item, cancelled))
        skipped = case in {"fresh_activity", "delegation_running", "delegation_finalizing",
                           "completion_pending", "ledger_unavailable"}
        async def finished():
            expected_sends = 0 if skipped else (1 if case in (
                "persist_failure", "tick_false", "tick_raise", "completion_schedule_failure",
                "postcommit_cancel") else 2)
            while (entry.session_id in restored._entries
                   or restored._worker_count
                   or (not skipped and case not in ("tick_false", "tick_raise") and not fake_wire)
                   or len(sends) < expected_sends):
                await asyncio.sleep(.02)
        await asyncio.wait_for(finished(), timeout=10)
        if skipped or case in ("tick_false", "tick_raise"):
            assert not fake_wire
            assert len(sends) == (0 if skipped else 1)
            assert all("上下文压缩已完成" not in notice[1] for notice in sends)
            assert runner._agent_cache[entry.session_key][0] is original
            assert not db.get_messages_as_conversation(entry.session_id)[-1].get("codex_reasoning_items")
            assert db.get_session(entry.session_id)["end_reason"] is None
            assert entry.session_id not in IdleTimetable(table.path)._read()["entries"]
            if skipped and case != "fresh_activity":
                cold_resume.assert_not_called()
            if case == "completion_pending":
                assert idle_delegation_blocked(db, entry.session_id)
            assert restored._worker_count == 0
            assert restored.waits < 20  # no overdue-revision redispatch loop
            restored.stop()
            return
        assert "Gateway context for this thread" in fake_wire[0]["instructions"]
        assert "Channel policy" in fake_wire[0]["instructions"]
        assert "Saved workspace context marker" in fake_wire[0]["instructions"]
        assert db.get_session(entry.session_id)["end_reason"] is None
        history_after = db.get_messages_as_conversation(entry.session_id)
        assert len(history_after) == 2  # adapter notices are not conversation turns
        if case == "persist_failure":
            assert not history_after[-1].get("codex_reasoning_items")
            assert runner._agent_cache[entry.session_key][0] is original
            assert original._session_messages and original._session_messages[0]["content"] == "Saved fact"
        else:
            if case == "fallback_success":
                assert [row["content"] for row in history_after] == [
                    "Summary of saved fact", "Acknowledged summary"]
            else:
                assert history_after[-1]["codex_reasoning_items"][-1]["encrypted_content"] == "opaque-test-checkpoint"
            assert entry.session_key not in runner._agent_cache
            assert original._session_messages == []
        assert sends[0][0] == "thread-007" and "正在后台压缩" in sends[0][1]
        assert sends[0][2]["thread_id"] == "thread-007"
        assert sends[0][2]["_interim_send"] is True
        if case not in ("persist_failure", "completion_schedule_failure", "postcommit_cancel"):
            assert len(sends) == 2
            assert sends[1][0] == "thread-007"
            assert sends[1][1] == "上下文压缩已完成，下次对话会接续压缩结果。"
            assert sends[1][2] == sends[0][2]
            assert sends[1][3] is True  # completion follows persisted checkpoint or summary
        else:
            assert len(sends) == 1  # rejected persistence or scheduling cannot claim success
        assert entry.session_id not in IdleTimetable(table.path)._read()["entries"]
        restored.stop()
        if case in ("persist_failure", "fallback_success"):
            assert db.get_session(entry.session_id)["end_reason"] is None
            assert any(row["content"] == "Saved fact" for row in db.get_messages(entry.session_id, include_inactive=True))
            return
        # Reclaim/eviction boundary: a NEW agent, from the same routing binding
        # and DB after the idle worker released its own agent, sends a real next
        # user turn with the persisted capsule rather than a blank/new session.
        fresh = native_idle._build_agent(runner, db, entry, source, db.get_session(entry.session_id))
        assert fresh is not None
        # A completion can still be pending when the parent's normal turn ends.
        # Do not let delivery acknowledgement delay or erase that turn's full timer.
        now = time.time()
        db._write_sql("""INSERT INTO async_delegations
            (delegation_id, origin_session, parent_session_id, state, dispatched_at,
             updated_at, delivery_state) VALUES (?, ?, ?, 'completed', ?, ?, 'pending')""",
            ("finishing-unit", entry.session_key, entry.session_id, now, now))
        def dispatch_to_fake_provider(wire):
            followup_wire.append(wire)
            return NS(output=[NS(type="message", content=[NS(type="output_text",
                text="Continued with context.")])], usage=NS(input_tokens=3000,
                output_tokens=25, total_tokens=3025), status="completed", model="gpt-5.6")
        fresh._interruptible_api_call = dispatch_to_fake_provider
        from agent import title_generator
        monkeypatch.setattr(title_generator, "maybe_auto_title", lambda *a, **kw: None)
        try:
            result = await asyncio.to_thread(
                fresh.run_conversation, "Next user message", conversation_history=db.get_messages_as_conversation(entry.session_id))
            assert result["final_response"] == "Continued with context."
            rearmed = IdleTimetable(table.path)._read()["entries"][entry.session_id]
            assert rearmed["delay"] == 1
            assert rearmed["idle_at"] >= now
            assert idle_delegation_blocked(db, entry.session_id)
            db._write_sql("UPDATE async_delegations SET delivery_state='delivered' WHERE delegation_id='finishing-unit'")
            assert not idle_delegation_blocked(db, entry.session_id)
            assert IdleTimetable(table.path)._read()["entries"][entry.session_id] == rearmed
            assert followup_wire
            assert any(item.get("type") == "compaction" and
                       item.get("encrypted_content") == "opaque-test-checkpoint"
                       for item in followup_wire[0]["input"])
            assert db.get_session(entry.session_id)["end_reason"] is None
            assert store.peek_session_id(entry.session_key) == entry.session_id
            assert any(row["content"] == "Saved fact" for row in db.get_messages(entry.session_id, include_inactive=True))
        finally:
            fresh.close()
    finally:
        if original is not None:
            original.close()
        db.close()


def test_disabled_profile_bootstrap_creates_no_timetable(tmp_path, monkeypatch):
    home = tmp_path / "disabled"
    home.mkdir()
    (home / "config.yaml").write_text("compression:\n  codex_responses_native: false\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(native_idle, "_profile_homes", lambda runner: [home])
    assert native_idle.bootstrap(NS(_running=True), None) == []
    assert not (home / "native_idle_timetable.json").exists()
    assert not (home / "native_idle_timetable.json.lock").exists()


def test_success_reclaims_only_claimed_revision_and_original_cached_agent(tmp_path):
    table = IdleTimetable(tmp_path / "table.json")
    key, sid = "gateway-key", "session"
    old = table.update(sid, key, time.time() - 5, 1)
    item = dict(table._entries[sid])
    original, replacement = NS(session_id=sid), NS(session_id=sid)
    released = []
    runner = NS(_agent_cache={key: (original, "sig")}, _agent_cache_lock=threading.Lock(),
                _running=True, _running_agent_ids=lambda: set(),
                _release_evicted_agent_soft=lambda a: released.append(a),
                _spawn_release_thread=lambda target, args, *a, **kw: target(*args))
    table.update(sid, key, time.time(), 100)
    native_idle._evict_compacted_cache(runner, table, sid, item, original)
    assert table._entries[sid]["revision"] != old and not released
    # Even when the claimed revision is still current, a newly cached
    # foreground agent must never be popped by the old background worker.
    table.update(sid, key, time.time() - 5, 1)
    current = dict(table._entries[sid])
    runner._agent_cache[key] = (replacement, "new-sig")
    native_idle._evict_compacted_cache(runner, table, sid, current, original)
    assert runner._agent_cache[key][0] is replacement and not released
