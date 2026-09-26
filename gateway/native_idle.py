"""Gateway integration for the profile-owned native idle timetable."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from pathlib import Path

from agent.idle_timetable import for_home
from agent.native_maintenance import _idle_tick, eligible, idle_delegation_blocked
from gateway.session import build_session_context

logger = logging.getLogger("gateway.run")

def enabled():
    """Call under the owning profile scope, before creating any timetable files."""
    from hermes_cli.config import load_config
    compression = load_config().get("compression") or {}
    return bool(compression.get("codex_responses_native_first")
                and compression.get("codex_responses_native")
                and (compression.get("codex_responses_native_idle_after_seconds") or 0) > 0)


_IDLE_START = "-# ⏳ 正在压缩上下文…"
_IDLE_END = {
    "completed": "-# ✓ 上下文已压缩，下次对话将接续压缩结果。",
    "failed": "-# ⚠ 上下文压缩未完成，原始记录已保留。",
    "cancelled": "-# ⏸ 上下文压缩已取消。",
}


class _IdleThreadNotice:
    """One pass owns one Discord message; completion waits for its start send."""

    def __init__(self, adapter, loop, profile_home, thread_id, metadata):
        self.adapter = adapter
        self.loop = loop
        self.profile_home = profile_home
        self.thread_id = thread_id
        self.metadata = metadata
        self._start = None
        self._started = False
        self._finished = False

    def _schedule(self, coroutine):
        from agent.async_utils import safe_schedule_threadsafe
        future = safe_schedule_threadsafe(
            coroutine, self.loop, logger=logger, log_message="Idle compaction notice scheduling error")
        if future is not None:
            def observe(done):
                with suppress(Exception):
                    done.result()
            future.add_done_callback(observe)
        return future

    def __call__(self):
        if self._started:
            return
        self._started = True
        self._start = self._schedule(self._send())

    async def _send(self):
        from gateway.run import _async_profile_runtime_scope
        try:
            async with asyncio.timeout(10):
                async with _async_profile_runtime_scope(self.profile_home):
                    result = await self.adapter.send(self.thread_id, _IDLE_START, metadata=self.metadata)
            if getattr(result, "success", False) and getattr(result, "message_id", None):
                return str(result.message_id)
            logger.debug("Discord idle notice not delivered to %s", self.thread_id)
        except Exception:
            logger.debug("Discord idle notice failed for %s", self.thread_id, exc_info=True)
        return None

    def finish(self, status):
        if self._finished or not self._started or self._start is None:
            return
        self._finished = True
        self._schedule(self._edit(status))

    async def _edit(self, status):
        from gateway.run import _async_profile_runtime_scope
        try:
            # The start send may be slow even after the durable commit. Await it
            # on the loop, never on the compression worker, before editing its ID.
            start = self._start
            if start is None:
                return
            message_id = await asyncio.wrap_future(start)
            if not message_id:
                return
            async with asyncio.timeout(10):
                async with _async_profile_runtime_scope(self.profile_home):
                    result = await self.adapter.edit_message(
                        self.thread_id, message_id, _IDLE_END[status], metadata=self.metadata)
            if not getattr(result, "success", False):
                logger.debug("Discord idle notice edit failed for %s", self.thread_id)
        except Exception:
            logger.debug("Discord idle notice edit failed for %s", self.thread_id, exc_info=True)


def native_idle_start_callback(source, adapter, loop, profile_home, *, muted=False):
    """Bind a pass to its originating Discord thread and owning profile."""
    from gateway.config import Platform
    from gateway.run import _interim_metadata, _non_conversational_metadata
    thread_id = str(getattr(source, "thread_id", "") or "")
    if source.platform != Platform.DISCORD or not thread_id or adapter is None or muted:
        return None
    metadata = _interim_metadata(_non_conversational_metadata(
        {"thread_id": thread_id}, platform=Platform.DISCORD))
    return _IdleThreadNotice(adapter, loop, profile_home, thread_id, metadata)


def _profile_homes(runner):
    from gateway.run_heartbeat_restore import _watched_homes
    from hermes_constants import get_hermes_home
    return _watched_homes(runner, getattr(getattr(runner, "session_store", None), "_routing_home", None)
                          or get_hermes_home())


def _candidates(runner, home):
    """Only current routing bindings; history/archived sessions are not candidates."""
    from gateway.run import _profile_runtime_scope
    from hermes_constants import get_hermes_home
    from hermes_cli.config import load_config
    result = []
    with _profile_runtime_scope(home):
        compression = (load_config().get("compression") or {})
        delay = int(compression.get("codex_responses_native_idle_after_seconds") or 0)
        if not delay or not compression.get("codex_responses_native_first") or not compression.get("codex_responses_native"):
            return result
        for entry in runner.session_store.list_sessions():
            if (not entry.origin or not entry.session_id or entry.suspended or entry.resume_pending
                    or entry.active_turn_token or entry.expiry_finalized):
                continue
            try:
                source = runner._restored_source(entry)
                with runner._profile_scope_for_source(source):
                    if Path(get_hermes_home()).resolve() != Path(home).resolve():
                        continue
                    db = runner.session_store._db_for_key(entry.session_key)
                    row = db.get_session(entry.session_id) if db else None
                    if (not row or row.get("end_reason") is not None or row.get("archived") or row.get("hidden")
                            or row.get("message_count", 0) < 2
                            or db.get_active_message_watermark(entry.session_id) <= 0):
                        continue
                    lease = db._read_one(
                        "SELECT expires_at FROM session_turn_leases WHERE conversation_id = ?",
                        (db._session_turn_lease_key(entry.session_id),))
                    if lease and float(lease["expires_at"]) > time.time():
                        continue
                    messages = db.get_messages_as_conversation(entry.session_id)
                    if (not messages or messages[-1].get("role") != "assistant"
                            or messages[-1].get("tool_calls")):
                        continue
                    last = row.get("last_activity_at")
                    idle_at = float(last) if last is not None else entry.updated_at.timestamp()
                    result.append((entry.session_id, entry.session_key, idle_at, delay))
            except Exception:
                logger.warning("Idle migration failed reading session %s; migration remains unmarked",
                               entry.session_id, exc_info=True)
                raise
    return result


def _build_agent(runner, db, entry, source, row):
    """Cold-resume the ordinary gateway agent surface, not the memory-only hygiene agent."""
    from gateway.run import _checkpoint_agent_kwargs, _load_gateway_config, _platform_config_key
    from gateway.run import _resolve_runtime_agent_kwargs_for_provider
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from agent.usage_anchor import restore_usage_anchor
    from tools.mcp_tool_agent import restore_agent_tool_prefix

    config = _load_gateway_config()
    model = row.get("model")
    route = SessionDB.session_gateway_runtime(row)
    provider = route.get("provider")
    if not model or not provider:
        logger.warning("Idle cold resume skipped %s: missing model/provider route", entry.session_id)
        return None
    runtime = _resolve_runtime_agent_kwargs_for_provider(provider, target_model=model)
    runtime.pop("model", None)
    runtime.pop("_fallback_notice", None)
    for field in ("base_url", "api_mode"):
        if route.get(field):
            runtime[field] = route[field]
    if not runtime.get("api_key") or runtime.get("api_mode") != "codex_responses":
        logger.warning("Idle cold resume skipped %s: unavailable saved Responses route", entry.session_id)
        return None
    platform_key = _platform_config_key(source.platform)
    enabled, disabled = runner._resolve_turn_toolsets(config, source, platform_key)
    context = build_session_context(source, runner.config, entry)
    redact = bool((config.get("privacy") or {}).get("redact_pii", False))
    ephemeral = runner._pinned_session_context_prompt(context, redact, entry.session_key)
    channel = runner._get_system_prompt_for_channel(source.platform, source.chat_id or "",
                    thread_id=source.thread_id, parent_id=source.parent_chat_id)
    if channel:
        ephemeral = (ephemeral + "\n\n" + channel).strip()
    agent = AIAgent(
        model=model, **runtime, **_checkpoint_agent_kwargs(config),
        quiet_mode=True, verbose_logging=False, enabled_toolsets=enabled, disabled_toolsets=disabled,
        ephemeral_system_prompt=ephemeral or None, session_id=entry.session_id,
        platform=platform_key, session_db=db, gateway_session_key=entry.session_key,
        skip_context_files=bool(((config.get("gateway") or {}).get("platforms") or {}).get(platform_key, {}).get("skip_context_files"))
            if isinstance((config.get("gateway") or {}).get("platforms"), dict) else False,
        load_soul_identity=True, **{field: getattr(source, field) for field in (
            "user_id", "user_id_alt", "user_name", "chat_id", "chat_name", "chat_type", "thread_id")},
    )
    agent._end_session_on_close = False
    agent._native_idle_owned_by_scheduler = True
    try:
        # Normal cold continuation uses the stored prompt if present; old rows without
        # it must rebuild with this same gateway surface, not send empty instructions.
        stored = row.get("system_prompt") or row.get("_system_prompt_resolved")
        agent._cached_system_prompt = stored if isinstance(stored, str) and stored.strip() else agent._build_system_prompt(None)
        saved_tools = row.get("tool_names")
        if saved_tools:
            restore_agent_tool_prefix(agent, json.loads(saved_tools))
        restore_usage_anchor(agent, db.get_messages_as_conversation(entry.session_id))
        return agent
    except Exception:
        agent.close()
        raise


def _execute(runner, home, loop, sid, item, cancellation):
    from gateway.run import _profile_runtime_scope
    from gateway.config import Platform
    from agent.native_maintenance import cancel_idle
    with _profile_runtime_scope(home):
        entry = runner.session_store.lookup_by_session_key(item["session_key"])
        if (not entry or entry.session_id != sid or entry.suspended or entry.resume_pending
                or entry.active_turn_token or entry.expiry_finalized):
            logger.info("Idle check skipped %s: routing changed or foreground active", sid)
            return
        source = runner._restored_source(entry)
        with runner._profile_scope_for_source(source):
            shared_db = runner.session_store._db_for_key(entry.session_key)
            if not shared_db or cancellation.is_set():
                return
            # This background job owns its handle. Gateway shutdown may close the
            # store's cached handles while a provider request is still unwinding.
            from hermes_state import SessionDB
            db = SessionDB(db_path=shared_db.db_path)
            agent = None
            cwd_token = None
            try:
                row = db.get_session(sid)
                if (not row or row.get("end_reason") is not None or row.get("archived")
                        or row.get("hidden") or cancellation.is_set()):
                    return
                # A failed timetable update/cancel or a second process must not
                # turn a stale file deadline into a request for a fresh turn.
                last_activity = row.get("last_activity_at")
                last_activity = float(last_activity) if last_activity is not None else entry.updated_at.timestamp()
                latest_user = db._read_one(
                    "SELECT MAX(timestamp) AS latest FROM messages WHERE session_id = ? AND active = 1 AND role = 'user'",
                    (sid,))
                if (last_activity > item["idle_at"] + .001
                        or (latest_user and latest_user["latest"] is not None
                            and float(latest_user["latest"]) > item["idle_at"] + .001)
                        or time.time() < max(float(last_activity), item["idle_at"]) + item["delay"]):
                    return
                lease = db._read_one(
                    "SELECT expires_at FROM session_turn_leases WHERE conversation_id = ?",
                    (db._session_turn_lease_key(sid),))
                if lease and float(lease["expires_at"]) > time.time():
                    logger.info("Idle check skipped %s: foreground turn lease is active", sid)
                    return
                # Retire this due revision before cold construction, notices or provider work.
                # The common tick repeats the guard for local timers and a later dispatch.
                if idle_delegation_blocked(db, sid):
                    logger.info("Idle check skipped %s: delegation active or completion pending", sid)
                    return
                cache = getattr(runner, "_agent_cache", None)
                lock = getattr(runner, "_agent_cache_lock", None)
                with lock if lock is not None else suppress():
                    cached = cache.get(entry.session_key) if cache is not None else None
                    original = cached[0] if isinstance(cached, tuple) and cached else None
                    if getattr(original, "session_id", None) != sid:
                        original = None
                # Cold prompt/context-file assembly must see this session's
                # persisted workspace, not the gateway process's launch cwd.
                from agent.runtime_cwd import set_session_cwd
                cwd_token = set_session_cwd(row.get("cwd"))
                agent = _build_agent(runner, db, entry, source, row)
                if agent is None or not eligible(agent) or cancellation.is_set():
                    return
                agent._native_idle_cancel = cancellation
                agent._native_idle_handle = True
                adapter = runner._delivery_adapter_for(source)
                notice = native_idle_start_callback(
                    source, adapter, loop, home) if source.platform == Platform.DISCORD else None
                setattr(agent, "_native_idle_start_callback", notice)
                timetable = for_home(home)
                with timetable._condition:
                    timetable._active_agents[sid] = agent
                committed = False
                try:
                    committed = _idle_tick(agent, sid, cancellation)
                    if committed and original is not None:
                        _evict_compacted_cache(runner, timetable, sid, item, original)
                    return committed
                finally:
                    if notice is not None:
                        try:
                            # Commit wins over a later foreground cancellation: a new
                            # turn cannot undo a checkpoint already durably attached.
                            notice.finish("completed" if committed else
                                          "cancelled" if cancellation.is_set() else "failed")
                        except Exception:
                            logger.debug("Discord idle notice finalization failed for %s", sid, exc_info=True)
            finally:
                if agent is not None:
                    cancel_idle(agent, timetable=False)
                    with for_home(home)._condition:
                        for_home(home)._active_agents.pop(sid, None)
                    agent.close()
                if cwd_token is not None:
                    from agent.runtime_cwd import reset_session_cwd
                    reset_session_cwd(cwd_token)
                db.close()

def _evict_compacted_cache(runner, timetable, sid, item, original):
    """Reclaim only the agent observed before this exact durable checkpoint."""
    from gateway.run_agent_cache import _first_agent
    cache = getattr(runner, "_agent_cache", None)
    lock = getattr(runner, "_agent_cache_lock", None)
    if cache is None or lock is None:
        return
    key = item["session_key"]
    with timetable._condition:
        current = timetable._entries.get(sid)
        if (not current or current["revision"] != item["revision"]
                or timetable._stopped or not runner._running):
            return
        with lock:
            if (_first_agent(cache.get(key)) is not original
                    or id(original) in runner._running_agent_ids()):
                return
            cache.pop(key)
    runner._spawn_release_thread(runner._release_evicted_agent_soft, (original,),
                                 f"native-idle-evict-{sid[:12]}", inline_fallback=True,
                                 session_key=key)


def bootstrap(runner, loop) -> list[tuple[Path, int]]:
    from gateway.run import _profile_runtime_scope
    restored = []
    for home in _profile_homes(runner):
        if not getattr(runner, "_running", False):
            break
        try:
            with _profile_runtime_scope(home):
                if not enabled():
                    continue
                table = for_home(home)
                count = table.backfill_once(lambda h=home: _candidates(runner, h))
                if not getattr(runner, "_running", False):
                    break
                table.start(lambda sid, item, cancelled, h=home: _execute(
                    runner, h, loop, sid, item, cancelled))
                restored.append((home, count))
                with table._condition:
                    pending = len(table._entries)
                logger.info("Idle timetable loaded: profile=%s backfilled=%d pending=%d workers=%d",
                            home.name, count, pending, table.workers)
        except Exception:
            logger.warning("Idle timetable unavailable for profile %s (no history rescan on corrupt file)",
                           home, exc_info=True)
    return restored


def stop(runner):
    from agent.idle_timetable import stop_for_home
    for home in _profile_homes(runner):
        stop_for_home(home)
