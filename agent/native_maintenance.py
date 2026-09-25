"""Opt-in native-first maintenance for eligible Codex Responses sessions.

A maintenance response is never a chat turn: only a completed, nonempty compaction item
is admitted. Raw transcript rows remain untouched; the opaque item is a sidecar on the
latest assistant row and can be replayed after a cache eviction or process restart.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)
IDLE_MIN_INPUT_TOKENS = 80_000
IDLE_MIN_GROWTH_TOKENS = 16_384
_IDLE_KEY = "native_idle_maintenance"


def _prefix_digest(messages: list[dict]) -> str:
    from agent.usage_anchor import message_fingerprint
    return hashlib.sha256("|".join(message_fingerprint(m) or "" for m in messages).encode()).hexdigest()


def eligible(agent: Any) -> bool:
    from agent.codex_responses_adapter import classify_responses_route
    from agent.native_compaction import native_compaction_context_management

    if (getattr(agent, "api_mode", None) != "codex_responses"
            or not getattr(agent, "compression_native_first", False)):
        return False
    return bool(native_compaction_context_management(agent, **classify_responses_route(agent)._asdict()))


def before_summary(agent: Any, messages: list[dict]) -> bool:
    """An unpriced prior checkpoint must get one real normal response first."""
    if not eligible(agent):
        return False
    if not getattr(agent.context_compressor, "awaiting_real_usage_after_compression", False):
        return True
    from agent.codex_responses_adapter import has_replayable_native_compaction_checkpoint
    return not has_replayable_native_compaction_checkpoint(agent, messages)


def _usage(usage: Any) -> tuple[Any, Any, Any]:
    from agent.codex_runtime import _event_field
    input_tokens = _event_field(usage, "input_tokens")
    output_tokens = _event_field(usage, "output_tokens")
    details = _event_field(usage, "input_tokens_details")
    cached = _event_field(details, "cached_tokens")
    return input_tokens, cached, output_tokens


def attempt(agent: Any, messages: list[dict], system_prompt: str, before_tokens: int,
            *, phase: str, expected_watermark: int | None = None,
            turn_lease_holder: str | None = None, commit_fence: Any = None,
            on_idle_start: Any = None) -> bool:
    """Try a forced inline /responses checkpoint; return False for *any* non-commit.

    A maintenance request must not use /responses/compact (404 on Codex OAuth).
    The SDK's high-level stream loses output_item.done on an empty completed.output,
    so consume the raw events with the production assembler. No output is delivered.
    """
    agent._native_maintenance_abort_fallback = False
    if not eligible(agent) or not messages or getattr(agent, "_interrupt_requested", False):
        agent._native_maintenance_abort_fallback = True
        return False
    carrier = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if carrier is None:
        return False
    db, sid = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    row_id = carrier.get("_row_id")
    plain = [m for m in messages if m.get("role") != "system"]
    if db is not None and sid and not getattr(agent, "_persist_disabled", False):
        # A durable attempt must have a durable carrier. Never publish an ephemeral
        # checkpoint which disappears as soon as the agent is evicted.
        if not isinstance(expected_watermark, int):
            agent._native_maintenance_abort_fallback = True
            return False
        # The checkpoint covers ALL input, not just its assistant carrier. An
        # unpersisted or changed tail would move its boundary and lose context.
        durable = db.get_messages_as_conversation(sid, repair_alternation=True, include_row_ids=True)
        if (len(plain) != len(durable) or not durable
                or durable[-1].get("_row_id") != expected_watermark
                or any(m.get("role") != d.get("role")
                       or db._loaded_view_content(m.get("role"), m.get("content")) != d.get("content")
                       or m.get("tool_calls") != d.get("tool_calls") for m, d in zip(plain, durable))):
            agent._native_maintenance_abort_fallback = True
            return False
        for m, d in zip(plain, durable):
            m["_row_id"] = d["_row_id"]
        row_id = carrier.get("_row_id")
        if not isinstance(row_id, int):
            agent._native_maintenance_abort_fallback = True
            return False
    started = time.monotonic()
    client = stream = None
    outcome = "failed"
    reason = "no_compaction_item"
    input_tokens = cached_tokens = output_tokens = None
    after_estimate = "unknown_until_normal_usage"
    try:
        from agent.chat_completion_helpers import build_api_kwargs
        from agent.turn_request_assembly import maintenance_api_prefix
        from agent.codex_runtime import (
            _consume_codex_event_stream, _event_field, _sanitize_consumer_codex_request,
        )
        from agent.codex_responses_adapter import classify_responses_route
        from agent.sdk_transform_bypass import bypass_sdk_request_transform

        route = classify_responses_route(agent)
        api_prefix = maintenance_api_prefix(agent, messages, system_prompt)
        covered = [m for m in api_prefix if m.get("role") in {"user", "assistant", "tool"}]
        # The checkpoint carrier must survive the ordinary send transformations.
        # Otherwise replay cannot locate its covered boundary and must fail closed.
        if not covered or not any(m.get("role") == "assistant" for m in covered):
            return False
        # The ordinary builder consumes one-shot reasoning/output overrides.
        # Maintenance must use the same frozen request prefix WITHOUT stealing
        # settings intended for the next user-facing model call.
        one_shots = {key: vars(agent)[key] for key in (
            "_ephemeral_reasoning_off", "_ephemeral_max_output_tokens", "_wire_reasoning_config") if key in vars(agent)}
        try:
            kwargs = build_api_kwargs(agent, api_prefix)
        finally:
            for key in ("_ephemeral_reasoning_off", "_ephemeral_max_output_tokens", "_wire_reasoning_config"):
                if key in one_shots:
                    setattr(agent, key, one_shots[key])
                else:
                    vars(agent).pop(key, None)
        if "context_management" not in kwargs:
            return False
        # The full frozen instructions/tools prefix comes from the ordinary builder.
        # A low maintenance-only threshold forces a real checkpoint even below the
        # automatic native threshold; it never changes the next normal call's policy.
        kwargs["context_management"] = [{"type": "compaction", "compact_threshold": 1024}]
        kwargs["tool_choice"] = "none"
        # The entire ordinary request is the frozen prefix. A maintenance-only
        # suffix makes an assistant/tool-ended transcript a closed, inert request.
        kwargs["input"] = list(kwargs["input"]) + [{"role": "user", "content":
            "Prepare the handoff. Reply only READY."}]
        kwargs["timeout"] = min(float(kwargs.get("timeout") or 300), 300.0)
        kwargs = agent._get_transport().preflight_kwargs(
            kwargs, allow_stream=True, is_github_responses=route.is_github_responses,
            sanitize_harmony_tokens=route.is_codex_backend,
        )
        kwargs = _sanitize_consumer_codex_request(agent, kwargs)
        kwargs["stream"] = True
        client = agent._create_request_openai_client(reason="native_maintenance", api_kwargs=kwargs)
        if phase == "idle":
            agent._native_idle_client = client
            if getattr(agent, "_native_idle_cancel", threading.Event()).is_set():
                reason = "new_turn_or_shutdown"
                agent._native_maintenance_abort_fallback = True
                return False
            if on_idle_start is not None:
                on_idle_start()
        stream = client.responses.create(**bypass_sdk_request_transform(kwargs))
        terminal_seen = False
        def _terminal_event(event):
            nonlocal terminal_seen
            if _event_field(event, "type") == "response.completed":
                terminal_seen = True
        result = _consume_codex_event_stream(stream, model=kwargs["model"], on_event=_terminal_event)
        input_tokens, cached_tokens, output_tokens = _usage(result.usage)
        if not terminal_seen or result.status != "completed":
            reason = "truncated_stream" if not terminal_seen else f"response_{result.status}"
            return False
        checkpoints = []
        for item in result.output:
            if _event_field(item, "type") == "compaction":
                encrypted = _event_field(item, "encrypted_content")
                if isinstance(encrypted, str) and encrypted:
                    checkpoints.append(encrypted)
        if not checkpoints:
            return False
        if phase == "idle" and getattr(agent, "_native_idle_cancel", threading.Event()).is_set():
            reason = "new_turn_or_shutdown"
            agent._native_maintenance_abort_fallback = True
            return False
        from agent.codex_responses_adapter import _classify_responses_issuer, _wire_model_identity
        checkpoint = {
            "type": "compaction", "encrypted_content": checkpoints[-1],
            "_issuer_kind": _classify_responses_issuer(
                base_url=getattr(agent, "base_url", None), **route._asdict()),
            "_issuer_model": _wire_model_identity(kwargs["model"]),
        }
        if isinstance(expected_watermark, int):
            checkpoint["_checkpoint_watermark"] = expected_watermark
            checkpoint["_checkpoint_count"] = len(covered)
            checkpoint["_checkpoint_prefix_digest"] = _prefix_digest(covered)
        if commit_fence is not None and not commit_fence.begin_commit(
                getattr(agent, "_native_idle_cancel", None) if phase == "idle" else None):
            reason = "commit_admission_revoked"
            agent._native_maintenance_abort_fallback = True
            return False
        try:
            if db is not None and sid and not getattr(agent, "_persist_disabled", False):
                try:
                    attached = db.attach_native_checkpoint(
                        sid, expected_watermark, row_id, checkpoint,
                        turn_lease_holder=turn_lease_holder,
                    )
                except Exception:
                    # A competing foreground admission is a refused CAS, never
                    # permission to begin an ordinary summary on stale input.
                    agent._native_maintenance_abort_fallback = True
                    raise
                if not attached:
                    reason = "stale_transcript_or_lease"
                    agent._native_maintenance_abort_fallback = True
                    return False
            # Only mutate the in-memory sidecar AFTER the durable CAS succeeded.
            # An idle turn may have been admitted immediately after that transaction;
            # its DB reload wins, rather than rewriting its live list behind its back.
            if phase != "idle" or not getattr(agent, "_native_idle_cancel", threading.Event()).is_set():
                carrier["codex_reasoning_items"] = list(carrier.get("codex_reasoning_items") or []) + [checkpoint]
            if phase == "idle" and not getattr(agent, "_native_idle_cancel", threading.Event()).is_set():
                live = getattr(agent, "_session_messages", None)
                if isinstance(live, list):
                    for m in reversed(live):
                        if m.get("role") == "assistant" and m.get("_row_id") == row_id:
                            m["codex_reasoning_items"] = list(m.get("codex_reasoning_items") or []) + [checkpoint]
                            break
        finally:
            if commit_fence is not None:
                commit_fence.finish_commit()
        # A foreground turn may start immediately after the durable CAS. It will
        # reload the checkpoint itself; never rewrite its live usage state afterward.
        update_local = phase != "idle" or not getattr(agent, "_native_idle_cancel", threading.Event()).is_set()
        if update_local:
            try:
                note = getattr(agent.context_compressor, "note_native_compaction_checkpoint", None)
                if callable(note):
                    note()
                from agent.usage_anchor import set_usage_anchor
                set_usage_anchor(agent, None)
            except Exception:
                logger.warning("Native checkpoint committed but local usage latch failed", exc_info=True)
            agent._native_maintenance_pending_followup = phase
        if phase != "idle" and db is not None and sid and isinstance(expected_watermark, int):
            try:
                db.patch_session_model_config(sid, {_IDLE_KEY: {"model": agent.model, "pressure": before_tokens}})
            except Exception:
                logger.warning("Native checkpoint committed but growth baseline write failed", exc_info=True)
        outcome, reason = "native", "checkpoint_committed"
        return True
    except Exception as exc:
        reason = type(exc).__name__
        logger.warning("Native maintenance failed (%s, session=%s, reason=%s)", phase, sid, reason, exc_info=True)
        return False
    finally:
        if phase == "idle" and getattr(agent, "_native_idle_client", None) is client:
            agent._native_idle_client = None
        if stream is not None:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Native maintenance stream close failed", exc_info=True)
        if client is not None:
            agent._close_request_openai_client(client, reason="native_maintenance_complete")
        logger.info(
            "Native maintenance phase=%s outcome=%s reason=%s session=%s "
            "before_full_estimate=%s after_full_estimate=%s "
            "usage_input=%s usage_cached=%s usage_output=%s duration_seconds=%.2f",
            phase, outcome, reason, sid, before_tokens, after_estimate,
            input_tokens, cached_tokens, output_tokens, time.monotonic() - started,
        )


def _idle_pressure(agent: Any, messages: list[dict]) -> int:
    from agent.codex_responses_adapter import has_replayable_native_compaction_checkpoint
    from agent.usage_anchor import anchored_context_tokens
    anchored = anchored_context_tokens(messages, getattr(agent, "_usage_anchor", None))
    if anchored is not None:
        return anchored
    if has_replayable_native_compaction_checkpoint(agent, messages):
        # Encrypted checkpoint size is not provider token usage. Wait for an
        # actual normal follow-up to price the full post-checkpoint request.
        return 0
    from agent.turn_context import _preflight_request_tokens
    return _preflight_request_tokens(agent, messages, getattr(agent, "_cached_system_prompt", "") or "")


def arm_idle(agent: Any, result: Any) -> None:
    """Persist gateway turns; retain the local timer for non-gateway agents."""
    seconds = getattr(agent, "compression_native_idle_after_seconds", 0)
    db, sid = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    if (not seconds or not eligible(agent) or db is None or not sid
            or getattr(agent, "_persist_disabled", False)
            or not isinstance(result, dict) or not result.get("completed")):
        return
    key = str(getattr(agent, "_gateway_session_key", None) or "")
    if not key:
        from agent.periodic_scheduler import schedule
        cancel_idle(agent, timetable=False)
        generation = threading.Event()
        agent._native_idle_cancel = generation
        agent._native_idle_handle = schedule(lambda: _idle_tick(agent, sid, generation), seconds)
        return
    from agent.idle_timetable import for_home
    from pathlib import Path
    for_home(Path(db.db_path).parent).update(sid, key, time.time(), seconds)


def cancel_idle(agent: Any, *, timetable: bool = True) -> None:
    if (timetable and getattr(agent, "_gateway_session_key", None)
            and getattr(agent, "compression_native_idle_after_seconds", 0) > 0
            and eligible(agent)
            and not getattr(agent, "_native_idle_owned_by_scheduler", False)
            and getattr(agent, "session_id", None) and getattr(agent, "_session_db", None)):
        from pathlib import Path
        from agent.idle_timetable import for_home
        try:
            for_home(Path(agent._session_db.db_path).parent).cancel(agent.session_id)
        except Exception:
            # A corrupt/full timetable must never prevent the foreground turn;
            # the DB watermark/lease still fences a late idle commit.
            logger.warning("Could not invalidate idle timetable for %s", agent.session_id, exc_info=True)
    cancellation = getattr(agent, "_native_idle_cancel", None)
    if cancellation is not None:
        cancellation.set()
    fence = getattr(agent, "_native_idle_fence", None)
    if fence is not None:
        fence.revoke_commit_admission()
        fence.try_cancel_before_commit()
    client = getattr(agent, "_native_idle_client", None)
    if client is not None:
        agent._abort_request_openai_client(client, reason="native_idle_superseded")
    handle = getattr(agent, "_native_idle_handle", None)
    agent._native_idle_handle = None
    if handle is not None and hasattr(handle, "cancel"):
        handle.cancel()


def _idle_tick(agent: Any, sid: str, generation: threading.Event | None = None) -> bool:
    """One bounded idle pass; zero-wait cross-process lease, fresh snapshot, no chat."""
    db = getattr(agent, "_session_db", None)
    generation = generation or getattr(agent, "_native_idle_cancel", None)
    if (generation is None or generation.is_set() or generation is not getattr(agent, "_native_idle_cancel", None)
            or db is None or sid != getattr(agent, "session_id", None) or not eligible(agent)
            or getattr(agent, "compression_native_idle_after_seconds", 0) <= 0):
        return False
    try:
        if (sid != getattr(agent, "session_id", None)
                or getattr(agent, "_native_idle_handle", None) is None
                or generation.is_set() or generation is not getattr(agent, "_native_idle_cancel", None)):
            return False
        messages = db.get_messages_as_conversation(sid, repair_alternation=True, include_row_ids=True)
        if not messages or messages[-1].get("role") != "assistant":
            return False
        pressure = _idle_pressure(agent, messages)
        compressor = agent.context_compressor
        minimum = getattr(agent, "compression_native_idle_min_tokens", IDLE_MIN_INPUT_TOKENS)
        if (pressure < minimum
                or getattr(compressor, "context_length", 0) < minimum
                or getattr(compressor, "awaiting_real_usage_after_compression", False)):
            return False
        previous = db.get_session_model_config_value(sid, _IDLE_KEY, None)
        if (isinstance(previous, dict) and previous.get("model") == agent.model
                and pressure < previous.get("pressure", 0) + IDLE_MIN_GROWTH_TOKENS):
            return False
        watermark = db.get_active_message_watermark(sid)
        # The network request never owns the session turn lease. The short
        # transactional checkpoint CAS rejects a foreground lease or moved tail.
        db.patch_session_model_config(sid, {_IDLE_KEY: {"model": agent.model, "pressure": pressure}})
        from agent.conversation_compression import CompressionCommitFence
        fence = CompressionCommitFence(total_ceiling_seconds=300)
        agent._native_idle_fence = fence
        # A single admission notice per pass, shared by native and summary fallback.
        # Capture the gateway's destination before any later turn rebinds the cached agent.
        notify = getattr(agent, "_native_idle_start_callback", None)
        notified = False
        def on_idle_start():
            nonlocal notified
            if (notified or generation.is_set()
                    or generation is not getattr(agent, "_native_idle_cancel", None)):
                return
            notified = True
            if callable(notify):
                try:
                    notify()
                except Exception:
                    logger.debug("Native idle start notice failed: session=%s", sid, exc_info=True)
        try:
            ok = attempt(agent, messages, getattr(agent, "_cached_system_prompt", "") or "", pressure,
                         phase="idle", expected_watermark=watermark, commit_fence=fence,
                         on_idle_start=on_idle_start)
        finally:
            if getattr(agent, "_native_idle_fence", None) is fence:
                agent._native_idle_fence = None
        if not ok and not getattr(agent, "_native_maintenance_abort_fallback", False) and not generation.is_set():
            return _ordinary_idle_fallback(agent, messages, sid, watermark, generation, pressure,
                                           on_idle_start=on_idle_start)
        return bool(ok)
    except Exception:
        logger.warning("Native idle maintenance declined for session=%s", sid, exc_info=True)
        return False


def _ordinary_idle_fallback(agent: Any, messages: list[dict], sid: str, watermark: int,
                            generation: threading.Event, pressure: int, *, on_idle_start: Any = None) -> bool:
    """Use the existing in-place compressor, never rotate or publish a stale idle result."""
    db = agent._session_db
    if (generation.is_set() or generation is not getattr(agent, "_native_idle_cancel", None)
            or db.get_active_message_watermark(sid) != watermark
            or not getattr(agent, "compression_in_place", True)):
        return False
    from agent.conversation_compression import CompressionCommitFence
    fence = CompressionCommitFence(total_ceiling_seconds=300)
    agent._native_idle_fence = fence
    agent._native_idle_summary_fallback = True
    agent._native_idle_fallback_watermark = watermark
    try:
        # archive_and_compact holds its own durable compression lock; its
        # transactional exact-watermark + turn-lease guard fences the commit.
        if generation.is_set() or generation is not getattr(agent, "_native_idle_cancel", None):
            return False
        if on_idle_start is not None:
            on_idle_start()
        compressed, _ = agent._compress_context(
            messages, getattr(agent, "_cached_system_prompt", "") or "",
            approx_tokens=pressure, commit_fence=fence, force=True,
        )
        if compressed is messages or not getattr(agent, "_last_compaction_in_place", False):
            return False
        # TUI keeps this exact list as its next-turn history; gateway reloads DB.
        live = getattr(agent, "_session_messages", None)
        if isinstance(live, list) and not generation.is_set():
            live[:] = compressed
        if not generation.is_set():
            agent._native_maintenance_pending_followup = "idle_summary"
        logger.info("Native idle ordinary fallback committed in place: session=%s", sid)
        return True
    except Exception:
        logger.warning("Native idle ordinary fallback failed: session=%s", sid, exc_info=True)
        return False
    finally:
        if getattr(agent, "_native_idle_fence", None) is fence:
            agent._native_idle_fence = None
        agent._native_idle_summary_fallback = False
        agent._native_idle_fallback_watermark = None
