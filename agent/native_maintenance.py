"""Opt-in native-first maintenance for eligible Codex Responses sessions.

A maintenance response is never a chat turn: only a completed, nonempty compaction item
is admitted. Raw transcript rows remain untouched; the opaque item is a sidecar on the
latest assistant row and can be replayed after a cache eviction or process restart.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def _prefix_digest(messages: list[dict]) -> str:
    from agent.usage_anchor import message_fingerprint
    return hashlib.sha256("|".join(message_fingerprint(m) or "" for m in messages).encode()).hexdigest()


def eligible(agent: Any) -> bool:
    from agent.codex_responses_adapter import classify_responses_route
    from agent.native_compaction import native_compaction_context_management, sol_native_route

    if getattr(agent, "compression_aux_native", None) is not None:
        return sol_native_route(agent)
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
            turn_lease_holder: str | None = None, commit_fence: Any = None) -> bool:
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
        from agent.codex_responses_adapter import (
            classify_responses_route, _chat_messages_to_responses_input,
            _classify_responses_issuer, _wire_model_identity,
        )
        from agent.native_compaction import sol_native_route
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
        sol_route = sol_native_route(agent)
        if not sol_route and "context_management" not in kwargs:
            return False
        if sol_route:
            # The ordinary builder supplies the frozen instructions/tools/cache key. Rebuild
            # only its input for the actual destination model; source stamps stay untouched.
            kwargs["model"] = "gpt-6-sol"
            kwargs["input"] = _chat_messages_to_responses_input(
                api_prefix, current_issuer_kind="codex_backend", current_issuer_model="gpt-6-sol",
                native_compaction_eligible=True, approved_sol_pair="maintenance",
                preserve_reasoning=bool(getattr(agent, "compression_aux_preserve_reasoning", False)),
                replay_encrypted_reasoning=bool(getattr(agent, "_codex_reasoning_replay_enabled", True)),
            )
        # The full frozen instructions/tools prefix comes from the ordinary builder.
        # A low maintenance-only threshold forces a real checkpoint even below the
        # automatic native threshold; it never changes the next normal call's policy.
        kwargs["context_management"] = [{"type": "compaction", "compact_threshold": 1024}]
        kwargs["tool_choice"] = "none"
        # The entire ordinary request is the frozen prefix. A maintenance-only
        # suffix makes an assistant/tool-ended transcript a closed, inert request.
        kwargs["input"] = list(kwargs["input"]) + [{"role": "user", "content":
            "Prepare the handoff. Reply only READY."}]
        if sol_route:
            from agent.auxiliary_client import _effective_aux_timeout
            kwargs["timeout"] = _effective_aux_timeout("compression", None)
        else:
            kwargs["timeout"] = min(float(kwargs.get("timeout") or 300), 300.0)
        kwargs = agent._get_transport().preflight_kwargs(
            kwargs, allow_stream=True, is_github_responses=route.is_github_responses,
            sanitize_harmony_tokens=route.is_codex_backend,
        )
        kwargs = _sanitize_consumer_codex_request(agent, kwargs)
        kwargs["stream"] = True
        client = agent._create_request_openai_client(reason="native_maintenance", api_kwargs=kwargs)
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
        if commit_fence is not None and not commit_fence.begin_commit():
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
            carrier["codex_reasoning_items"] = list(carrier.get("codex_reasoning_items") or []) + [checkpoint]
        finally:
            if commit_fence is not None:
                commit_fence.finish_commit()
        try:
            note = getattr(agent.context_compressor, "note_native_compaction_checkpoint", None)
            if callable(note):
                note()
            from agent.usage_anchor import set_usage_anchor
            set_usage_anchor(agent, None)
        except Exception:
            logger.warning("Native checkpoint committed but local usage latch failed", exc_info=True)
        agent._native_maintenance_pending_followup = phase
        outcome, reason = "native", "checkpoint_committed"
        return True
    except Exception as exc:
        reason = type(exc).__name__
        logger.warning("Native maintenance failed (%s, session=%s, reason=%s)", phase, sid, reason, exc_info=True)
        return False
    finally:
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
