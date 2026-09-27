"""Gateway never detaches or adopts a second automatic compressor.

The complete transcript goes to the AIAgent assembled-request preflight when
compression is enabled. The disabled-compression safety bound is the only
Gateway-side truncation; native timeout/cancellation/CAS live in
``tests/agent/test_unified_auto_compaction.py``.
"""

import pytest

from hermes_state import SessionDB
from tests.gateway.test_session_hygiene import (
    _make_bound_probe_transcript, _make_cooldown_runner, _make_history,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("hard_limit,history", [
    (5, _make_history(12, content_size=40)),
    (20, _make_bound_probe_transcript()),
])
async def test_gateway_turn_passes_full_history_without_detached_worker(
    monkeypatch, tmp_path, hard_limit, history,
):
    """Old turn-hold expiry cannot release a worker to commit behind the live turn."""
    class DetachedAgentMustNotExist:
        def __init__(self, **kwargs):
            pytest.fail("gateway built a detached compression agent")

    sid = "unified-gateway"
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(sid, "telegram")
        runner, adapter, event = _make_cooldown_runner(
            monkeypatch, tmp_path, DetachedAgentMustNotExist, db, sid
        )
        runner.session_store.load_transcript.return_value = history
        (tmp_path / "config.yaml").write_text(
            "compression:\n  enabled: true\n"
            f"  hygiene_hard_message_limit: {hard_limit}\n"
            "  hygiene_max_turn_hold_seconds: 0.01\n"
        )
        assert await runner._handle_message(event) == "ok"
        handed_off = runner._run_agent.call_args.kwargs["history"]
        assert handed_off is history  # not clipped, copied, or replaced by an idle summary
        runner.session_store.rewrite_transcript.assert_not_called()
        assert db.get_compression_failure_cooldown(sid) is None
        assert not any("deferred" in sent["content"].lower() or
                       "took too long" in sent["content"].lower() for sent in adapter.sent)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_compression_disabled_keeps_gateway_payload_bound(monkeypatch, tmp_path):
    """With no agent-side compression, the old fail-closed head/tail bound remains."""
    class DetachedAgentMustNotExist:
        def __init__(self, **kwargs):
            pytest.fail("gateway built a detached compression agent")

    sid = "disabled-gateway"
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(sid, "telegram")
        runner, _adapter, event = _make_cooldown_runner(
            monkeypatch, tmp_path, DetachedAgentMustNotExist, db, sid
        )
        history = _make_bound_probe_transcript()
        runner.session_store.load_transcript.return_value = history
        (tmp_path / "config.yaml").write_text(
            "compression:\n  enabled: false\n  hygiene_hard_message_limit: 20\n"
        )
        assert await runner._handle_message(event) == "ok"
        payload = runner._run_agent.call_args.kwargs["history"]
        assert len(payload) <= 20
        assert payload[0] is history[0] and payload[-1] is history[-1]
        assert payload[1].get("role") != "tool"
        assert len(history) > len(payload)  # only the model input is bounded
        runner.session_store.rewrite_transcript.assert_not_called()
    finally:
        db.close()
