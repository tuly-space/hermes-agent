"""Regression coverage for explicit multi-message cron delivery."""

from cron import scheduler


def test_deliver_result_splits_explicit_message_breaks(monkeypatch):
    calls = []
    adapters = object()
    loop = object()

    def fake_deliver_single(job, content, adapters=None, loop=None):
        calls.append((job, content, adapters, loop))
        return None

    monkeypatch.setattr(scheduler, "_deliver_single_result", fake_deliver_single)

    job = {"id": "job-1"}
    error = scheduler._deliver_result(
        job,
        "Case 1\n\n[CRON_MESSAGE_BREAK]\n\nCase 2\n",
        adapters=adapters,
        loop=loop,
    )

    assert error is None
    assert calls == [
        (job, "Case 1", adapters, loop),
        (job, "Case 2", adapters, loop),
    ]


def test_deliver_result_keeps_normal_and_inline_marker_content_single(monkeypatch):
    calls = []

    def fake_deliver_single(job, content, adapters=None, loop=None):
        calls.append(content)
        return None

    monkeypatch.setattr(scheduler, "_deliver_single_result", fake_deliver_single)

    assert scheduler._deliver_result(
        {"id": "job-1"},
        "Explain [CRON_MESSAGE_BREAK] literally in this report.",
    ) is None
    assert calls == ["Explain [CRON_MESSAGE_BREAK] literally in this report."]


def test_deliver_result_ignores_empty_segments(monkeypatch):
    calls = []

    def fake_deliver_single(job, content, adapters=None, loop=None):
        calls.append(content)
        return None

    monkeypatch.setattr(scheduler, "_deliver_single_result", fake_deliver_single)

    assert scheduler._deliver_result(
        {"id": "job-1"},
        "[CRON_MESSAGE_BREAK]\nCase 1\n[CRON_MESSAGE_BREAK]\n   ",
    ) is None
    assert calls == ["Case 1"]


def test_deliver_result_aggregates_part_errors_and_continues(monkeypatch):
    calls = []

    def fake_deliver_single(job, content, adapters=None, loop=None):
        calls.append(content)
        return "network down" if content == "Case 1" else None

    monkeypatch.setattr(scheduler, "_deliver_single_result", fake_deliver_single)

    error = scheduler._deliver_result(
        {"id": "job-1"},
        "Case 1\n[CRON_MESSAGE_BREAK]\nCase 2",
    )

    assert calls == ["Case 1", "Case 2"]
    assert error == "message 1: network down"
