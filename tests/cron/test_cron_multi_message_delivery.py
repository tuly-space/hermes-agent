"""Regression coverage for explicit multi-message cron delivery."""

from unittest.mock import MagicMock, patch

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


def test_single_explicit_discord_target_requires_per_job_opt_in():
    targets = [{"platform": "discord", "chat_id": "1510950505835270144"}]

    assert scheduler._single_explicit_discord_target(
        {
            "deliver": "discord:1510950505835270144",
            "attach_to_session": True,
        },
        targets,
    ) is True
    assert scheduler._single_explicit_discord_target(
        {"deliver": "discord:1510950505835270144"},
        targets,
    ) is False
    assert scheduler._single_explicit_discord_target(
        {
            "deliver": "discord:1510950505835270144,telegram",
            "attach_to_session": True,
        },
        targets + [{"platform": "telegram", "chat_id": "123"}],
    ) is False


def test_explicit_discord_forum_result_seeds_created_thread_session():
    adapter = MagicMock()
    job = {"id": "job-1", "attach_to_session": True}

    with patch("cron.scheduler._seed_cron_thread_session") as seed:
        seeded = scheduler._seed_explicit_discord_forum_session(
            job,
            adapter,
            "discord",
            "1510950505835270144",
            {"thread_id": "1530000000000000001", "message_ids": ["m1"]},
            "Feedback case one",
            enabled=True,
        )

    assert seeded is True
    seed.assert_called_once_with(
        job,
        adapter,
        "discord",
        "1510950505835270144",
        "1530000000000000001",
        "Feedback case one",
    )


def test_explicit_discord_non_forum_result_does_not_seed_session():
    with patch("cron.scheduler._seed_cron_thread_session") as seed:
        seeded = scheduler._seed_explicit_discord_forum_session(
            {"id": "job-1", "attach_to_session": True},
            MagicMock(),
            "discord",
            "1510950505835270144",
            {"message_ids": ["m1"]},
            "ordinary channel delivery",
            enabled=True,
        )

    assert seeded is False
    seed.assert_not_called()
