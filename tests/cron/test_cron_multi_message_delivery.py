"""Regression coverage for explicit multi-message cron delivery."""

from unittest.mock import MagicMock, patch

from cron import scheduler_delivery as scheduler


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


def test_deliver_result_extracts_per_message_forum_titles(monkeypatch):
    calls = []

    def fake_deliver_single(
        job, content, adapters=None, loop=None, message_title=None
    ):
        calls.append((content, message_title))
        return None

    monkeypatch.setattr(scheduler, "_deliver_single_result", fake_deliver_single)

    content = (
        "[CRON_MESSAGE_TITLE] Downloaded video is missing\n"
        "Feedback report one\n"
        "[CRON_MESSAGE_BREAK]\n"
        "[CRON_MESSAGE_TITLE] Subscription cancellation help\n"
        "Feedback report two"
    )
    assert scheduler._deliver_result({"id": "job-1"}, content) is None
    assert calls == [
        ("Feedback report one", "Downloaded video is missing"),
        ("Feedback report two", "Subscription cancellation help"),
    ]


def test_extract_cron_delivery_title_preserves_unmarked_content():
    content = "Feedback report\n[CRON_MESSAGE_TITLE] literal later line"
    assert scheduler._extract_cron_delivery_title(content) == (None, content)


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

    with patch("cron.scheduler_delivery._seed_cron_thread_session", return_value=True) as seed:
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
    with patch("cron.scheduler_delivery._seed_cron_thread_session") as seed:
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


def test_discord_thread_seed_mirrors_into_exact_created_session():
    adapter = MagicMock()
    entry = MagicMock()
    entry.session_id = "seeded-session"
    adapter._session_store.get_or_create_session.return_value = entry

    with patch("gateway.mirror.mirror_to_session", return_value=True) as mirror:
        seeded = scheduler._seed_cron_thread_session(
            {"id": "job-1", "name": "Forum cases"},
            adapter,
            "discord",
            "1510950505835270144",
            "1538381421289148417",
            "Feedback case one",
        )

    assert seeded is True
    mirror.assert_called_once_with(
        "discord",
        "1510950505835270144",
        "[Cron delivery: Forum cases]\nFeedback case one",
        source_label="cron",
        thread_id="1538381421289148417",
        user_id="system:cron",
        role="user",
        session_id="seeded-session",
    )


def test_multi_message_forum_delivery_seeds_the_real_reply_sessions(tmp_path, monkeypatch):
    """Exercise the facade → router → native adapter → real transcript path."""
    import asyncio
    from concurrent.futures import Future
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from cron.scheduler import _deliver_result
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.session import SessionSource, SessionStore
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="test")})
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("cron.scheduler.load_config", lambda: {"cron": {"wrap_response": False}})
    store = SessionStore(tmp_path / "sessions", config)
    adapter = DiscordAdapter(config.platforms[Platform.DISCORD])
    adapter._session_store = store
    posts = []

    async def create_thread(*, name, content):
        thread_id = str(200 + len(posts))
        posts.append((thread_id, name, content))
        return SimpleNamespace(
            thread=SimpleNamespace(id=int(thread_id), send=AsyncMock()),
            message=SimpleNamespace(id=int(thread_id)),
        )

    forum = SimpleNamespace(id=100, create_thread=create_thread)
    adapter._is_forum_parent = lambda channel: channel is forum
    adapter._client = SimpleNamespace(get_channel=lambda cid: forum, fetch_channel=AsyncMock())
    loop = MagicMock()
    loop.is_running.return_value = True

    def run_coro(coro, loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    monkeypatch.setattr("asyncio.run_coroutine_threadsafe", run_coro)
    job = {"id": "forum-cases", "name": "Forum cases", "deliver": "discord:100",
           "attach_to_session": True}
    content = ("[CRON_MESSAGE_TITLE] First case\nFirst brief\n[CRON_MESSAGE_BREAK]\n"
               "[CRON_MESSAGE_TITLE] Second case\nSecond brief")
    assert _deliver_result(job, content, adapters={Platform.DISCORD: adapter}, loop=loop) is None
    assert [(title, body) for _, title, body in posts] == [
        ("First case", "First brief"), ("Second case", "Second brief")]
    for thread_id, _, body in posts:
        source = SessionSource(platform=Platform.DISCORD, chat_id=thread_id,
                               thread_id=thread_id, chat_type="thread", user_id="human")
        entry = store.get_or_create_session(source)
        assert [(m["role"], m["content"]) for m in store.load_transcript(entry.session_id)] == [
            ("user", "[Cron delivery: Forum cases]\n" + body)]


def test_external_worker_hands_off_all_parts_under_one_execution(tmp_path, monkeypatch):
    import cron.delivery_queue as queue

    content = "First brief\n[CRON_MESSAGE_BREAK]\nSecond brief"
    job = {"id": "job-1", "execution_id": "exec-1", "deliver": "discord:100"}
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", "exec-1")
    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "delivery.db")

    def enqueue_without_wait(execution, job, content, *, for_failure=False):
        queue.enqueue(execution, job, content, for_failure=for_failure)
        return None

    monkeypatch.setattr(queue, "enqueue_and_wait", enqueue_without_wait)
    assert scheduler._deliver_result(job, content) is None
    record = queue.claim_next()
    assert record["content"] == content
    assert queue.claim_next() is None
