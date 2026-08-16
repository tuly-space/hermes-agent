"""Which title stage is allowed to spend a platform rename.

Titling is two-stage: a derived slice of the user's own words lands inline, and
the model's version replaces it a moment later. A local sidebar wants both. A
Discord thread or a Telegram topic wants only the second — renaming twice lands
on the same name at twice the cost, and Discord allows two channel renames per
ten minutes, so the throwaway can be the one that survives.
"""

from __future__ import annotations

import types

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner, TurnRunner
from gateway.session import SessionSource


def _attach(lane):
    """Attach the title callback for *lane* and return (callback, renames)."""
    renames: list = []
    source = types.SimpleNamespace(platform=Platform.DISCORD, chat_id="chan-1")

    runner = types.SimpleNamespace(
        _is_telegram_topic_lane=lambda src: lane == "telegram",
        _is_discord_auto_thread_lane=lambda src: lane == "discord",
        _is_relay_discord_channel_lane=lambda src: False,
        _schedule_telegram_topic_title_rename=(
            lambda src, sid, title: renames.append(title)
        ),
        _schedule_discord_semantic_thread_rename=(
            lambda src, sid, title: renames.append(title)
        ),
    )
    holder = types.SimpleNamespace(
        _runner=runner,
        _attach_session_title_callback=TurnRunner._attach_session_title_callback,
    )
    agent = types.SimpleNamespace(session_id="sess-1")
    holder._attach_session_title_callback(
        holder, agent, types.SimpleNamespace(source=source)
    )
    return agent._on_session_title, renames


@pytest.mark.parametrize("lane", ["telegram", "discord"])
def test_the_rename_waits_for_the_model_title(lane):
    callback, renames = _attach(lane)

    callback("fix the flaky auth test in log", "derived")
    assert renames == []

    callback("Fix flaky auth test", "llm")
    assert renames == ["Fix flaky auth test"]


@pytest.mark.asyncio
async def test_native_discord_rename_uses_only_native_adapter_kwargs():
    """Relay-only kwargs must not be sent to the native Discord adapter."""
    renames: list = []

    class NativeAdapter:
        async def rename_thread(
            self,
            thread_id,
            name,
            *,
            only_if_current_name=None,
        ):
            renames.append((thread_id, name, only_if_current_name))
            return True

    class Runner:
        _rename_discord_auto_thread_for_session_title = (
            GatewayRunner._rename_discord_auto_thread_for_session_title
        )
        _sanitize_discord_thread_title = GatewayRunner._sanitize_discord_thread_title

        def __init__(self):
            self.adapters = {Platform.DISCORD: NativeAdapter()}

        def _is_discord_auto_thread_lane(self, source):
            return True

        def _is_relay_discord_channel_lane(self, source):
            return False

        def _adapter_for_source(self, source):
            return self.adapters[source.platform]

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        auto_thread_created=True,
        auto_thread_initial_name="Original thread name",
    )

    await Runner()._rename_discord_auto_thread_for_session_title(
        source,
        "session-1",
        "Generated session title",
    )

    assert renames == [
        ("thread-1", "Generated session title", "Original thread name")
    ]
