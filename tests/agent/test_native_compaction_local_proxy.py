"""Local proxy capability resolution and request gating must agree."""
from types import SimpleNamespace

import pytest

from agent.native_compaction import (
    is_native_compaction_model,
    native_compaction_context_management,
    resolve_native_compaction_capabilities,
)

URL = "https://us-lrv03-sj4srl0f0-react-work.taila837d.ts.net/v1"


def make_agent(model="gpt-6-astra", url=URL, provider="custom:local-codex-proxy"):
    return SimpleNamespace(
        model=model, base_url=url, provider=provider,
        capabilities={},
        runtime_capabilities=resolve_native_compaction_capabilities(
            model=model, base_url=url, provider=provider,
        ),
        codex_responses_native_compaction=True, compression_enabled=True,
        codex_responses_compact_threshold=200_000, context_compressor=None,
    )


@pytest.mark.parametrize("model", ["gpt-6", "gpt-6-astra", "GPT-6-ASTRA", "gpt-6.1", "gpt-5.6-sol"])
@pytest.mark.parametrize("provider", ["custom:local-codex-proxy", "custom"])
def test_supported_family_through_verified_proxy(model, provider):
    agent = make_agent(model=model, provider=provider)
    assert agent.runtime_capabilities == {"native_compaction": True}
    assert native_compaction_context_management(agent, is_codex_backend=False) == [
        {"type": "compaction", "compact_threshold": 200_000}
    ]


@pytest.mark.parametrize("model", ["gpt-60", "gpt-6fake", "not-gpt-6", "gpt-5.2", "claude-sonnet-4-6"])
def test_unsupported_models_remain_denied(model):
    assert not is_native_compaction_model(model)
    agent = make_agent(model=model)
    assert agent.runtime_capabilities == {"native_compaction": False}
    assert native_compaction_context_management(agent, is_codex_backend=False) is None


@pytest.mark.parametrize("url", [
    "https://proxy.example/v1", URL.replace("https:", "http:"),
    URL.replace(".ts.net", ".ts.net.evil.com"), URL + "?redirect=1",
    URL.replace("/v1", ":8443/v1"), URL.replace("/v1", "/other"),
])
def test_other_routes_remain_denied(url):
    agent = make_agent(url=url)
    assert agent.runtime_capabilities == {"native_compaction": False}
    assert native_compaction_context_management(agent, is_codex_backend=False) is None


def test_other_provider_identity_remains_denied():
    agent = make_agent(provider="custom:other")
    assert agent.runtime_capabilities == {"native_compaction": False}


@pytest.mark.parametrize("attr,value", [
    ("runtime_capabilities", {"native_compaction": False}),
    ("compression_enabled", False),
    ("codex_responses_native_compaction", False),
    ("compression_checkpoint_required", True),
])
def test_safety_switches_still_win(attr, value):
    agent = make_agent()
    setattr(agent, attr, value)
    assert native_compaction_context_management(agent, is_codex_backend=False) is None


def test_switch_away_recomputes_denial_and_return_restores_support():
    agent = make_agent()
    for url, expected in [("https://proxy.example/v1", False), (URL, True)]:
        agent.base_url = url
        agent.runtime_capabilities = resolve_native_compaction_capabilities(
            model=agent.model, base_url=url, provider=agent.provider,
        )
        assert bool(native_compaction_context_management(agent, is_codex_backend=False)) is expected
