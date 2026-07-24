"""Tests for the bundled ``codex-image-proxy`` image_gen plugin.

Targets the provider that calls a local/remote ``hermes-codex-oauth-proxy``
OpenAI-compatible ``POST /v1/images/generations`` endpoint over plain HTTP,
without touching the global ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import base64
import importlib
from pathlib import Path

import httpx
import pytest

# The plugin directory uses hyphens, which are not valid in the dotted-import
# form — load it via importlib so tests don't need to touch sys.path.
proxy_plugin = importlib.import_module("plugins.image_gen.codex-image-proxy")


# 1×1 transparent PNG — valid bytes for save_b64_image()
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json = json_body
        self.status_code = status_code
        self.text = "" if json_body is None else "body"

    def raise_for_status(self):
        return None

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Keep config resolution deterministic — no config.yaml overrides.
    monkeypatch.delenv("CODEX_IMAGE_PROXY_BASE_URL", raising=False)
    monkeypatch.delenv("CODEX_IMAGE_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_IMAGE_MODEL", raising=False)
    yield tmp_path


@pytest.fixture
def provider():
    return proxy_plugin.CodexImageProxyImageGenProvider()


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "codex-image-proxy"

    def test_display_name(self, provider):
        assert provider.display_name == "Codex Image Proxy"

    def test_default_model(self, provider):
        assert provider.default_model() == "gpt-image-2-medium"

    def test_list_models_three_tiers(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert ids == ["gpt-image-2-low", "gpt-image-2-medium", "gpt-image-2-high"]

    def test_setup_schema_exposes_base_url_and_secret_key(self, provider):
        schema = provider.get_setup_schema()
        keys = {ev["key"]: ev for ev in schema["env_vars"]}
        assert keys["CODEX_IMAGE_PROXY_BASE_URL"]["secret"] is False
        assert keys["CODEX_IMAGE_PROXY_API_KEY"]["secret"] is True

    def test_capabilities_are_text_only(self, provider):
        assert provider.capabilities() == {"modalities": ["text"], "max_reference_images": 0}


# ── Resolution helpers ───────────────────────────────────────────────────────


class TestResolution:
    def test_base_url_default(self):
        assert proxy_plugin._resolve_base_url() == proxy_plugin.DEFAULT_BASE_URL

    def test_base_url_env_override(self, monkeypatch):
        monkeypatch.setenv("CODEX_IMAGE_PROXY_BASE_URL", "http://example:9000/v1")
        assert proxy_plugin._resolve_base_url() == "http://example:9000/v1"

    def test_api_key_prefers_specific_then_fallback(self, monkeypatch):
        monkeypatch.setenv("CODEX_PROXY_API_KEY", "shared")
        assert proxy_plugin._resolve_api_key() == "shared"
        monkeypatch.setenv("CODEX_IMAGE_PROXY_API_KEY", "specific")
        assert proxy_plugin._resolve_api_key() == "specific"

    def test_api_key_none_when_unset(self):
        assert proxy_plugin._resolve_api_key() is None

    def test_model_env_override(self, monkeypatch):
        monkeypatch.setenv("CODEX_IMAGE_MODEL", "gpt-image-2-high")
        tier_id, meta = proxy_plugin._resolve_model()
        assert tier_id == "gpt-image-2-high"
        assert meta["quality"] == "high"


# ── Availability ─────────────────────────────────────────────────────────────


class TestAvailability:
    def test_available_with_default_base_url(self, provider):
        # No key, but a base URL always resolves to a default → available.
        assert provider.is_available() is True

    def test_available_with_api_key(self, provider, monkeypatch):
        monkeypatch.setenv("CODEX_IMAGE_PROXY_API_KEY", "k")
        assert provider.is_available() is True


# ── Generate ─────────────────────────────────────────────────────────────────


class TestGenerate:
    def test_invalid_argument_for_empty_prompt(self, provider):
        result = provider.generate("   ")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_image_input_returns_modality_unsupported(self, provider):
        result = provider.generate("edit this", image_url="https://example.com/source.png")
        assert result["success"] is False
        assert result["error_type"] == "modality_unsupported"
        assert result["provider"] == "codex-image-proxy"

    def test_generate_request_shape_and_b64_save(self, provider, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEX_IMAGE_PROXY_API_KEY", "secret-key")
        captured = {}

        def _fake_post(url, *, json, headers, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            return _FakeResponse({"data": [{"b64_json": _b64_png()}]})

        monkeypatch.setattr(httpx, "post", _fake_post)

        result = provider.generate("a cat", aspect_ratio="portrait")

        assert result["success"] is True
        assert result["provider"] == "codex-image-proxy"
        assert result["model"] == "gpt-image-2-medium"
        assert result["quality"] == "medium"
        assert result["size"] == "1024x1536"
        assert result["base_url"] == proxy_plugin.DEFAULT_BASE_URL

        # Request shape
        assert captured["url"] == "http://127.0.0.1:8088/v1/images/generations"
        assert captured["json"] == {
            "model": "gpt-image-2",
            "prompt": "a cat",
            "size": "1024x1536",
            "quality": "medium",
            "n": 1,
        }
        assert captured["headers"]["Authorization"] == "Bearer secret-key"
        user_agent = captured["headers"]["User-Agent"]
        assert user_agent.startswith("Mozilla/5.0 ")
        assert "Chrome/" in user_agent
        assert "python" not in user_agent.lower()
        assert "httpx" not in user_agent.lower()
        assert captured["headers"]["Accept"] == "application/json"

        # Saved file
        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        assert saved.name.startswith("codex_image_proxy_gpt-image-2-medium")

    def test_no_auth_header_without_api_key(self, provider, monkeypatch):
        captured = {}

        def _fake_post(url, *, json, headers, timeout):
            captured["headers"] = headers
            return _FakeResponse({"data": [{"b64_json": _b64_png()}]})

        monkeypatch.setattr(httpx, "post", _fake_post)

        result = provider.generate("a cat")
        assert result["success"] is True
        assert "Authorization" not in captured["headers"]

    def test_empty_data_returns_error(self, provider, monkeypatch):
        monkeypatch.setattr(
            httpx, "post",
            lambda *a, **kw: _FakeResponse({"data": []}),
        )
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_http_status_error_returns_api_error(self, provider, monkeypatch):
        import httpx

        def _fake_post(*a, **kw):
            request = httpx.Request("POST", "http://127.0.0.1:8088/v1/images/generations")
            response = httpx.Response(500, request=request, text="boom")
            raise httpx.HTTPStatusError("err", request=request, response=response)

        monkeypatch.setattr(httpx, "post", _fake_post)
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "500" in result["error"]

    def test_transport_error_returns_api_error(self, provider, monkeypatch):
        import httpx

        def _fake_post(*a, **kw):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "post", _fake_post)
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "api_error"


# ── Plugin entry point ───────────────────────────────────────────────────────


class TestRegistration:
    def test_register_calls_register_image_gen_provider(self):
        registered = []

        class _Ctx:
            def register_image_gen_provider(self, prov):
                registered.append(prov)

        proxy_plugin.register(_Ctx())
        assert len(registered) == 1
        assert registered[0].name == "codex-image-proxy"
