"""Codex Image Proxy image generation backend.

Talks to a local (or remote) ``hermes-codex-oauth-proxy`` that exposes an
OpenAI-compatible ``POST /v1/images/generations`` endpoint and returns
``data[0].b64_json``. This lets users who run the Codex OAuth proxy generate
images over plain HTTP without configuring the global ``OPENAI_API_KEY`` /
``OPENAI_BASE_URL`` (which would also re-route every other OpenAI call).

Mirrors the ``openai`` plugin's tier catalog: three virtual model IDs that all
hit the same underlying API model (``gpt-image-2``) with a different
``quality`` knob:

    gpt-image-2-low     ~15s   fastest, good for iteration
    gpt-image-2-medium  ~40s   default — balanced
    gpt-image-2-high    ~2min  slowest, highest fidelity

Output is base64 JSON → saved under ``$HERMES_HOME/cache/images/``.

Base URL precedence (first hit wins):

1. ``CODEX_IMAGE_PROXY_BASE_URL`` env var
2. ``image_gen.codex-image-proxy.base_url`` in ``config.yaml``
3. :data:`DEFAULT_BASE_URL` — ``http://127.0.0.1:8088/v1``

API key precedence (first hit wins, optional — the proxy may run without auth):

1. ``CODEX_IMAGE_PROXY_API_KEY`` env var
2. ``CODEX_PROXY_API_KEY`` env var (shared with the chat proxy)

Tier selection precedence (first hit wins):

1. ``CODEX_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.codex-image-proxy.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
4. :data:`DEFAULT_MODEL` — ``gpt-image-2-medium``
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog — mirrors the ``openai`` plugin so the picker UX is identical.
# ---------------------------------------------------------------------------

API_MODEL = "gpt-image-2"

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-2-low": {
        "display": "GPT Image 2 (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-2-medium": {
        "display": "GPT Image 2 (Medium)",
        "speed": "~40s",
        "strengths": "Balanced — default",
        "quality": "medium",
    },
    "gpt-image-2-high": {
        "display": "GPT Image 2 (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
}

DEFAULT_MODEL = "gpt-image-2-medium"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}

DEFAULT_BASE_URL = "http://127.0.0.1:8088/v1"

# Generous timeout — gpt-image-2 high quality can take ~2min, and the proxy
# adds OAuth refresh / upstream latency on top.
_REQUEST_TIMEOUT = 300.0


# ---------------------------------------------------------------------------
# Config + env helpers
# ---------------------------------------------------------------------------


def _load_image_gen_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_base_url() -> str:
    """Resolve the proxy base URL (env → config → default)."""
    env_override = os.environ.get("CODEX_IMAGE_PROXY_BASE_URL")
    if env_override and env_override.strip():
        return env_override.strip()

    cfg = _load_image_gen_config()
    sub = cfg.get("codex-image-proxy")
    if isinstance(sub, dict):
        value = sub.get("base_url")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return DEFAULT_BASE_URL


def _resolve_api_key() -> Optional[str]:
    """Resolve the proxy API key, or None when the proxy needs no auth.

    Falls back to ``CODEX_PROXY_API_KEY`` so the image proxy can share the
    same key as the chat proxy when an operator sets only one.
    """
    for key in ("CODEX_IMAGE_PROXY_API_KEY", "CODEX_PROXY_API_KEY"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which tier to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("CODEX_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_image_gen_config()
    sub = cfg.get("codex-image-proxy") if isinstance(cfg.get("codex-image-proxy"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(sub, dict):
        value = sub.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CodexImageProxyImageGenProvider(ImageGenProvider):
    """gpt-image-2 routed through a local/remote hermes-codex-oauth-proxy."""

    @property
    def name(self) -> str:
        return "codex-image-proxy"

    @property
    def display_name(self) -> str:
        return "Codex Image Proxy"

    def is_available(self) -> bool:
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        # The proxy may run unauthenticated, so a configured base URL is enough
        # on its own; an explicit API key also qualifies.
        if _resolve_api_key():
            return True
        return bool(_resolve_base_url())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "varies",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Codex Image Proxy",
            "badge": "local",
            "tag": "gpt-image-2 via a hermes-codex-oauth-proxy — no OPENAI_API_KEY",
            "env_vars": [
                {
                    "key": "CODEX_IMAGE_PROXY_BASE_URL",
                    "prompt": "Codex image proxy base URL",
                    "secret": False,
                    "default": DEFAULT_BASE_URL,
                },
                {
                    "key": "CODEX_IMAGE_PROXY_API_KEY",
                    "prompt": "Codex image proxy API key (optional — leave blank if the proxy has no auth)",
                    "secret": True,
                },
            ],
            "post_setup_hint": (
                "Point this at your running hermes-codex-oauth-proxy. The "
                f"default is {DEFAULT_BASE_URL}. An API key is only needed if "
                "the proxy enforces auth."
            ),
        }

    def capabilities(self) -> Dict[str, Any]:
        # The proxy currently exposes OpenAI-compatible image generations only.
        # Keep the dynamic tool schema honest instead of accepting source images
        # and silently ignoring them.
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if (isinstance(image_url, str) and image_url.strip()) or reference_image_urls:
            return error_response(
                error=(
                    "This model is not capable of image-to-image / editing. "
                    "Please provide a text-only prompt (drop image_url and "
                    "reference_image_urls)."
                ),
                error_type="modality_unsupported",
                provider="codex-image-proxy",
                aspect_ratio=aspect,
            )

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="codex-image-proxy",
                aspect_ratio=aspect,
            )

        try:
            import httpx
        except ImportError:
            return error_response(
                error="httpx Python package not installed (pip install httpx)",
                error_type="missing_dependency",
                provider="codex-image-proxy",
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])
        base_url = _resolve_base_url()
        api_key = _resolve_api_key()

        endpoint = f"{base_url.rstrip('/')}/images/generations"
        payload: Dict[str, Any] = {
            "model": API_MODEL,
            "prompt": prompt,
            "size": size,
            "quality": meta["quality"],
            "n": 1,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = httpx.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500] if exc.response is not None else ""
            status = exc.response.status_code if exc.response is not None else "?"
            logger.debug("Codex image proxy returned HTTP %s", status, exc_info=True)
            return error_response(
                error=f"Codex image proxy returned HTTP {status}: {body}",
                error_type="api_error",
                provider="codex-image-proxy",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except httpx.HTTPError as exc:
            logger.debug("Codex image proxy request failed", exc_info=True)
            return error_response(
                error=f"Codex image proxy request failed: {exc}",
                error_type="api_error",
                provider="codex-image-proxy",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            body = response.json()
        except ValueError as exc:
            return error_response(
                error=f"Codex image proxy returned non-JSON response: {exc}",
                error_type="invalid_response",
                provider="codex-image-proxy",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or not data:
            return error_response(
                error="Codex image proxy returned no image data",
                error_type="empty_response",
                provider="codex-image-proxy",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0] if isinstance(data[0], dict) else {}
        b64 = first.get("b64_json")
        url = first.get("url")

        if b64:
            try:
                saved_path = save_b64_image(b64, prefix=f"codex_image_proxy_{tier_id}")
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="codex-image-proxy",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            # Defensive — the proxy returns b64_json today, but cache the bytes
            # locally if it ever hands back an ephemeral URL so downstream
            # delivery never races an expiring link.
            try:
                saved_path = save_url_image(url, prefix=f"codex_image_proxy_{tier_id}")
            except Exception as exc:
                logger.warning(
                    "Codex image proxy URL %s could not be cached (%s); falling back to bare URL.",
                    url,
                    exc,
                )
                image_ref = url
            else:
                image_ref = str(saved_path)
        else:
            return error_response(
                error="Codex image proxy response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="codex-image-proxy",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=image_ref,
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="codex-image-proxy",
            extra={
                "size": size,
                "quality": meta["quality"],
                "base_url": base_url,  # base URL is not a secret
            },
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — register the Codex-image-proxy backend."""
    ctx.register_image_gen_provider(CodexImageProxyImageGenProvider())
