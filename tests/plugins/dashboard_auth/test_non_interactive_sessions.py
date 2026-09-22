"""Hidden Basic sessions retain lifecycle support without public login entry points."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth import native_flow
from hermes_cli.dashboard_auth.routes import router, _reset_password_rate_limit
from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password


@pytest.mark.parametrize("visible_count", [0, 1, 2])
def test_hidden_session_is_not_a_native_login_candidate(visible_count):
    clear_providers()
    _reset_password_rate_limit()
    hidden = BasicAuthProvider(
        username="machine", password_hash=hash_password("secret"), secret=b"s" * 32,
        interactive_login=False)
    register_provider(hidden)
    for index in range(visible_count):
        visible = BasicAuthProvider(
            username="owner", password_hash=hidden._password_hash, secret=b"v" * 32)
        visible.name = f"visible-{index}"
        register_provider(visible)
    app = FastAPI()
    app.include_router(router)
    params = {
        "code_challenge": "a" * 43, "code_challenge_method": "S256",
        "redirect_uri": "http://127.0.0.1:53999/cb", "state": "desktop-state"}
    native_flow._reset_for_tests()
    try:
        with TestClient(app, base_url="https://gateway.example", follow_redirects=False) as client:
            rejected = client.get("/auth/native/authorize", params={**params, "provider": "basic"})
            assert rejected.status_code == 400
            assert "set-cookie" not in rejected.headers
            response = client.get("/auth/native/authorize", params=params)
            if visible_count == 0:
                assert response.status_code == 404
                assert client.get("/api/auth/providers").status_code == 503
            elif visible_count == 1:
                assert response.status_code == 302
                assert response.headers["location"] == "/login"
                from hermes_cli.dashboard_auth.cookies import parse_pkce_payload
                payload = parse_pkce_payload(next(iter(client.cookies.values())))
                assert payload["provider"] == "visible-0"
                assert payload["broker"]
            else:
                assert response.status_code == 200
                assert "provider=basic" not in response.text
                assert "provider=visible-0" in response.text
                assert "provider=visible-1" in response.text
                assert "set-cookie" not in response.headers
        # Disabling login does not disable machine-minted session verification or rotation.
        session = hidden._mint_session("machine")
        assert hidden.verify_session(access_token=session.access_token).user_id == "machine"
        refreshed = hidden.refresh_session(refresh_token=session.refresh_token)
        assert hidden.verify_session(access_token=refreshed.access_token).user_id == "machine"
    finally:
        clear_providers()
        native_flow._reset_for_tests()
