
"""Integration tests for hallucination-lens FastAPI service wrapper."""

from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from hallucination_lens import api
from hallucination_lens.scorer import FaithfulnessResult, SentenceScore


class FakeScorer:
    """Deterministic scorer stub for API tests."""

    model_name = "fake-embedding"

    def faithfulness_score(self, context: str, response: str, threshold: float | None = None) -> FaithfulnessResult:
        """Return predictable single-score payload for endpoint contract testing."""

        active_threshold = 0.6 if threshold is None else threshold
        return FaithfulnessResult(
            score=0.82,
            verdict="faithful" if 0.82 >= active_threshold else "hallucinated",
            threshold=active_threshold,
            sentence_scores=[SentenceScore(sentence=response, max_similarity=0.82)],
        )

    def batch_faithfulness_scores(
        self,
        pairs: list[tuple[str, str]],
        threshold: float | None = None,
    ) -> list[FaithfulnessResult]:
        """Return deterministic batch result list aligned to input order."""

        active_threshold = 0.6 if threshold is None else threshold
        return [
            FaithfulnessResult(
                score=0.82,
                verdict="faithful" if 0.82 >= active_threshold else "hallucinated",
                threshold=active_threshold,
                sentence_scores=[SentenceScore(sentence=response, max_similarity=0.82)],
            )
            for _, response in pairs
        ]


class FailingLimiter:
    """Rate limiter stub that simulates backend outages."""

    def allow(self, key: str, limit: int) -> bool:  # noqa: ARG002
        """Raise runtime error to emulate unavailable distributed backend."""

        raise RuntimeError("rate limiter backend unavailable")

    def is_healthy(self) -> bool:
        """Report unhealthy backend status for readiness checks."""

        return False


def _client_with_fake_scorer(monkeypatch):
    """Create API test client with fake scorer to avoid model downloads."""

    monkeypatch.setattr(api, "get_scorer", lambda: FakeScorer())
    return TestClient(api.app)


def _override_settings(monkeypatch, **changes):
    """Apply temporary runtime setting overrides during endpoint tests."""

    monkeypatch.setattr(api, "settings", replace(api.settings, **changes))


def test_health_endpoint_exposes_limits_and_version(monkeypatch):
    """Health endpoint should return runtime metadata for observability."""

    client = _client_with_fake_scorer(monkeypatch)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["max_batch_items"] > 0


def test_ready_endpoint_reports_model_loaded(monkeypatch):
    """Readiness endpoint should confirm model backend initialization status."""

    client = _client_with_fake_scorer(monkeypatch)
    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["model_loaded"] is True


def test_score_endpoint_returns_contract(monkeypatch):
    """Single score endpoint should return verdict and sentence-level evidence."""

    client = _client_with_fake_scorer(monkeypatch)
    response = client.post(
        "/score",
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "faithful"
    assert payload["model_name"] == "fake-embedding"
    assert len(payload["sentence_scores"]) == 1


def test_batch_endpoint_returns_aggregate(monkeypatch):
    """Batch endpoint should return aggregate fields and per-item results."""

    client = _client_with_fake_scorer(monkeypatch)
    response = client.post(
        "/batch",
        json={
            "items": [
                {
                    "context": "Paris is the capital of France.",
                    "response": "Paris is in France.",
                },
                {
                    "context": "Paris is the capital of France.",
                    "response": "Paris is in France.",
                },
            ],
            "threshold": 0.6,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["item_count"] == 2
    assert len(payload["results"]) == 2
    assert payload["verdict_counts"]["faithful"] == 2


def test_batch_threshold_outside_governance_is_rejected(monkeypatch):
    """Threshold values outside governance range should return 422 error payload."""

    client = _client_with_fake_scorer(monkeypatch)
    response = client.post(
        "/batch",
        json={
            "items": [
                {
                    "context": "Paris is the capital of France.",
                    "response": "Paris is in France.",
                }
            ],
            "threshold": 0.95,
        },
    )

    assert response.status_code == 422
    assert "governance limits" in response.json()["detail"]


def test_score_requires_api_key_when_configured(monkeypatch):
    """Score endpoint should return 401 without valid API key when enabled."""

    _override_settings(monkeypatch, api_key="secret-key")
    client = _client_with_fake_scorer(monkeypatch)

    unauthorized = client.post(
        "/score",
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )
    assert unauthorized.status_code == 401

    authorized = client.post(
        "/score",
        headers={"X-API-Key": "secret-key"},
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )
    assert authorized.status_code == 200


def test_gateway_auth_is_enforced_before_api_key(monkeypatch):
    """Gateway auth should be checked before API key auth on scoring endpoints."""

    _override_settings(
        monkeypatch,
        gateway_auth_enabled=True,
        gateway_auth_secret="gateway-secret",
        api_key="secret-key",
    )
    client = _client_with_fake_scorer(monkeypatch)

    gateway_missing = client.post(
        "/score",
        headers={"X-API-Key": "secret-key"},
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )
    assert gateway_missing.status_code == 401
    assert "Gateway authentication failed" in gateway_missing.json()["detail"]

    api_key_missing = client.post(
        "/score",
        headers={"X-Gateway-Auth": "gateway-secret"},
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )
    assert api_key_missing.status_code == 401

    success = client.post(
        "/score",
        headers={
            "X-Gateway-Auth": "gateway-secret",
            "X-API-Key": "secret-key",
        },
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )
    assert success.status_code == 200


def test_gateway_principal_header_required_when_enabled(monkeypatch):
    """Gateway principal header should be required when policy is enabled."""

    _override_settings(
        monkeypatch,
        gateway_auth_enabled=True,
        gateway_auth_secret="gateway-secret",
        require_gateway_principal=True,
    )
    client = _client_with_fake_scorer(monkeypatch)

    missing_principal = client.post(
        "/score",
        headers={"X-Gateway-Auth": "gateway-secret"},
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )
    assert missing_principal.status_code == 401
    assert "Gateway principal header missing" in missing_principal.json()["detail"]

    with_principal = client.post(
        "/score",
        headers={
            "X-Gateway-Auth": "gateway-secret",
            "X-Principal-Id": "service-account-1",
        },
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )
    assert with_principal.status_code == 200


def test_health_stays_public_with_api_key_enabled(monkeypatch):
    """Health endpoint remains public for orchestrator liveness checks."""

    _override_settings(monkeypatch, api_key="secret-key")
    client = _client_with_fake_scorer(monkeypatch)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_request_body_over_limit_is_rejected(monkeypatch):
    """Requests that exceed configured body size should fail with 413."""

    _override_settings(monkeypatch, max_request_bytes=120)
    client = _client_with_fake_scorer(monkeypatch)
    response = client.post(
        "/score",
        json={
            "context": "Paris is the capital of France." * 8,
            "response": "Paris is in France." * 6,
            "threshold": 0.6,
        },
    )

    assert response.status_code == 413
    assert "Request body too large" in response.json()["detail"]


def test_security_headers_are_present(monkeypatch):
    """Middleware should attach baseline security headers to normal responses."""

    client = _client_with_fake_scorer(monkeypatch)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Content-Security-Policy"].startswith("default-src")


def test_ready_reports_unhealthy_when_rate_limiter_backend_is_unavailable(monkeypatch):
    """Readiness should fail when distributed rate-limiter backend is down."""

    monkeypatch.setattr(api, "limiter", FailingLimiter())
    client = _client_with_fake_scorer(monkeypatch)
    response = client.get("/ready")

    assert response.status_code == 503
    assert "rate limiter backend unavailable" in response.json()["detail"]


def test_score_returns_503_when_rate_limiter_backend_is_unavailable(monkeypatch):
    """Scoring endpoints should fail with 503 when limiter backend is unavailable."""

    monkeypatch.setattr(api, "limiter", FailingLimiter())
    client = _client_with_fake_scorer(monkeypatch)
    response = client.post(
        "/score",
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )

    assert response.status_code == 503
    assert "Rate limiter backend unavailable" in response.json()["detail"]
