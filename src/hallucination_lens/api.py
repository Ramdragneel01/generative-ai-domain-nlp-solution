
"""FastAPI service wrapper for hallucination-lens scoring workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .rate_limit import InMemoryRateLimiter
from .scorer import HallucinationScorer
from .validators import validate_text_argument, validate_threshold


class SentenceScoreResponse(BaseModel):
    """Serializable sentence-level evidence score payload."""

    sentence: str
    max_similarity: float = Field(ge=0.0, le=1.0)


class ScoreRequest(BaseModel):
    """Request payload for single faithfulness scoring."""

    context: str = Field(min_length=5)
    response: str = Field(min_length=5)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class ScoreResponse(BaseModel):
    """Response payload for single faithfulness scoring."""

    score: float = Field(ge=0.0, le=1.0)
    verdict: str
    threshold: float = Field(ge=0.0, le=1.0)
    model_name: str
    sentence_scores: list[SentenceScoreResponse]
    duration_ms: float = Field(ge=0.0)
    request_id: str


class BatchScoreItem(BaseModel):
    """One context-response pair for batch scoring requests."""

    context: str = Field(min_length=5)
    response: str = Field(min_length=5)


class BatchScoreRequest(BaseModel):
    """Request payload for batch scoring endpoint."""

    items: list[BatchScoreItem] = Field(min_length=1, max_length=200)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class BatchScoreItemResponse(BaseModel):
    """Per-item scoring output in batch response."""

    index: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    verdict: str
    sentence_scores: list[SentenceScoreResponse]


class BatchScoreResponse(BaseModel):
    """Response payload for batch scoring endpoint."""

    item_count: int = Field(gt=0)
    average_score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    model_name: str
    verdict_counts: dict[str, int]
    results: list[BatchScoreItemResponse]
    duration_ms: float = Field(ge=0.0)
    request_id: str


class HealthResponse(BaseModel):
    """Health diagnostics payload for monitoring and deployments."""

    status: str
    timestamp: str
    app_version: str
    model_name: str
    default_threshold: float
    max_batch_items: int


settings = get_settings()
limiter = InMemoryRateLimiter(window_seconds=60)
_scorer: HallucinationScorer | None = None

REQUEST_COUNTER = Counter(
    "hallucination_lens_http_requests_total",
    "HTTP request count",
    labelnames=("method", "endpoint", "status"),
)
REQUEST_LATENCY = Histogram(
    "hallucination_lens_http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=("method", "endpoint"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)
SCORE_HISTOGRAM = Histogram(
    "hallucination_lens_score_distribution",
    "Distribution of hallucination-lens scores",
    buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
)
VERDICT_COUNTER = Counter(
    "hallucination_lens_verdict_total",
    "Count of faithful vs hallucinated verdicts",
    labelnames=("verdict",),
)


app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_scorer() -> HallucinationScorer:
    """Return lazily initialized scorer instance to avoid startup model overhead."""

    global _scorer
    if _scorer is None:
        _scorer = HallucinationScorer(
            model_name=settings.model_name,
            threshold=settings.default_threshold,
        )
    return _scorer


def _client_key(request: Request) -> str:
    """Build client identity key used for simple in-memory rate limiting."""

    return request.client.host if request.client else "unknown"


def _validate_governed_threshold(value: float, runtime_settings: Settings) -> float:
    """Enforce threshold within global safety and governance boundaries."""

    threshold = validate_threshold(value)
    if threshold < runtime_settings.min_threshold or threshold > runtime_settings.max_threshold:
        raise HTTPException(
            status_code=422,
            detail=(
                f"threshold must be between governance limits "
                f"{runtime_settings.min_threshold} and {runtime_settings.max_threshold}"
            ),
        )
    return threshold


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attach request id, security headers, and request-level metrics."""

    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = request_id
    started_at = perf_counter()

    response = await call_next(request)

    REQUEST_COUNTER.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
    ).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(perf_counter() - started_at)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return standardized error payloads that include request trace id."""

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail), "request_id": getattr(request.state, "request_id", None)},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service readiness metadata for liveness and startup checks."""

    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        app_version=settings.app_version,
        model_name=settings.model_name,
        default_threshold=settings.default_threshold,
        max_batch_items=settings.max_batch_items,
    )


@app.post("/score", response_model=ScoreResponse)
def score(request: Request, payload: ScoreRequest) -> ScoreResponse:
    """Score one context-response pair and return sentence-level evidence details."""

    if not limiter.allow(_client_key(request), settings.rate_limit_per_minute):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    context = validate_text_argument(
        "context",
        payload.context,
        min_length=5,
        max_length=settings.max_context_chars,
    )
    response_text = validate_text_argument(
        "response",
        payload.response,
        min_length=5,
        max_length=settings.max_response_chars,
    )
    threshold = _validate_governed_threshold(
        settings.default_threshold if payload.threshold is None else payload.threshold,
        settings,
    )

    scorer = get_scorer()
    started_at = perf_counter()
    result = scorer.faithfulness_score(context=context, response=response_text, threshold=threshold)
    duration_ms = round((perf_counter() - started_at) * 1000, 2)

    SCORE_HISTOGRAM.observe(result.score)
    VERDICT_COUNTER.labels(verdict=result.verdict).inc()

    return ScoreResponse(
        score=result.score,
        verdict=result.verdict,
        threshold=result.threshold,
        model_name=scorer.model_name,
        sentence_scores=[
            SentenceScoreResponse(sentence=item.sentence, max_similarity=item.max_similarity)
            for item in result.sentence_scores
        ],
        duration_ms=duration_ms,
        request_id=str(request.state.request_id),
    )


@app.post("/batch", response_model=BatchScoreResponse)
def batch_score(request: Request, payload: BatchScoreRequest) -> BatchScoreResponse:
    """Score multiple context-response pairs in one request with shared model backend."""

    if not limiter.allow(_client_key(request), settings.rate_limit_per_minute):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    if len(payload.items) > settings.max_batch_items:
        raise HTTPException(
            status_code=422,
            detail=f"items must not exceed {settings.max_batch_items} entries",
        )

    threshold = _validate_governed_threshold(
        settings.default_threshold if payload.threshold is None else payload.threshold,
        settings,
    )

    pairs: list[tuple[str, str]] = []
    for item in payload.items:
        context = validate_text_argument(
            "context",
            item.context,
            min_length=5,
            max_length=settings.max_context_chars,
        )
        response_text = validate_text_argument(
            "response",
            item.response,
            min_length=5,
            max_length=settings.max_response_chars,
        )
        pairs.append((context, response_text))

    scorer = get_scorer()
    started_at = perf_counter()
    results = scorer.batch_faithfulness_scores(pairs=pairs, threshold=threshold)
    duration_ms = round((perf_counter() - started_at) * 1000, 2)

    verdict_counts = {"faithful": 0, "hallucinated": 0}
    serialized_results: list[BatchScoreItemResponse] = []
    for index, result in enumerate(results):
        verdict_counts[result.verdict] = verdict_counts.get(result.verdict, 0) + 1
        SCORE_HISTOGRAM.observe(result.score)
        VERDICT_COUNTER.labels(verdict=result.verdict).inc()
        serialized_results.append(
            BatchScoreItemResponse(
                index=index,
                score=result.score,
                verdict=result.verdict,
                sentence_scores=[
                    SentenceScoreResponse(sentence=item.sentence, max_similarity=item.max_similarity)
                    for item in result.sentence_scores
                ],
            )
        )

    average_score = sum(item.score for item in serialized_results) / len(serialized_results)

    return BatchScoreResponse(
        item_count=len(serialized_results),
        average_score=average_score,
        threshold=threshold,
        model_name=scorer.model_name,
        verdict_counts=verdict_counts,
        results=serialized_results,
        duration_ms=duration_ms,
        request_id=str(request.state.request_id),
    )


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics for scoring service observability."""

    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def run() -> None:
    """Run API server for local package script usage."""

    import uvicorn

    uvicorn.run("hallucination_lens.api:app", host="0.0.0.0", port=8003, reload=False)
