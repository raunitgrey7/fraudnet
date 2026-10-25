"""FastAPI application: scoring, alert queue, labels, simulator replay, ring and network views.

``create_app`` accepts an already-built ``ScoringService``, which is how the tests and the latency benchmark
run the real application in-process; without one, it loads the trained artifacts at startup.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from fraudnet import __version__
from fraudnet.config import Settings
from fraudnet.features.schema import FEATURE_NAMES
from fraudnet.logging_utils import configure_logging
from fraudnet.serving.loader import load_service
from fraudnet.serving.metrics import Metrics, ServiceCollector
from fraudnet.serving.middleware import ApiKeyMiddleware, RequestContextMiddleware
from fraudnet.serving.schemas import (
    CHANNEL_CODES,
    AlertList,
    AlertOut,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    LabelRequest,
    ModelsResponse,
    ReplayRequest,
    ResolveRequest,
    ScoreRequest,
    ScoreResponse,
)
from fraudnet.serving.service import (
    OutOfOrderError,
    ScoreResult,
    ScoringService,
    Transaction,
    from_datetime,
    to_iso,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


def _service(request: Request) -> ScoringService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="model bundle not loaded")
    return service


def _transaction(service: ScoringService, body: ScoreRequest) -> Transaction:
    ts = from_datetime(body.timestamp) if body.timestamp is not None else max(service.store.clock, 0)
    return Transaction(
        ts=ts,
        account=body.account_id,
        device=body.device_id,
        merchant=body.merchant_id if body.merchant_id is not None else -1,
        dst=body.payee_account_id if body.payee_account_id is not None else -1,
        amount=float(body.amount),
        channel=CHANNEL_CODES[body.channel],
        txn_id=body.txn_id or service.new_txn_id(),
    )


def _response(
    service: ScoringService, result: ScoreResult, include_features: bool, request_id: str
) -> ScoreResponse:
    features = None
    if include_features:
        features = {name: round(float(v), 6) for name, v in zip(FEATURE_NAMES, result.features, strict=True)}
    return ScoreResponse(
        txn_id=result.txn.txn_id,
        timestamp=to_iso(result.txn.ts),
        model=result.model,  # type: ignore[arg-type]
        score=round(result.score, 6),
        decision=result.decision,  # type: ignore[arg-type]
        thresholds=service.bundle.thresholds[result.model],
        reasons=result.reasons,  # type: ignore[arg-type]
        alert_id=result.alert_id,
        latency_ms=round(result.latency_ms, 3),
        features=features,
        request_id=request_id,
    )


def _score(request: Request, body: ScoreRequest) -> ScoreResponse:
    service = _service(request)
    metrics: Metrics = request.app.state.metrics
    model = body.model or service.default_model
    if model not in service.bundle.models:
        raise HTTPException(status_code=404, detail=f"model {model} is not loaded")
    try:
        result = service.score(_transaction(service, body), model, body.explain)
    except OutOfOrderError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    metrics.score_latency.labels(model).observe(result.latency_ms / 1000.0)
    return _response(service, result, body.include_features, _request_id(request))


def create_app(settings: Settings | None = None, service: ScoringService | None = None) -> FastAPI:
    """Build the application; artifacts are loaded at startup unless ``service`` is given."""
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    metrics = Metrics.create()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            app.state.service = service
        else:
            app.state.service = load_service(settings)
        metrics.registry.register(ServiceCollector(app.state.service.stats))
        logger.info("service ready", extra={"models": app.state.service.bundle.models})
        yield

    app = FastAPI(title="fraudnet", version=__version__, lifespan=lifespan)
    app.state.metrics = metrics
    app.state.settings = settings
    _ops_routes(app)
    _scoring_routes(app)
    _queue_routes(app)
    _view_routes(app)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.add_middleware(ApiKeyMiddleware, api_key=settings.api_key)
    app.add_middleware(RequestContextMiddleware, metrics=metrics)
    return app


def _ops_routes(app: FastAPI) -> None:
    """Error handlers, health, metrics, model info and the dashboard page."""

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        return JSONResponse(status_code=422, content={"detail": detail, "request_id": _request_id(request)})

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.detail, "request_id": _request_id(request)}
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(request: Request) -> HealthResponse:
        service = _service(request)
        clock = to_iso(service.store.clock) if service.store.clock > 0 else None
        uptime = round(time.time() - service.started, 1)
        return HealthResponse(
            status="ok",
            version=__version__,
            models=service.bundle.models,
            default_model=service.default_model,
            clock=clock,
            uptime_s=uptime,
        )

    @app.get("/metrics", include_in_schema=False)
    async def prometheus(request: Request) -> Response:
        return Response(generate_latest(request.app.state.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models", response_model=ModelsResponse, tags=["ops"])
    async def models(request: Request) -> ModelsResponse:
        bundle = _service(request).bundle
        return ModelsResponse(
            models=bundle.models, default=_service(request).default_model, thresholds=bundle.thresholds,
            trained_at=bundle.trained_at, fit_seconds=bundle.fit_seconds, build_day=bundle.split.build_day,
            features=list(FEATURE_NAMES),
        )  # fmt: skip

    @app.get("/v1/stats", tags=["ops"])
    def stats(request: Request) -> dict[str, Any]:
        return _service(request).stats()

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        page = STATIC_DIR / "index.html"
        if page.exists():
            return FileResponse(page)
        return JSONResponse({"service": "fraudnet", "docs": "/docs"})


def _scoring_routes(app: FastAPI) -> None:
    """Single and batch scoring, fraud reports and the simulator replay."""

    @app.post("/v1/score", response_model=ScoreResponse, tags=["scoring"])
    def score(request: Request, body: ScoreRequest) -> ScoreResponse:
        return _score(request, body)

    @app.post("/v1/score/batch", response_model=BatchResponse, tags=["scoring"])
    def score_batch(request: Request, body: BatchRequest) -> BatchResponse:
        results = [_score(request, item) for item in body.transactions]
        return BatchResponse(results=results, count=len(results), request_id=_request_id(request))

    @app.post("/v1/labels", tags=["scoring"])
    def report_label(request: Request, body: LabelRequest) -> dict[str, Any]:
        if not _service(request).report_fraud(body.txn_id):
            raise HTTPException(status_code=404, detail=f"unknown or expired transaction {body.txn_id}")
        return {"txn_id": body.txn_id, "status": "scheduled", "request_id": _request_id(request)}

    @app.post("/v1/replay", tags=["scoring"])
    def replay_feed(request: Request, body: ReplayRequest) -> dict[str, Any]:
        model = body.model or _service(request).default_model
        if model not in _service(request).bundle.models:
            raise HTTPException(status_code=404, detail=f"model {model} is not loaded")
        return {**_service(request).replay_feed(body.n, model), "request_id": _request_id(request)}


def _alert_out(item: dict[str, Any]) -> AlertOut:
    return AlertOut(
        **{k: v for k, v in item.items() if k in AlertOut.model_fields}, timestamp=to_iso(item["ts"])
    )


def _queue_routes(app: FastAPI) -> None:
    """The analyst queue."""

    @app.get("/v1/alerts", response_model=AlertList, tags=["alerts"])
    def alerts(
        request: Request,
        status: Annotated[str, Query(pattern="^(open|fraud|legit|all)$")] = "open",
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> AlertList:
        queue = _service(request).alerts
        if queue is None:
            raise HTTPException(status_code=503, detail="alert queue disabled")
        items = queue.list(None if status == "all" else status, limit)  # type: ignore[arg-type]
        return AlertList(alerts=[_alert_out(i) for i in items], count=len(items), stats=queue.stats())

    @app.post("/v1/alerts/{alert_id}/resolve", response_model=AlertOut, tags=["alerts"])
    def resolve(request: Request, alert_id: int, body: ResolveRequest) -> AlertOut:
        resolved = _service(request).resolve_alert(alert_id, body.label)
        if resolved is None:
            raise HTTPException(status_code=404, detail=f"no open alert {alert_id}")
        return _alert_out(resolved)


def _view_routes(app: FastAPI) -> None:
    """Graph views for the dashboard."""

    @app.get("/v1/rings", tags=["graph"])
    def rings(request: Request, limit: Annotated[int, Query(ge=1, le=50)] = 8) -> dict[str, Any]:
        return {"rings": _service(request).rings(limit), "request_id": _request_id(request)}

    @app.get("/v1/accounts/{account_id}/network", tags=["graph"])
    def network(request: Request, account_id: int) -> dict[str, Any]:
        view = _service(request).network(account_id)
        if view is None:
            raise HTTPException(status_code=404, detail=f"unknown account {account_id}")
        return view
