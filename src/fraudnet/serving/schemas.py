"""Typed request and response models for the scoring API.

Validation happens here, before anything touches the feature store: ids are bounded integers, amounts are
positive and finite, a payment goes either to a merchant or to a payee (never both), and transfers must use
the p2p channel. A request that fails these checks gets a 422 and leaves the store untouched.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelName = Literal["rules", "gbdt", "graphsage"]
ChannelName = Literal["pos", "ecom", "p2p"]
CHANNEL_CODES: dict[str, int] = {"pos": 0, "ecom": 1, "p2p": 2}
MAX_ID = 1_000_000_000
MAX_BATCH = 500


class TransactionIn(BaseModel):
    """A payment to score. Omit ``timestamp`` to use the store clock (useful for demos)."""

    model_config = ConfigDict(extra="forbid")

    txn_id: str | None = Field(None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    timestamp: datetime | None = None
    account_id: int = Field(ge=0, le=MAX_ID)
    device_id: int = Field(ge=0, le=MAX_ID)
    merchant_id: int | None = Field(None, ge=0, le=MAX_ID)
    payee_account_id: int | None = Field(None, ge=0, le=MAX_ID)
    amount: float = Field(gt=0, le=1_000_000, allow_inf_nan=False)
    channel: ChannelName

    @model_validator(mode="after")
    def _counterparty(self) -> Self:
        if (self.merchant_id is None) == (self.payee_account_id is None):
            raise ValueError("set exactly one of merchant_id and payee_account_id")
        if self.payee_account_id is not None:
            if self.channel != "p2p":
                raise ValueError("transfers to a payee must use channel p2p")
            if self.payee_account_id == self.account_id:
                raise ValueError("an account cannot pay itself")
        elif self.channel == "p2p":
            raise ValueError("merchant payments must use channel pos or ecom")
        return self


class ScoreRequest(TransactionIn):
    """A transaction plus scoring options."""

    model: ModelName | None = None
    explain: bool = True
    include_features: bool = False


class ReasonOut(BaseModel):
    """One reason code."""

    code: str
    description: str
    detail: str | None = None
    contribution: float


class ScoreResponse(BaseModel):
    """Score, decision and reasons for one transaction."""

    txn_id: str
    timestamp: str
    model: ModelName
    score: float
    decision: Literal["approve", "review", "block"]
    thresholds: dict[str, float]
    reasons: list[ReasonOut]
    alert_id: int | None
    latency_ms: float
    features: dict[str, float] | None = None
    request_id: str = ""


class BatchRequest(BaseModel):
    """Up to 500 transactions, scored in order."""

    model_config = ConfigDict(extra="forbid")

    transactions: list[ScoreRequest] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    """Results in request order."""

    results: list[ScoreResponse]
    count: int
    request_id: str = ""


class AlertOut(BaseModel):
    """An alert in the analyst queue."""

    alert_id: int
    txn_id: str
    timestamp: str
    account: int
    device: int
    merchant: int
    dst: int
    amount: float
    score: float
    model: str
    decision: str
    reasons: list[ReasonOut]
    status: Literal["open", "fraud", "legit"]
    true_fraud: int | None = Field(
        None, description="planted ground truth, only for replayed simulator transactions"
    )


class AlertList(BaseModel):
    """A page of alerts, highest score first."""

    alerts: list[AlertOut]
    count: int
    stats: dict[str, int]


class ResolveRequest(BaseModel):
    """An analyst's verdict."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["fraud", "legit"]


class LabelRequest(BaseModel):
    """A chargeback or fraud report for a recently scored transaction."""

    model_config = ConfigDict(extra="forbid")

    txn_id: str = Field(min_length=1, max_length=64)


class ReplayRequest(BaseModel):
    """Stream the next held-out simulator transactions through the scoring path."""

    model_config = ConfigDict(extra="forbid")

    n: int = Field(500, ge=1, le=5000)
    model: ModelName | None = None


class HealthResponse(BaseModel):
    """Liveness and what is loaded."""

    status: Literal["ok"]
    version: str
    models: list[str]
    default_model: str
    clock: str | None
    uptime_s: float


class ModelsResponse(BaseModel):
    """Loaded models, thresholds and provenance."""

    models: list[str]
    default: str
    thresholds: dict[str, dict[str, float]]
    trained_at: str
    fit_seconds: dict[str, float]
    build_day: int
    features: list[str]
