"""Contrato HTTP de `POST /v1/evaluations`."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.evaluation import EvaluationResult, Question

EVALUATION_SCHEMA_VERSION = "condition-evaluation.v1"


class QuestionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,59}$")
    statement: str = Field(min_length=3, max_length=500)

    def to_domain(self) -> Question:
        return Question(self.code, self.statement.strip())


class EvaluationRequest(BaseModel):
    """`state` es el estado del caso tal como lo arma el backend; acá no se interpreta, solo se reenvía."""

    model_config = ConfigDict(extra="forbid")

    jobId: UUID
    incidentId: int = Field(gt=0)
    state: dict[str, Any]
    questions: list[QuestionIn] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def _unique_codes(self) -> "EvaluationRequest":
        codes = [question.code for question in self.questions]
        if len(set(codes)) != len(codes):
            raise ValueError("questions no puede repetir códigos")
        return self

    def to_domain(self) -> list[Question]:
        return [question.to_domain() for question in self.questions]


def evaluation_response(result: EvaluationResult) -> dict[str, Any]:
    return {
        "jobId": result.job_id,
        "incidentId": result.incident_id,
        "schemaVersion": EVALUATION_SCHEMA_VERSION,
        "answers": [{"code": answer.code, "probability": answer.probability} for answer in result.answers],
        "provenance": {
            "provider": result.provider,
            "model": result.model,
            "generatedAt": result.generated_at.isoformat(),
            "method": "decision_model",
        },
        "usage": {"inputTokens": result.input_tokens, "cost": result.cost},
    }
