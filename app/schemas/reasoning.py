"""Contrato HTTP de `POST /v1/reasonings`."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.evaluation import Question
from app.domain.reasoning import ReasoningResult
from app.schemas.evaluation import QuestionIn

REASONING_SCHEMA_VERSION = "condition-reasoning.v1"


class ReasoningRequest(BaseModel):
    """`state` es el estado del caso tal como lo arma el backend; el razonador lo recibe entero."""

    model_config = ConfigDict(extra="forbid")

    jobId: UUID
    incidentId: int = Field(gt=0)
    state: dict[str, Any]
    questions: list[QuestionIn] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def _unique_codes(self) -> "ReasoningRequest":
        codes = [question.code for question in self.questions]
        if len(set(codes)) != len(codes):
            raise ValueError("questions no puede repetir códigos")
        return self

    def to_domain(self) -> list[Question]:
        return [question.to_domain() for question in self.questions]


def reasoning_response(result: ReasoningResult) -> dict[str, Any]:
    return {
        "jobId": result.job_id,
        "incidentId": result.incident_id,
        "schemaVersion": REASONING_SCHEMA_VERSION,
        "conclusions": [
            {"code": conclusion.code, "verdict": conclusion.verdict, "rationale": conclusion.rationale,
             "sources": list(conclusion.sources)}
            for conclusion in result.conclusions
        ],
        "toolCalls": list(result.tool_calls),
        "provenance": {
            "provider": result.provider,
            "model": result.model,
            "promptVersion": result.prompt_version,
            "generatedAt": result.generated_at.isoformat(),
            "method": "agent",
        },
        "usage": {"inputTokens": result.input_tokens, "outputTokens": result.output_tokens,
                  "requests": result.requests},
    }
