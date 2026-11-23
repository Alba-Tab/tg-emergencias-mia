"""Contrato HTTP de `POST /v1/protocol-reviews`."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.protocol_review import Fact, NormExcerpt, ProtocolReview

PROTOCOL_REVIEW_SCHEMA_VERSION = "protocol-review.v1"
CODE = r"^[A-Z][A-Z0-9_]{1,59}$"
REFERENCE = r"^[A-Z][A-Z0-9]{0,11}$"


class FactIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=REFERENCE)
    text: str = Field(min_length=3, max_length=1000)
    eventType: str | None = Field(default=None, pattern=CODE)
    condition: str | None = Field(default=None, pattern=CODE)

    def to_domain(self) -> Fact:
        return Fact(self.id, self.text.strip(), self.eventType, self.condition)


class NormExcerptIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=REFERENCE)
    citation: str = Field(min_length=3, max_length=600)
    text: str = Field(min_length=3, max_length=8000)

    def to_domain(self) -> NormExcerpt:
        return NormExcerpt(self.id, self.citation.strip(), self.text.strip())


class ProtocolReviewRequest(BaseModel):
    """`protocol` es el protocolo vigente tal como lo arma el backend; acá no se interpreta más que sus códigos."""

    model_config = ConfigDict(extra="forbid")

    jobId: UUID
    protocol: dict[str, Any]
    facts: list[FactIn] = Field(min_length=1, max_length=300)
    normExcerpts: list[NormExcerptIn] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def _unique_ids(self) -> "ProtocolReviewRequest":
        for items, name in ((self.facts, "facts"), (self.normExcerpts, "normExcerpts")):
            ids = [item.id for item in items]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name} no puede repetir ids")
        return self


def protocol_review_response(review: ProtocolReview) -> dict[str, Any]:
    return {
        "jobId": review.job_id,
        "schemaVersion": PROTOCOL_REVIEW_SCHEMA_VERSION,
        "proposals": [
            {"kind": proposal.kind, "eventType": proposal.event_type, "condition": proposal.condition,
             "proposal": proposal.proposal, "rationale": proposal.rationale, "factIds": list(proposal.fact_ids),
             "normIds": list(proposal.norm_ids)}
            for proposal in review.proposals
        ],
        "discarded": review.discarded,
        "provenance": {
            "provider": review.provider,
            "model": review.model,
            "promptVersion": review.prompt_version,
            "generatedAt": review.generated_at.isoformat(),
            "method": "model",
        },
    }
