"""Contrato HTTP de `POST /v1/embeddings`."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.domain.embedding import EmbeddingResult

EMBEDDING_SCHEMA_VERSION = "text-embedding.v1"

# Un fragmento de norma o el resumen de un caso caben holgados; más que esto conviene partirlo antes.
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobId: UUID
    texts: list[Text] = Field(min_length=1, max_length=64)


def embedding_response(result: EmbeddingResult) -> dict[str, Any]:
    return {
        "jobId": result.job_id,
        "schemaVersion": EMBEDDING_SCHEMA_VERSION,
        "dimensions": result.dimensions,
        "vectors": [list(vector) for vector in result.vectors],
        "provenance": {
            "provider": result.provider,
            "model": result.model,
            "generatedAt": result.generated_at.isoformat(),
            "method": "embedding_model",
        },
        "usage": {"inputTokens": result.input_tokens, "cost": result.cost},
    }
