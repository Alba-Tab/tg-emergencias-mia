"""Convierte textos en vectores con un modelo de embeddings.

El backend decide qué textos guardar y cómo compararlos; acá solo se consulta al modelo y se devuelve un
vector por texto, en el mismo orden, con el modelo que los produjo.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from app.application.ports.embedding_model import EmbeddingModel
from app.domain.embedding import EmbeddingResult


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EmbedTexts:
    def __init__(self, model: EmbeddingModel, clock: Callable[[], datetime] = _now) -> None:
        self.model = model
        self.clock = clock

    async def execute(self, job_id: str, texts: Sequence[str]) -> EmbeddingResult:
        reply = await self.model.embed(texts=texts)
        return EmbeddingResult(
            job_id=job_id,
            vectors=reply.vectors,
            dimensions=len(reply.vectors[0]),
            provider=reply.provider,
            model=reply.model,
            generated_at=self.clock(),
            input_tokens=reply.input_tokens,
            cost=reply.cost,
        )
