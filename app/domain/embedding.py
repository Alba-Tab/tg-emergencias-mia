"""Textos convertidos en vectores que representan su significado, para buscar por parecido."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """`vectors` va en el mismo orden que los textos recibidos, y todos tienen `dimensions` números."""

    job_id: str
    vectors: tuple[tuple[float, ...], ...]
    dimensions: int
    provider: str
    model: str
    generated_at: datetime
    input_tokens: int | None = None
    cost: float | None = None
