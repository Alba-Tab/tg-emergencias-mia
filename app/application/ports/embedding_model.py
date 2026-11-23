"""Capacidad de convertir textos en vectores.

No genera texto ni decide nada: devuelve, por cada texto, una lista de números que representa su significado.
Dos vectores solo se pueden comparar si salieron del mismo modelo: por eso la respuesta siempre dice cuál fue.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class EmbeddingReply:
    vectors: tuple[tuple[float, ...], ...]  # uno por texto, en el mismo orden
    provider: str
    model: str
    input_tokens: int | None = None
    cost: float | None = None


class EmbeddingModel(Protocol):
    async def embed(self, *, texts: Sequence[str]) -> EmbeddingReply: ...
