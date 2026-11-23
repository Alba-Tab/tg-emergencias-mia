"""Capacidad de responder preguntas cerradas sobre un estado, con la probabilidad de cada respuesta.

No genera texto: recibe un estado y afirmaciones de sí o no, y devuelve qué tan probable es cada una.
Cambiar de proveedor (Jev por OpenRouter, la API de TypeSafe o un modelo abierto) solo requiere otro adaptador.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.domain.evaluation import Question


@dataclass(frozen=True, slots=True)
class DecisionReply:
    probabilities: dict[str, float]  # por código de pregunta, de 0 a 1
    provider: str
    model: str
    input_tokens: int | None = None
    cost: float | None = None


class DecisionModel(Protocol):
    async def decide(self, *, state: Any, questions: Sequence[Question]) -> DecisionReply: ...
