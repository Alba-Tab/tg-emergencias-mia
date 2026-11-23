"""Evaluación de condiciones de un incidente: cada condición es una afirmación de sí o no."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Question:
    """Una condición del protocolo: `code` la identifica y `statement` es la afirmación que se evalúa."""

    code: str
    statement: str


@dataclass(frozen=True, slots=True)
class Answer:
    """`probability` es la probabilidad, de 0 a 1, de que la afirmación sea cierta según el estado recibido."""

    code: str
    probability: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    job_id: str
    incident_id: int
    answers: tuple[Answer, ...]
    provider: str
    model: str
    generated_at: datetime
    input_tokens: int | None = None
    cost: float | None = None
