"""Evalúa las condiciones de un incidente con un modelo de decisión.

El backend arma el estado del caso y manda las condiciones del protocolo; acá solo se consulta al modelo
y se devuelve la probabilidad de cada una. Qué hacer con esas probabilidades lo decide el backend.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from app.application.ports.decision_model import DecisionModel
from app.domain.errors import AiError
from app.domain.evaluation import Answer, EvaluationResult, Question

# Tope del estado serializado: un caso con muchas evidencias cabe holgado, y más que esto baja el acierto.
MAX_STATE_CHARS = 60_000


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EvaluateConditions:
    def __init__(self, model: DecisionModel, clock: Callable[[], datetime] = _now) -> None:
        self.model = model
        self.clock = clock

    async def execute(self, job_id: str, incident_id: int, state: Any,
                      questions: Sequence[Question]) -> EvaluationResult:
        if len(json.dumps(state, ensure_ascii=False)) > MAX_STATE_CHARS:
            raise AiError("state_too_large")
        reply = await self.model.decide(state=state, questions=questions)
        return EvaluationResult(
            job_id=job_id,
            incident_id=incident_id,
            answers=tuple(Answer(question.code, reply.probabilities[question.code]) for question in questions),
            provider=reply.provider,
            model=reply.model,
            generated_at=self.clock(),
            input_tokens=reply.input_tokens,
            cost=reply.cost,
        )
