"""Razona sobre las condiciones que quedaron en duda después de las reglas y del modelo de decisión.

El backend manda el estado del caso y las condiciones abiertas. El razonador mira lo que haga falta y propone una
conclusión por condición; acá se le aplican las reglas del dominio antes de devolverla. Qué se hace con cada
conclusión lo decide el backend: lo que quede en `unknown` va a una persona.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from app.application.ports.reasoner import Reasoner
from app.application.use_cases.evaluate_conditions import MAX_STATE_CHARS
from app.domain.errors import AiError
from app.domain.evaluation import Question
from app.domain.reasoning import ReasoningResult, ground


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ReasonAboutConditions:
    def __init__(self, reasoner: Reasoner, prompt_version: str | None = None,
                 clock: Callable[[], datetime] = _now) -> None:
        self.reasoner = reasoner
        self.prompt_version = prompt_version
        self.clock = clock

    async def execute(self, job_id: str, incident_id: int, state: Any,
                      questions: Sequence[Question]) -> ReasoningResult:
        if len(json.dumps(state, ensure_ascii=False)) > MAX_STATE_CHARS:
            raise AiError("state_too_large")
        reply = await self.reasoner.reason(incident_id=incident_id, state=state, questions=questions)
        return ReasoningResult(
            job_id=job_id,
            incident_id=incident_id,
            conclusions=ground(reply.conclusions, questions),
            tool_calls=reply.tool_calls,
            provider=reply.provider,
            model=reply.model,
            prompt_version=self.prompt_version,
            generated_at=self.clock(),
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            requests=reply.requests,
        )
