"""Razonador de prueba: implementa `Reasoner` sin modelo ni herramientas.

Concluye solo con lo que el estado ya dice en `respuestasDelTestigo`: un «SI» da `met`, un «NO» da `not_met` y
cualquier otra cosa, `unknown`. Sirve para probar el flujo completo sin clave ni costo: no razona de verdad.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.application.ports.reasoner import ReasoningReply
from app.domain.evaluation import Question
from app.domain.reasoning import MET, NOT_MET, UNKNOWN, Conclusion

PROVIDER = "prueba"
MODEL = "razonador-de-prueba"
ANSWERS_KEY = "respuestasDelTestigo"
SOURCE = "respuesta del testigo"


def _conclusion(code: str, answer: Any) -> Conclusion:
    if answer == "SI":
        return Conclusion(code, MET, "El testigo contestó que sí.", (SOURCE,))
    if answer == "NO":
        return Conclusion(code, NOT_MET, "El testigo contestó que no.", (SOURCE,))
    return Conclusion(code, UNKNOWN, "El razonador de prueba no saca conclusiones propias.")


class PruebaReasoner:
    async def reason(self, *, incident_id: int, state: Any, questions: Sequence[Question]) -> ReasoningReply:
        answers = state.get(ANSWERS_KEY) if isinstance(state, dict) else None
        if not isinstance(answers, dict):
            answers = {}
        return ReasoningReply(
            conclusions=tuple(_conclusion(question.code, answers.get(question.code)) for question in questions),
            tool_calls=(), provider=PROVIDER, model=MODEL,
        )
