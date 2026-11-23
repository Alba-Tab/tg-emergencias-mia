"""Modelo de decisión de prueba: implementa `DecisionModel` sin llamar a ningún proveedor.

La probabilidad de cada condición sale solo de lo que el estado ya dice de ella en `respuestasDelTestigo`:
un «SI» da 0,95, un «NO» da 0,05 y cualquier otra cosa, o nada, da 0,5. No deduce nada de las fotos ni
de los audios: sirve para probar el flujo completo sin clave ni costo, no para decidir de verdad.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.application.ports.decision_model import DecisionReply
from app.domain.evaluation import Question

PROVIDER = "prueba"
MODEL = "decisor-de-prueba"
ANSWERS_KEY = "respuestasDelTestigo"
YES, NO, UNKNOWN = 0.95, 0.05, 0.5


def _probability(answer: Any) -> float:
    if answer == "SI":
        return YES
    if answer == "NO":
        return NO
    return UNKNOWN


class PruebaDecisionModel:
    async def decide(self, *, state: Any, questions: Sequence[Question]) -> DecisionReply:
        answers = state.get(ANSWERS_KEY) if isinstance(state, dict) else None
        if not isinstance(answers, dict):
            answers = {}
        return DecisionReply(
            probabilities={question.code: _probability(answers.get(question.code)) for question in questions},
            provider=PROVIDER, model=MODEL,
        )
