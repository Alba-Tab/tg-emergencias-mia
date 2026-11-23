"""Capacidad de razonar sobre condiciones en duda, consultando herramientas si las tiene.

A diferencia del modelo de decisión, que responde en una sola llamada con probabilidades, el razonador puede
dar varios pasos: mirar el caso, buscar casos parecidos, leer la norma. Devuelve conclusiones con su porqué.
Cambiar de modelo o de biblioteca de agentes solo requiere otro adaptador.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.domain.evaluation import Question
from app.domain.reasoning import Conclusion


@dataclass(frozen=True, slots=True)
class ReasoningReply:
    """Lo que devolvió el razonador, todavía sin pasar por las reglas del dominio."""

    conclusions: tuple[Conclusion, ...]
    tool_calls: tuple[str, ...]  # los nombres de las herramientas que usó, en orden
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    requests: int | None = None


class Reasoner(Protocol):
    async def reason(self, *, incident_id: int, state: Any, questions: Sequence[Question]) -> ReasoningReply: ...
