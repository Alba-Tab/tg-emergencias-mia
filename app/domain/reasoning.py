"""Razonamiento sobre condiciones que quedaron en duda: una conclusión por condición, siempre con su porqué.

La regla que importa está acá y no en el modelo: una conclusión de «se cumple» o «no se cumple» vale solo si
trae una justificación y al menos una fuente. Si no, pasa a «no se puede saber» y la decide una persona.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from app.domain.evaluation import Question

MET = "met"
NOT_MET = "not_met"
UNKNOWN = "unknown"
VERDICTS = (MET, NOT_MET, UNKNOWN)

MAX_RATIONALE_CHARS = 600
MAX_SOURCES = 6
MAX_SOURCE_CHARS = 160

NO_CONCLUSION = "El agente no llegó a una conclusión sobre esta condición."
UNGROUNDED = "El agente dio una conclusión sin justificarla o sin citar una fuente: no se toma."


@dataclass(frozen=True, slots=True)
class Conclusion:
    """`verdict` es `met`, `not_met` o `unknown`; `sources` dice en qué se apoya."""

    code: str
    verdict: str
    rationale: str
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    job_id: str
    incident_id: int
    conclusions: tuple[Conclusion, ...]
    tool_calls: tuple[str, ...]
    provider: str
    model: str
    prompt_version: str | None
    generated_at: datetime
    input_tokens: int | None = None
    output_tokens: int | None = None
    requests: int | None = None


def _clean(text: object, limit: int) -> str:
    return " ".join(str(text).split())[:limit] if isinstance(text, str) else ""


def ground(raw: Iterable[Conclusion], questions: Sequence[Question]) -> tuple[Conclusion, ...]:
    """Una conclusión por condición pedida, en el mismo orden.

    - Las conclusiones sobre condiciones que nadie pidió se descartan.
    - Si el agente repite una condición, vale la primera.
    - Una condición sin conclusión, o con un veredicto que no existe, queda en `unknown`.
    - `met` y `not_met` sin justificación o sin ninguna fuente pasan a `unknown`: sin porqué no se actúa.
    """
    by_code: dict[str, Conclusion] = {}
    for conclusion in raw:
        by_code.setdefault(conclusion.code, conclusion)
    grounded: list[Conclusion] = []
    for question in questions:
        conclusion = by_code.get(question.code)
        if conclusion is None or conclusion.verdict not in VERDICTS:
            grounded.append(Conclusion(question.code, UNKNOWN, NO_CONCLUSION))
            continue
        rationale = _clean(conclusion.rationale, MAX_RATIONALE_CHARS)
        sources = tuple(source for source in (_clean(item, MAX_SOURCE_CHARS) for item in conclusion.sources)
                        if source)[:MAX_SOURCES]
        if conclusion.verdict != UNKNOWN and not (rationale and sources):
            note = f"{rationale} {UNGROUNDED}".strip()
            grounded.append(Conclusion(question.code, UNKNOWN, note[:MAX_RATIONALE_CHARS], sources))
            continue
        grounded.append(Conclusion(question.code, conclusion.verdict, rationale or NO_CONCLUSION, sources))
    return tuple(grounded)
