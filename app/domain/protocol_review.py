"""Revisión del protocolo de despacho: propuestas de mejora para que las decida una persona.

El analista trabaja fuera de cualquier caso. Mira lo que pasó con el protocolo vigente (los hechos que le
manda el backend, ya contados) y propone cambios. No cambia nada: cada propuesta la lee una persona.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

KINDS = ("adjust_threshold", "change_question", "change_response", "add_condition", "remove_condition", "other")
MAX_PROPOSALS = 12


@dataclass(frozen=True, slots=True)
class Fact:
    """Algo que pasó con el protocolo, ya contado por el backend. `id` es con lo que se lo cita."""

    id: str
    text: str
    event_type: str | None = None
    condition: str | None = None


@dataclass(frozen=True, slots=True)
class NormExcerpt:
    id: str
    citation: str
    text: str


@dataclass(frozen=True, slots=True)
class Proposal:
    kind: str
    event_type: str
    condition: str | None
    proposal: str
    rationale: str
    fact_ids: tuple[str, ...]
    norm_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProtocolReview:
    job_id: str
    proposals: tuple[Proposal, ...]
    discarded: int  # propuestas del modelo que no pasaron las reglas
    provider: str
    model: str
    prompt_version: str | None
    generated_at: datetime


def ground(proposals: Iterable[Proposal], *, fact_ids: set[str], norm_ids: set[str],
           conditions: Mapping[str, set[str]]) -> tuple[tuple[Proposal, ...], int]:
    """Deja solo las propuestas que se pueden defender y cuenta las que se descartan.

    Estas reglas las aplica el código, no el modelo:
    - El tipo de suceso tiene que estar en el protocolo.
    - La condición tiene que existir en ese tipo, salvo en `add_condition`, donde tiene que ser nueva o faltar.
    - Sin propuesta, sin porqué o sin ningún hecho recibido que la respalde, no pasa.
    - Las citas a hechos o a extractos de la norma que no se mandaron se quitan.
    - No se repite la misma propuesta para la misma condición, y hay un tope.
    """
    kept: list[Proposal] = []
    seen: set[tuple[str, str, str | None]] = set()
    discarded = 0
    for proposal in proposals:
        known = conditions.get(proposal.event_type)
        facts = tuple(dict.fromkeys(fact for fact in proposal.fact_ids if fact in fact_ids))
        key = (proposal.kind, proposal.event_type, proposal.condition)
        new_condition = proposal.kind == "add_condition"
        valid = (
            proposal.kind in KINDS
            and known is not None
            and (proposal.condition not in known if new_condition else proposal.condition in known)
            and bool(proposal.proposal.strip())
            and bool(proposal.rationale.strip())
            and bool(facts)
            and key not in seen
            and len(kept) < MAX_PROPOSALS
        )
        if not valid:
            discarded += 1
            continue
        seen.add(key)
        kept.append(Proposal(
            kind=proposal.kind, event_type=proposal.event_type, condition=proposal.condition,
            proposal=proposal.proposal.strip(), rationale=proposal.rationale.strip(), fact_ids=facts,
            norm_ids=tuple(dict.fromkeys(norm for norm in proposal.norm_ids if norm in norm_ids)),
        ))
    return tuple(kept), discarded
