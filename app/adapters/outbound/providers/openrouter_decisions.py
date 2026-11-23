"""Adaptador del modelo de decisión Jev a través del endpoint de decisiones de OpenRouter.

Cada pregunta viaja como una afirmación de tipo `noul`; la respuesta trae, por pregunta, la probabilidad
de que sea cierta. No se le pide texto ni explicación: quien decide qué hacer con cada probabilidad es
el backend, con los umbrales del protocolo.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import httpx

from app.application.ports.decision_model import DecisionReply
from app.domain.errors import AiError
from app.domain.evaluation import Question

logger = logging.getLogger(__name__)

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"


def decisions_payload(model: str, state: Any, questions: Sequence[Question]) -> dict[str, Any]:
    return {
        "model": model,
        "state": state,
        "questions": {question.code: {"type": "noul", "instructions": question.statement} for question in questions},
    }


def probabilities(body: Any, questions: Sequence[Question]) -> dict[str, float]:
    """Una probabilidad entre 0 y 1 por cada pregunta enviada; si falta alguna, la respuesta no sirve."""
    answers = body["answers"]
    result: dict[str, float] = {}
    for question in questions:
        value = answers[question.code]["noul"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError("probabilidad fuera de rango")
        result[question.code] = float(value)
    return result


class OpenRouterDecisionModel:
    def __init__(
        self,
        api_key: str,
        model: str,
        client: httpx.AsyncClient,
        limiter: asyncio.Semaphore | None = None,
        timeout: float = 15.0,
        attempts: int = 2,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client
        self.limiter = limiter or asyncio.Semaphore(4)
        self.timeout = timeout
        self.attempts = attempts

    async def decide(self, *, state: Any, questions: Sequence[Question]) -> DecisionReply:
        payload = decisions_payload(self.model, state, questions)
        async with self.limiter:
            for attempt in range(self.attempts):
                last = attempt == self.attempts - 1
                try:
                    response = await asyncio.wait_for(self.client.post(
                        ENDPOINT, headers={"Authorization": f"Bearer {self.api_key}"}, json=payload, timeout=self.timeout,
                    ), self.timeout)
                except (httpx.TimeoutException, httpx.TransportError, asyncio.TimeoutError) as exc:
                    if last:
                        raise AiError("provider_unavailable", retryable=True) from exc
                    await asyncio.sleep(0.25 * 2**attempt)
                    continue
                if response.status_code in {408, 429} or response.status_code >= 500:
                    if last:
                        raise AiError("provider_unavailable", retryable=True)
                    await asyncio.sleep(0.25 * 2**attempt)
                    continue
                if response.status_code in {401, 402, 403}:
                    raise AiError("provider_misconfigured")
                if response.status_code >= 400:
                    raise AiError("provider_rejected")
                return self._reply(response, questions)
        raise AiError("provider_unavailable", retryable=True)

    def _reply(self, response: httpx.Response, questions: Sequence[Question]) -> DecisionReply:
        try:
            body = response.json()
        except ValueError as exc:
            raise AiError("invalid_model_output", retryable=True) from exc
        if isinstance(body, dict) and body.get("error"):
            raise AiError("provider_unavailable", retryable=True)
        try:
            answers = probabilities(body, questions)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise AiError("invalid_model_output", retryable=True) from exc
        usage = body.get("usage") or {}
        logger.info("Modelo de decisión %s: preguntas=%d tokens entrada=%s costo=%s", body.get("model", self.model),
                    len(questions), usage.get("input_tokens"), usage.get("cost"))
        return DecisionReply(
            probabilities=answers, provider="openrouter", model=body.get("model") or self.model,
            input_tokens=usage.get("input_tokens"), cost=usage.get("cost"),
        )
