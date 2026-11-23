"""Adaptador de embeddings a través de OpenRouter (`POST /api/v1/embeddings`).

Manda todos los textos en un solo pedido y devuelve un vector por texto, en el mismo orden. La respuesta se
revisa entera: si falta un vector, viene vacío o no todos miden lo mismo, no sirve para comparar y se descarta.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from typing import Any

import httpx

from app.application.ports.embedding_model import EmbeddingReply
from app.domain.errors import AiError

logger = logging.getLogger(__name__)

ENDPOINT = "https://openrouter.ai/api/v1/embeddings"


def vectors(body: Any, expected: int) -> tuple[tuple[float, ...], ...]:
    """Los vectores ordenados por `index`. Tiene que haber uno por texto y todos del mismo tamaño."""
    items = sorted(body["data"], key=lambda item: item["index"])
    if [item["index"] for item in items] != list(range(expected)):
        raise ValueError("faltan vectores")
    result = []
    for item in items:
        vector = item["embedding"]
        if not isinstance(vector, list) or not vector:
            raise ValueError("vector vacío")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in vector):
            raise ValueError("vector con valores que no son números")
        result.append(tuple(float(value) for value in vector))
    if len({len(vector) for vector in result}) != 1:
        raise ValueError("vectores de distinto tamaño")
    return tuple(result)


class OpenRouterEmbeddingModel:
    def __init__(
        self,
        api_key: str,
        model: str,
        client: httpx.AsyncClient,
        limiter: asyncio.Semaphore | None = None,
        timeout: float = 30.0,
        attempts: int = 2,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client
        self.limiter = limiter or asyncio.Semaphore(4)
        self.timeout = timeout
        self.attempts = attempts

    async def embed(self, *, texts: Sequence[str]) -> EmbeddingReply:
        payload = {"model": self.model, "input": list(texts), "encoding_format": "float"}
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
                return self._reply(response, len(texts))
        raise AiError("provider_unavailable", retryable=True)

    def _reply(self, response: httpx.Response, expected: int) -> EmbeddingReply:
        try:
            body = response.json()
        except ValueError as exc:
            raise AiError("invalid_model_output", retryable=True) from exc
        if isinstance(body, dict) and body.get("error"):
            raise AiError("provider_unavailable", retryable=True)
        try:
            result = vectors(body, expected)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise AiError("invalid_model_output", retryable=True) from exc
        usage = body.get("usage") or {}
        # Solo cantidades: los textos llevan lo que se sabe de los casos y no van a los registros.
        logger.info("Modelo de embeddings %s: textos=%d dimensiones=%d tokens=%s costo=%s",
                    body.get("model", self.model), expected, len(result[0]), usage.get("prompt_tokens"),
                    usage.get("cost"))
        return EmbeddingReply(
            vectors=result, provider="openrouter", model=body.get("model") or self.model,
            input_tokens=usage.get("prompt_tokens"), cost=usage.get("cost"),
        )
