"""Razonador hecho con PydanticAI: un agente con salida estructurada y, si se configuran, herramientas MCP.

El agente no guarda nada entre pedidos: cada razonamiento empieza de cero con el estado que manda el backend. Las
herramientas las sirve el backend por MCP (ver el caso, buscar casos parecidos, leer la norma, preguntarle al
testigo, proponerle algo al operador). Si no hay servidor MCP configurado, razona solo con el estado recibido.

Los límites van por código, no por prompt: una cantidad máxima de pedidos al modelo y un tiempo total.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any, Literal

import httpx2
from fastmcp.exceptions import FastMCPError
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, Field
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import (
    ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded,
)
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.toolsets import AbstractToolset

from app.application.ports.reasoner import ReasoningReply
from app.domain.errors import AiError
from app.domain.evaluation import Question
from app.domain.reasoning import Conclusion

logger = logging.getLogger(__name__)

PROVIDER = "openrouter"


class ConclusionOut(BaseModel):
    """Lo que el modelo tiene que devolver por cada condición."""

    code: str = Field(description="El código de la condición, tal como se recibió.")
    verdict: Literal["met", "not_met", "unknown"] = Field(
        description="met: se cumple. not_met: no se cumple. unknown: con lo que hay no se puede saber.")
    rationale: str = Field(description="Una o dos frases con el porqué.")
    sources: list[str] = Field(default_factory=list, description="En qué se apoya la conclusión.")


class ReasoningOutput(BaseModel):
    conclusions: list[ConclusionOut]


def user_prompt(incident_id: int, state: Any, questions: Sequence[Question]) -> str:
    """El pedido, como JSON: así el estado no se mezcla con las instrucciones."""
    return json.dumps({
        "incidenteId": incident_id,
        "condiciones": [{"codigo": question.code, "enunciado": question.statement} for question in questions],
        "estado": state,
    }, ensure_ascii=False)


def tool_calls(messages: Sequence[Any], output_tools: Sequence[str] = ("final_result",)) -> tuple[str, ...]:
    """Los nombres de las herramientas que llamó el modelo, en orden, sin la de la salida final."""
    names: list[str] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            names.extend(part.tool_name for part in message.parts
                         if isinstance(part, ToolCallPart) and not part.tool_name.startswith(tuple(output_tools)))
    return tuple(names)


class PydanticAiReasoner:
    def __init__(
        self,
        model: Model,
        instructions: str,
        *,
        model_name: str,
        toolsets: Sequence[AbstractToolset[Any]] = (),
        provider: str = PROVIDER,
        max_requests: int = 8,
        timeout: float = 60.0,
        limiter: asyncio.Semaphore | None = None,
    ) -> None:
        self.agent = Agent(model, output_type=ReasoningOutput, instructions=instructions, toolsets=list(toolsets))
        self.model_name = model_name
        self.provider = provider
        self.max_requests = max_requests
        self.timeout = timeout
        self.limiter = limiter or asyncio.Semaphore(2)

    async def reason(self, *, incident_id: int, state: Any, questions: Sequence[Question]) -> ReasoningReply:
        prompt = user_prompt(incident_id, state, questions)
        try:
            async with self.limiter:
                result = await asyncio.wait_for(
                    self.agent.run(prompt, usage_limits=UsageLimits(request_limit=self.max_requests)), self.timeout)
        except asyncio.TimeoutError as exc:
            raise AiError("provider_unavailable", retryable=True) from exc
        except UsageLimitExceeded as exc:
            # Dio todas las vueltas permitidas sin concluir: repetirlo daría lo mismo.
            raise AiError("reasoning_limit_exceeded") from exc
        except ModelHTTPError as exc:
            raise _model_error(exc.status_code) from exc
        except ModelAPIError as exc:
            raise AiError("provider_unavailable", retryable=True) from exc
        except UnexpectedModelBehavior as exc:
            raise AiError("invalid_model_output", retryable=True) from exc
        except (MCPError, FastMCPError, httpx2.HTTPError, OSError) as exc:
            logger.warning("Las herramientas del razonador no respondieron (%s)", type(exc).__name__)
            raise AiError("tools_unavailable", retryable=True) from exc
        usage = result.usage
        calls = tool_calls(result.all_messages())
        logger.info("Razonador %s: condiciones=%d pedidos=%s herramientas=%s tokens entrada=%s salida=%s",
                    self.model_name, len(questions), usage.requests, list(calls), usage.input_tokens,
                    usage.output_tokens)
        return ReasoningReply(
            conclusions=tuple(Conclusion(item.code, item.verdict, item.rationale, tuple(item.sources))
                              for item in result.output.conclusions),
            tool_calls=calls, provider=self.provider, model=self.model_name,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, requests=usage.requests,
        )


def _model_error(status: int) -> AiError:
    if status in {401, 402, 403}:
        return AiError("provider_misconfigured")
    if status in {408, 429} or status >= 500:
        return AiError("provider_unavailable", retryable=True)
    return AiError("provider_rejected")
