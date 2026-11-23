"""Embeddings síncronos: un vector por texto, o un error con `retryable`."""

import logging
import time

from fastapi import APIRouter, Depends

from app.adapters.inbound.http.dependencies import authorize, get_services
from app.core.composition import Services
from app.domain.errors import AiError
from app.schemas.embedding import EmbeddingRequest, embedding_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", dependencies=[Depends(authorize)])


@router.post("/embeddings")
async def embed(request: EmbeddingRequest, services: Services = Depends(get_services)) -> dict:
    if services.embed is None:
        raise AiError("service_not_configured")
    started = time.monotonic()
    outcome = "error"
    try:
        result = await services.embed.execute(str(request.jobId), request.texts)
        outcome = "ok"
        return embedding_response(result)
    except AiError as exc:
        outcome = exc.code
        raise
    finally:
        # Sin los textos: llevan lo que se sabe de los casos y no tienen por qué quedar en los registros.
        logger.info("embedding job=%s texts=%d outcome=%s ms=%d", request.jobId, len(request.texts), outcome,
                    (time.monotonic() - started) * 1000)
