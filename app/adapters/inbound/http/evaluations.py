"""Evaluación síncrona de las condiciones de un incidente: la probabilidad de cada una, o un error con `retryable`."""

import logging
import time

from fastapi import APIRouter, Depends

from app.adapters.inbound.http.dependencies import authorize, get_services
from app.core.composition import Services
from app.domain.errors import AiError
from app.schemas.evaluation import EvaluationRequest, evaluation_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", dependencies=[Depends(authorize)])


@router.post("/evaluations")
async def evaluate(request: EvaluationRequest, services: Services = Depends(get_services)) -> dict:
    if services.evaluate is None:
        raise AiError("service_not_configured")
    started = time.monotonic()
    outcome = "error"
    try:
        result = await services.evaluate.execute(str(request.jobId), request.incidentId, request.state,
                                                 request.to_domain())
        outcome = "ok"
        return evaluation_response(result)
    except AiError as exc:
        outcome = exc.code
        raise
    finally:
        # Sin el estado: lleva lo que se sabe del caso y no tiene por qué quedar en los registros.
        logger.info("evaluation job=%s incident=%s questions=%d outcome=%s ms=%d", request.jobId,
                    request.incidentId, len(request.questions), outcome, (time.monotonic() - started) * 1000)
