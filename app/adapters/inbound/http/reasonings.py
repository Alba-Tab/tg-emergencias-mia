"""Razonamiento síncrono sobre condiciones en duda: una conclusión por condición, o un error con `retryable`."""

import logging
import time

from fastapi import APIRouter, Depends

from app.adapters.inbound.http.dependencies import authorize, get_services
from app.core.composition import Services
from app.domain.errors import AiError
from app.schemas.reasoning import ReasoningRequest, reasoning_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", dependencies=[Depends(authorize)])


@router.post("/reasonings")
async def reason(request: ReasoningRequest, services: Services = Depends(get_services)) -> dict:
    if services.reason is None:
        raise AiError("service_not_configured")
    started = time.monotonic()
    outcome = "error"
    try:
        result = await services.reason.execute(str(request.jobId), request.incidentId, request.state,
                                               request.to_domain())
        outcome = "ok"
        return reasoning_response(result)
    except AiError as exc:
        outcome = exc.code
        raise
    finally:
        # Sin el estado ni las conclusiones: llevan lo que se sabe del caso.
        logger.info("reasoning job=%s incident=%s questions=%d outcome=%s ms=%d", request.jobId,
                    request.incidentId, len(request.questions), outcome, (time.monotonic() - started) * 1000)
