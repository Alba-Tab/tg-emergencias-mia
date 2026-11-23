"""Revisión síncrona del protocolo de despacho: propuestas para una persona, o un error con `retryable`."""

import logging
import time

from fastapi import APIRouter, Depends

from app.adapters.inbound.http.dependencies import authorize, get_services
from app.core.composition import Services
from app.domain.errors import AiError
from app.schemas.protocol_review import ProtocolReviewRequest, protocol_review_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", dependencies=[Depends(authorize)])


@router.post("/protocol-reviews")
async def review(request: ProtocolReviewRequest, services: Services = Depends(get_services)) -> dict:
    if services.review is None:
        raise AiError("service_not_configured")
    started = time.monotonic()
    outcome = "error"
    try:
        result = await services.review.execute(
            str(request.jobId), request.protocol, [fact.to_domain() for fact in request.facts],
            [excerpt.to_domain() for excerpt in request.normExcerpts])
        outcome = "ok"
        return protocol_review_response(result)
    except AiError as exc:
        outcome = exc.code
        raise
    finally:
        logger.info("protocol review job=%s facts=%d norm=%d outcome=%s ms=%d", request.jobId, len(request.facts),
                    len(request.normExcerpts), outcome, (time.monotonic() - started) * 1000)
