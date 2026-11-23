"""Composición de casos de uso y adaptadores a partir de la configuración."""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass

import httpx

from app.adapters.outbound.media.ffmpeg_preparer import FfmpegPreparer
from app.adapters.outbound.providers.openrouter_decisions import OpenRouterDecisionModel
from app.adapters.outbound.providers.openrouter_model import OpenRouterModel
from app.adapters.outbound.providers.prueba_decisions import PruebaDecisionModel
from app.adapters.outbound.providers.prueba_model import PruebaModel
from app.adapters.outbound.providers.prueba_reasoner import PruebaReasoner
from app.adapters.outbound.providers.pydantic_ai_reasoner import PydanticAiReasoner
from app.adapters.outbound.storage.http_evidence_reader import HttpEvidenceReader
from app.application.pipelines.media import (
    AUDIO_MIME_TYPES, IMAGE_MIME_TYPES, VIDEO_MIME_TYPES, MediaPolicy, SilencePolicy,
)
from app.application.ports.media_preparer import MediaPreparer
from app.application.ports.multimodal_model import MultimodalModel
from app.application.ports.reasoner import Reasoner
from app.application.prompts import load_prompt
from app.application.structured_output import AudioOutput, EvidenceOutput, VideoOutput
from app.application.ports.decision_model import DecisionModel
from app.application.use_cases.analyze_evidence import AnalyzeEvidence, FrameSampling, ModalityProfile
from app.application.use_cases.evaluate_conditions import EvaluateConditions
from app.application.use_cases.reason_about_conditions import ReasonAboutConditions
from app.application.use_cases.synthesize_summary import SynthesizeIncidentSummary
from app.core.config import Settings
from app.domain.evidence_reference import Modality

logger = logging.getLogger(__name__)


def build_preparer(config: Settings) -> MediaPreparer | None:
    """ffprobe/ffmpeg si están instalados; sin ellos la duración se lee de la cabecera, el video va entero
    y no hay control de silencio."""
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        logger.warning("ffmpeg/ffprobe no están instalados: duración por cabecera, video completo "
                       "y sin control de silencio del audio")
        return None
    return FfmpegPreparer(config.ffmpeg_timeout_seconds, config.max_concurrent_model_calls)


@dataclass(frozen=True, slots=True)
class Services:
    analyze: AnalyzeEvidence
    synthesize: SynthesizeIncidentSummary
    evaluate: EvaluateConditions | None = None
    reason: ReasonAboutConditions | None = None


def build_reasoner(config: Settings) -> Reasoner:
    """El agente que razona sobre las condiciones en duda. Sus herramientas las sirve el backend por MCP."""
    if config.provider == "prueba":
        return PruebaReasoner()
    # Se importan acá: son las dependencias más pesadas del servicio y con el proveedor de prueba no hacen falta.
    from pydantic_ai.mcp import MCPToolset
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    name = config.reasoning_model or config.openrouter_model
    toolsets = []
    if config.backend_mcp_url and config.backend_mcp_token:
        toolsets.append(MCPToolset(
            config.backend_mcp_url,
            headers={"Authorization": f"Bearer {config.backend_mcp_token}"},
            init_timeout=config.backend_mcp_timeout_seconds,
            read_timeout=config.backend_mcp_timeout_seconds,
        ))
    else:
        logger.warning("Sin AI_BACKEND_MCP_URL y AI_BACKEND_MCP_TOKEN: el razonador trabaja solo con el estado "
                       "recibido, sin herramientas")
    return PydanticAiReasoner(
        OpenRouterModel(name, provider=OpenRouterProvider(api_key=config.openrouter_api_key)),
        load_prompt("reasoning_v1").text,
        model_name=name, toolsets=toolsets, max_requests=config.reasoning_max_requests,
        timeout=config.reasoning_timeout_seconds,
    )


def build_services(config: Settings, client: httpx.AsyncClient) -> Services | None:
    """`None` si falta la clave de OpenRouter: el servicio arranca, pero responde 503.

    Con `provider="prueba"` no hace falta clave: todas las tareas usan el analizador de prueba.
    """
    if config.provider == "openrouter" and not config.openrouter_api_key:
        return None
    limiter = asyncio.Semaphore(config.max_concurrent_model_calls)

    def model(name: str | None) -> MultimodalModel:
        if config.provider == "prueba":
            return PruebaModel()
        return OpenRouterModel(config.openrouter_api_key, name or config.openrouter_model, client,
                               limiter=limiter, timeout=config.provider_timeout_seconds)

    profiles = {
        Modality.IMAGE: ModalityProfile(
            MediaPolicy(Modality.IMAGE, IMAGE_MIME_TYPES, config.image_max_bytes),
            load_prompt("image_v4"), EvidenceOutput, model(config.image_model),
        ),
        Modality.AUDIO: ModalityProfile(
            MediaPolicy(Modality.AUDIO, AUDIO_MIME_TYPES, config.audio_max_bytes, config.audio_max_seconds),
            load_prompt("audio_v4"), AudioOutput, model(config.audio_model),
        ),
        Modality.VIDEO: ModalityProfile(
            MediaPolicy(Modality.VIDEO, VIDEO_MIME_TYPES, config.video_max_bytes, config.video_max_seconds),
            load_prompt("video_v4"), VideoOutput, model(config.video_model),
            FrameSampling(load_prompt("video_v3"), config.video_max_frames, config.video_scene_threshold,
                          config.video_fallback_to_full) if config.video_mode == "frames" else None,
        ),
    }
    decisions: DecisionModel = PruebaDecisionModel() if config.provider == "prueba" else OpenRouterDecisionModel(
        config.openrouter_api_key, config.decision_model, client, limiter=limiter,
        timeout=config.decision_timeout_seconds)
    reader = HttpEvidenceReader(client, timeout=config.download_timeout_seconds)
    return Services(
        analyze=AnalyzeEvidence(
            reader, profiles, preparer=build_preparer(config),
            silence=SilencePolicy(config.silence_max_volume_db, config.silence_min_audible_seconds),
        ),
        synthesize=SynthesizeIncidentSummary(model(config.summary_model), load_prompt("summary_v2")),
        evaluate=EvaluateConditions(decisions),
        reason=ReasonAboutConditions(build_reasoner(config),
                                     None if config.provider == "prueba" else load_prompt("reasoning_v1").version),
    )
