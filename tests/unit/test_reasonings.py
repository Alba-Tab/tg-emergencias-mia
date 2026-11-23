import json
import os
import unittest
from datetime import datetime, timezone
from uuid import uuid4

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import httpx
import httpx2
from pydantic_ai import models
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from app.adapters.inbound.http.dependencies import get_services, get_settings
from app.adapters.outbound.providers.prueba_reasoner import PruebaReasoner
from app.adapters.outbound.providers.pydantic_ai_reasoner import PydanticAiReasoner, user_prompt
from app.application.ports.reasoner import ReasoningReply
from app.application.use_cases.evaluate_conditions import MAX_STATE_CHARS
from app.application.use_cases.reason_about_conditions import ReasonAboutConditions
from app.core.composition import Services, build_services
from app.core.config import Settings
from app.domain.errors import AiError
from app.domain.evaluation import Question
from app.domain.reasoning import MET, NO_CONCLUSION, NOT_MET, UNGROUNDED, UNKNOWN, Conclusion, ground
from app.main import app
from tests.mcp_backend import TOKEN, FakeBackend

models.ALLOW_MODEL_REQUESTS = False

NOW = datetime(2026, 11, 22, tzinfo=timezone.utc)
TRAPPED = Question("PERSONA_ATRAPADA", "Hay una persona atrapada")
CONFLICT = Question("FUENTES_CONTRADICTORIAS", "Las fuentes se contradicen")
STATE = {"tipoDeSuceso": {"codigo": "ACCIDENTE_DE_TRANSITO"}, "respuestasDelTestigo": {"PERSONA_ATRAPADA": "NO_SE"}}
INSTRUCTIONS = "Razona sobre las condiciones."


def conclusions(*items):
    return {"conclusions": [{"code": code, "verdict": verdict, "rationale": rationale, "sources": list(sources)}
                            for code, verdict, rationale, sources in items]}


def final(info, payload):
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])


class GroundingTests(unittest.TestCase):
    def test_keeps_a_justified_conclusion(self):
        result = ground([Conclusion("PERSONA_ATRAPADA", MET, "  La foto muestra   una puerta aplastada. ",
                                    ("evidencia 12", " "))], [TRAPPED])

        self.assertEqual(result, (Conclusion("PERSONA_ATRAPADA", MET, "La foto muestra una puerta aplastada.",
                                             ("evidencia 12",)),))

    def test_one_conclusion_per_question_in_the_same_order(self):
        result = ground([
            Conclusion("FUENTES_CONTRADICTORIAS", NOT_MET, "Las dos alertas coinciden.", ("alertas",)),
            Conclusion("OTRA_QUE_NADIE_PIDIO", MET, "Inventada.", ("nada",)),
            Conclusion("FUENTES_CONTRADICTORIAS", MET, "Repetida.", ("nada",)),
        ], [TRAPPED, CONFLICT])

        self.assertEqual([item.code for item in result], ["PERSONA_ATRAPADA", "FUENTES_CONTRADICTORIAS"])
        self.assertEqual((result[0].verdict, result[0].rationale), (UNKNOWN, NO_CONCLUSION))
        self.assertEqual((result[1].verdict, result[1].rationale), (NOT_MET, "Las dos alertas coinciden."))

    def test_a_verdict_without_a_reason_or_a_source_is_not_taken(self):
        result = ground([
            Conclusion("PERSONA_ATRAPADA", MET, "Seguro que sí.", ()),
            Conclusion("FUENTES_CONTRADICTORIAS", NOT_MET, "  ", ("alertas",)),
        ], [TRAPPED, CONFLICT])

        self.assertEqual([item.verdict for item in result], [UNKNOWN, UNKNOWN])
        self.assertEqual(result[0].rationale, f"Seguro que sí. {UNGROUNDED}")
        self.assertEqual(result[1].rationale, UNGROUNDED)

    def test_an_unknown_needs_no_source_and_a_made_up_verdict_is_unknown(self):
        result = ground([
            Conclusion("PERSONA_ATRAPADA", UNKNOWN, "La foto no muestra el interior.", ()),
            Conclusion("FUENTES_CONTRADICTORIAS", "maybe", "Quizás.", ("alertas",)),
        ], [TRAPPED, CONFLICT])

        self.assertEqual((result[0].verdict, result[0].rationale), (UNKNOWN, "La foto no muestra el interior."))
        self.assertEqual((result[1].verdict, result[1].rationale), (UNKNOWN, NO_CONCLUSION))

    def test_long_texts_are_cut(self):
        result = ground([Conclusion("PERSONA_ATRAPADA", MET, "a" * 5000, tuple(f"fuente {n} " + "x" * 400
                                                                               for n in range(20)))], [TRAPPED])

        self.assertEqual(len(result[0].rationale), 600)
        self.assertEqual(len(result[0].sources), 6)
        self.assertTrue(all(len(source) <= 160 for source in result[0].sources))


class PydanticAiReasonerTests(unittest.IsolatedAsyncioTestCase):
    """El agente de verdad, con un modelo guionado y sus herramientas servidas por un servidor MCP en memoria."""

    async def reason(self, script, backend=None, questions=(TRAPPED, CONFLICT), token=TOKEN, **options):
        backend = backend or FakeBackend()
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=backend.app),
                                      headers={"Authorization": f"Bearer {token}"}) as client:
            toolset = MCPToolset("http://backend.invalid/mcp", http_client=client, init_timeout=5, read_timeout=5)
            reasoner = PydanticAiReasoner(FunctionModel(script), INSTRUCTIONS, model_name="modelo/guionado",
                                          toolsets=[toolset], **options)
            return await reasoner.reason(incident_id=7, state=STATE, questions=questions)

    async def assert_code(self, script, code, retryable, **options):
        with self.assertRaises(AiError) as caught:
            await self.reason(script, **options)
        self.assertEqual((caught.exception.code, caught.exception.retryable), (code, retryable))

    async def test_looks_at_the_case_through_mcp_and_concludes(self):
        backend = FakeBackend()
        seen = {}

        def script(messages, info):
            if len(messages) == 1:
                seen["tools"] = [tool.name for tool in info.function_tools]
                seen["instructions"] = info.instructions
                seen["prompt"] = messages[0].parts[-1].content
                return ModelResponse(parts=[ToolCallPart("ver_caso", {"incidenteId": 7})])
            seen["tool_result"] = next(part.content for part in messages[-1].parts
                                       if isinstance(part, ToolReturnPart))
            return final(info, conclusions(
                ("PERSONA_ATRAPADA", "met", "El análisis dice que un conductor sigue adentro.", ["evidencia 12"]),
                ("FUENTES_CONTRADICTORIAS", "unknown", "Hay una sola fuente.", []),
            ))

        reply = await self.reason(script, backend)

        self.assertEqual(seen["tools"], ["ver_caso", "buscar_en_norma"])
        self.assertIn(INSTRUCTIONS, seen["instructions"])
        self.assertEqual(json.loads(seen["prompt"]), json.loads(user_prompt(7, STATE, (TRAPPED, CONFLICT))))
        self.assertEqual(seen["tool_result"]["evidencias"][0]["resumen"], "Un conductor sigue adentro.")
        self.assertEqual(backend.calls, [("ver_caso", {"incidenteId": 7})])
        self.assertEqual(reply.conclusions[0], Conclusion("PERSONA_ATRAPADA", "met",
                                                           "El análisis dice que un conductor sigue adentro.",
                                                           ("evidencia 12",)))
        self.assertEqual(reply.conclusions[1].verdict, "unknown")
        self.assertEqual(reply.tool_calls, ("ver_caso",))
        self.assertEqual((reply.provider, reply.model, reply.requests), ("openrouter", "modelo/guionado", 2))
        self.assertGreater(reply.input_tokens, 0)

    async def test_speaks_the_stateless_json_transport_of_the_backend(self):
        backend = FakeBackend()

        def script(messages, info):
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart("buscar_en_norma", {"consulta": "persona atrapada"})])
            return final(info, conclusions())

        await self.reason(script, backend)

        # Prueba el protocolo nuevo, el servidor no lo conoce y cae al saludo de siempre. Sin sesión, no abre
        # ningún flujo ni la cierra: no hay GET ni DELETE.
        self.assertEqual(backend.methods, ["server/discover", "initialize", "notifications/initialized",
                                           "tools/list", "tools/call"])
        self.assertEqual(backend.versions[-1], "2025-06-18")
        self.assertEqual(backend.calls, [("buscar_en_norma", {"consulta": "persona atrapada"})])

    async def test_the_prompt_carries_the_state_as_data(self):
        prompt = json.loads(user_prompt(7, STATE, (TRAPPED,)))

        self.assertEqual(prompt, {"incidenteId": 7, "estado": STATE, "condiciones": [
            {"codigo": "PERSONA_ATRAPADA", "enunciado": "Hay una persona atrapada"}]})

    async def test_a_failing_tool_is_told_to_the_model_which_can_still_conclude(self):
        backend = FakeBackend(failing_tool="ver_caso")
        seen = []

        def script(messages, info):
            seen.append(" ".join(str(getattr(part, "content", "")) for part in messages[-1].parts))
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart("ver_caso", {"incidenteId": 999})])
            return final(info, conclusions(("PERSONA_ATRAPADA", "unknown", "No se pudo ver el caso.", [])))

        reply = await self.reason(script, backend, questions=(TRAPPED,))

        # El error de la herramienta le llega al modelo como dato: no corta el razonamiento.
        self.assertIn("No existe el incidente.", seen[1])
        self.assertEqual(reply.conclusions[0].verdict, "unknown")
        self.assertEqual(reply.tool_calls, ("ver_caso",))

    async def test_an_agent_that_never_finishes_is_stopped(self):
        def script(messages, info):
            return ModelResponse(parts=[ToolCallPart("ver_caso", {"incidenteId": len(messages)})])

        await self.assert_code(script, "reasoning_limit_exceeded", False, max_requests=3)

    async def test_a_slow_agent_is_cut_at_the_time_limit(self):
        import asyncio

        async def slow(messages, info):
            await asyncio.sleep(5)
            return final(info, conclusions())

        await self.assert_code(slow, "provider_unavailable", True, timeout=0.05)

    async def test_model_errors_are_mapped(self):
        for status, code, retryable in ((401, "provider_misconfigured", False), (402, "provider_misconfigured", False),
                                        (429, "provider_unavailable", True), (503, "provider_unavailable", True),
                                        (400, "provider_rejected", False)):
            def failing(messages, info, status=status):
                raise ModelHTTPError(status, "modelo/guionado")

            with self.subTest(status=status):
                await self.assert_code(failing, code, retryable)

    async def test_a_wrong_token_leaves_the_agent_without_tools(self):
        def script(messages, info):
            return final(info, conclusions())

        await self.assert_code(script, "tools_unavailable", True, token="otro-token")

    async def test_without_tools_it_reasons_with_the_state_alone(self):
        def script(messages, info):
            self.assertEqual(info.function_tools, [])
            return final(info, conclusions(("PERSONA_ATRAPADA", "not_met", "El testigo dijo que no.",
                                            ["respuesta del testigo"])))

        reply = await PydanticAiReasoner(FunctionModel(script), INSTRUCTIONS, model_name="modelo/guionado").reason(
            incident_id=7, state=STATE, questions=(TRAPPED,))

        self.assertEqual(reply.conclusions[0].verdict, "not_met")
        self.assertEqual(reply.tool_calls, ())


class PruebaReasonerTests(unittest.IsolatedAsyncioTestCase):
    async def test_concludes_only_from_the_witness_answers(self):
        state = {"respuestasDelTestigo": {"PERSONA_ATRAPADA": "SI", "FUENTES_CONTRADICTORIAS": "NO"}}
        other = Question("HAY_FUEGO", "Hay fuego")

        reply = await PruebaReasoner().reason(incident_id=7, state=state, questions=(TRAPPED, CONFLICT, other))

        self.assertEqual([item.verdict for item in reply.conclusions], [MET, NOT_MET, UNKNOWN])
        self.assertEqual(reply.conclusions[0].sources, ("respuesta del testigo",))
        self.assertEqual((reply.provider, reply.model, reply.tool_calls), ("prueba", "razonador-de-prueba", ()))

    async def test_a_state_without_answers_knows_nothing(self):
        for state in ({}, {"respuestasDelTestigo": "SI"}, []):
            reply = await PruebaReasoner().reason(incident_id=7, state=state, questions=(TRAPPED,))
            self.assertEqual(reply.conclusions[0].verdict, UNKNOWN)

    async def test_prueba_provider_needs_no_key(self):
        async with httpx.AsyncClient() as client:
            services = build_services(Settings(_env_file=None, provider="prueba"), client)

        result = await services.reason.execute("job", 7, {"respuestasDelTestigo": {"PERSONA_ATRAPADA": "SI"}},
                                               [TRAPPED])

        self.assertEqual((result.conclusions[0].verdict, result.provider, result.prompt_version),
                         (MET, "prueba", None))

    async def test_openrouter_provider_builds_the_agent_with_or_without_mcp(self):
        async with httpx.AsyncClient() as client:
            alone = build_services(Settings(_env_file=None, openrouter_api_key="clave"), client)
            with_tools = build_services(Settings(_env_file=None, openrouter_api_key="clave",
                                                 backend_mcp_url="http://backend.invalid/mcp",
                                                 backend_mcp_token=TOKEN, reasoning_model="otro/modelo"), client)

        self.assertEqual(alone.reason.prompt_version, "reasoning-v1")
        self.assertEqual(alone.reason.reasoner.model_name, Settings(_env_file=None).openrouter_model)
        self.assertEqual(len(alone.reason.reasoner.agent.toolsets), len(with_tools.reason.reasoner.agent.toolsets) - 1)
        self.assertEqual(with_tools.reason.reasoner.model_name, "otro/modelo")


class FakeReasoner:
    def __init__(self, items=(), error=None):
        self.items = tuple(items)
        self.error = error
        self.seen = []

    async def reason(self, *, incident_id, state, questions):
        self.seen.append((incident_id, state, tuple(questions)))
        if self.error:
            raise self.error
        return ReasoningReply(self.items, ("ver_caso", "buscar_casos_parecidos"), "openrouter", "modelo/real",
                              input_tokens=900, output_tokens=120, requests=3)


class ReasonAboutConditionsTests(unittest.IsolatedAsyncioTestCase):
    async def test_applies_the_grounding_rules_to_what_the_reasoner_returns(self):
        reasoner = FakeReasoner([Conclusion("PERSONA_ATRAPADA", MET, "Seguro.", ())])

        result = await ReasonAboutConditions(reasoner, "reasoning-v1", clock=lambda: NOW).execute(
            "job", 7, STATE, [TRAPPED, CONFLICT])

        self.assertEqual(reasoner.seen, [(7, STATE, (TRAPPED, CONFLICT))])
        self.assertEqual([item.verdict for item in result.conclusions], [UNKNOWN, UNKNOWN])
        self.assertEqual((result.job_id, result.incident_id, result.generated_at), ("job", 7, NOW))
        self.assertEqual((result.tool_calls, result.prompt_version, result.requests),
                         (("ver_caso", "buscar_casos_parecidos"), "reasoning-v1", 3))

    async def test_an_oversized_state_is_rejected_before_reasoning(self):
        reasoner = FakeReasoner()

        with self.assertRaises(AiError) as caught:
            await ReasonAboutConditions(reasoner).execute("job", 7, {"x": "a" * MAX_STATE_CHARS}, [TRAPPED])

        self.assertEqual(caught.exception.code, "state_too_large")
        self.assertEqual(reasoner.seen, [])


class ReasoningApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.reasoner = FakeReasoner([
            Conclusion("PERSONA_ATRAPADA", MET, "El análisis dice que un conductor sigue adentro.", ("evidencia 12",)),
        ])
        services = Services(analyze=None, synthesize=None,
                            reason=ReasonAboutConditions(self.reasoner, "reasoning-v1", clock=lambda: NOW))
        app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, service_token="service-secret")
        app.dependency_overrides[get_services] = lambda: services

    def tearDown(self):
        app.dependency_overrides.clear()

    async def post(self, payload, token="service-secret"):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/reasonings", json=payload, headers=headers)

    @staticmethod
    def payload(**changes):
        body = {"jobId": str(uuid4()), "incidentId": 7, "state": STATE, "questions": [
            {"code": "PERSONA_ATRAPADA", "statement": "Hay una persona atrapada"},
            {"code": "FUENTES_CONTRADICTORIAS", "statement": "Las fuentes se contradicen"}]}
        body.update(changes)
        return body

    async def test_success(self):
        payload = self.payload()

        response = await self.post(payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "jobId": payload["jobId"], "incidentId": 7, "schemaVersion": "condition-reasoning.v1",
            "conclusions": [
                {"code": "PERSONA_ATRAPADA", "verdict": "met",
                 "rationale": "El análisis dice que un conductor sigue adentro.", "sources": ["evidencia 12"]},
                {"code": "FUENTES_CONTRADICTORIAS", "verdict": "unknown", "rationale": NO_CONCLUSION, "sources": []},
            ],
            "toolCalls": ["ver_caso", "buscar_casos_parecidos"],
            "provenance": {"provider": "openrouter", "model": "modelo/real", "promptVersion": "reasoning-v1",
                           "generatedAt": NOW.isoformat(), "method": "agent"},
            "usage": {"inputTokens": 900, "outputTokens": 120, "requests": 3},
        })

    async def test_authentication(self):
        self.assertEqual((await self.post(self.payload(), token=None)).status_code, 401)
        self.assertEqual((await self.post(self.payload(), token="otro")).json(),
                         {"errorCode": "unauthorized", "retryable": False})
        self.assertEqual(self.reasoner.seen, [])

    async def test_validation(self):
        same = {"code": "PERSONA_ATRAPADA", "statement": "Hay una persona atrapada"}
        for changes in ({"questions": []}, {"questions": [same, same]}, {"incidentId": 0}, {"state": "texto"},
                        {"questions": [{"code": "minusculas", "statement": "Hay algo"}]},
                        {"questions": [same] * 21}, {"otro": 1}):
            with self.subTest(changes=changes):
                response = await self.post(self.payload(**changes))
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["errorCode"], "invalid_request")
        self.assertEqual(self.reasoner.seen, [])

    async def test_error_mapping(self):
        for code, retryable, status in (("reasoning_limit_exceeded", False, 422), ("tools_unavailable", True, 503),
                                        ("provider_unavailable", True, 503), ("provider_misconfigured", False, 503),
                                        ("provider_rejected", False, 502), ("invalid_model_output", True, 503)):
            self.reasoner.error = AiError(code, retryable=retryable)
            with self.subTest(code=code):
                response = await self.post(self.payload())
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json(), {"errorCode": code, "retryable": retryable})

    async def test_without_a_reasoner_the_service_is_not_configured(self):
        app.dependency_overrides[get_services] = lambda: Services(analyze=None, synthesize=None)

        response = await self.post(self.payload())

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"errorCode": "service_not_configured", "retryable": False})
