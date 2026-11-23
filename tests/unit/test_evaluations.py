import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import httpx

from app.adapters.inbound.http.dependencies import get_services, get_settings
from app.adapters.outbound.providers.openrouter_decisions import ENDPOINT, OpenRouterDecisionModel
from app.adapters.outbound.providers.prueba_decisions import PruebaDecisionModel
from app.application.ports.decision_model import DecisionReply
from app.application.use_cases.evaluate_conditions import MAX_STATE_CHARS, EvaluateConditions
from app.core.composition import Services, build_services
from app.core.config import Settings
from app.domain.errors import AiError
from app.domain.evaluation import Question
from app.main import app

NOW = datetime(2026, 11, 22, tzinfo=timezone.utc)
SLEEP = "app.adapters.outbound.providers.openrouter_decisions.asyncio.sleep"
TRAPPED = Question("PERSONA_ATRAPADA", "Hay una persona atrapada")
CONFLICT = Question("FUENTES_CONTRADICTORIAS", "Las fuentes se contradicen")
STATE = {"tipoDeSuceso": "ACCIDENTE_DE_TRANSITO", "respuestasDelTestigo": {"PERSONA_ATRAPADA": "SI"}}


class FakeDecisions:
    def __init__(self, probabilities=None, error=None):
        self.probabilities = probabilities or {}
        self.error = error
        self.seen = []

    async def decide(self, *, state, questions):
        self.seen.append((state, tuple(questions)))
        if self.error:
            raise self.error
        return DecisionReply({q.code: self.probabilities.get(q.code, 0.5) for q in questions}, "openrouter",
                             "typesafe/jev-1.13", input_tokens=120, cost=0.00003)


class OpenRouterDecisionTests(unittest.IsolatedAsyncioTestCase):
    async def decide(self, handler, questions=(TRAPPED, CONFLICT)):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch(SLEEP):
                return await OpenRouterDecisionModel("secret", "typesafe/jev-1.13", client).decide(
                    state=STATE, questions=questions)

    async def assert_code(self, handler, code, retryable):
        with self.assertRaises(AiError) as caught:
            await self.decide(handler)
        self.assertEqual((caught.exception.code, caught.exception.retryable), (code, retryable))

    @staticmethod
    def answers(**values):
        return {"model": "typesafe/jev-1.13", "answers": {code: {"type": "noul", "noul": value}
                                                            for code, value in values.items()},
                "usage": {"input_tokens": 428, "output_tokens": 73, "cost": 0.00003}}

    async def test_sends_each_condition_as_a_noul_question(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=self.answers(PERSONA_ATRAPADA=0.91, FUENTES_CONTRADICTORIAS=0.12))

        reply = await self.decide(handler)

        self.assertEqual(str(seen[0].url), ENDPOINT)
        self.assertEqual(seen[0].headers["Authorization"], "Bearer secret")
        self.assertEqual(json.loads(seen[0].content), {
            "model": "typesafe/jev-1.13",
            "state": STATE,
            "questions": {
                "PERSONA_ATRAPADA": {"type": "noul", "instructions": "Hay una persona atrapada"},
                "FUENTES_CONTRADICTORIAS": {"type": "noul", "instructions": "Las fuentes se contradicen"},
            },
        })
        self.assertEqual(reply.probabilities, {"PERSONA_ATRAPADA": 0.91, "FUENTES_CONTRADICTORIAS": 0.12})
        self.assertEqual((reply.provider, reply.model, reply.input_tokens, reply.cost),
                         ("openrouter", "typesafe/jev-1.13", 428, 0.00003))

    async def test_a_missing_or_out_of_range_answer_is_not_trusted(self):
        for body in (self.answers(PERSONA_ATRAPADA=0.9),  # falta una
                     self.answers(PERSONA_ATRAPADA=1.4, FUENTES_CONTRADICTORIAS=0.1),
                     self.answers(PERSONA_ATRAPADA=True, FUENTES_CONTRADICTORIAS=0.1),
                     self.answers(PERSONA_ATRAPADA="alta", FUENTES_CONTRADICTORIAS=0.1),
                     {"answers": None}, {}):
            await self.assert_code(lambda _, body=body: httpx.Response(200, json=body), "invalid_model_output", True)
        await self.assert_code(lambda _: httpx.Response(200, content=b"no es json"), "invalid_model_output", True)

    async def test_error_mapping(self):
        await self.assert_code(lambda _: httpx.Response(401), "provider_misconfigured", False)
        await self.assert_code(lambda _: httpx.Response(402), "provider_misconfigured", False)
        await self.assert_code(lambda _: httpx.Response(400), "provider_rejected", False)
        await self.assert_code(lambda _: httpx.Response(200, json={"error": {"message": "x"}}),
                               "provider_unavailable", True)

    async def test_retries_transient_failures_once(self):
        calls = []

        def always_busy(_):
            calls.append(1)
            return httpx.Response(429)

        await self.assert_code(always_busy, "provider_unavailable", True)
        self.assertEqual(len(calls), 2)

        calls.clear()

        def busy_then_ok(_):
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json=self.answers(PERSONA_ATRAPADA=0.8, FUENTES_CONTRADICTORIAS=0.2))

        reply = await self.decide(busy_then_ok)
        self.assertEqual(reply.probabilities["PERSONA_ATRAPADA"], 0.8)
        self.assertEqual(len(calls), 2)

        def unreachable(_):
            raise httpx.ConnectError("sin red")

        await self.assert_code(unreachable, "provider_unavailable", True)


class PruebaDecisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_probability_comes_only_from_the_witness_answers(self):
        state = {"respuestasDelTestigo": {"PERSONA_ATRAPADA": "SI", "NO_RESPIRA": "NO", "HAY_FUEGO": "NO_SE"}}
        questions = [TRAPPED, Question("NO_RESPIRA", "La persona no respira"),
                     Question("HAY_FUEGO", "Hay fuego"), CONFLICT]

        reply = await PruebaDecisionModel().decide(state=state, questions=questions)

        self.assertEqual(reply.probabilities, {"PERSONA_ATRAPADA": 0.95, "NO_RESPIRA": 0.05, "HAY_FUEGO": 0.5,
                                               "FUENTES_CONTRADICTORIAS": 0.5})
        self.assertEqual((reply.provider, reply.model), ("prueba", "decisor-de-prueba"))

    async def test_a_state_without_answers_knows_nothing(self):
        for state in ({}, {"respuestasDelTestigo": "SI"}, {"respuestasDelTestigo": None}):
            reply = await PruebaDecisionModel().decide(state=state, questions=[TRAPPED])
            self.assertEqual(reply.probabilities, {"PERSONA_ATRAPADA": 0.5})

    async def test_prueba_provider_needs_no_key(self):
        async with httpx.AsyncClient() as client:
            services = build_services(Settings(provider="prueba", openrouter_api_key=None), client)
        result = await services.evaluate.execute("job", 3, STATE, [TRAPPED])
        self.assertEqual((result.answers[0].probability, result.provider), (0.95, "prueba"))


class EvaluateConditionsTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_one_answer_per_question_in_the_same_order(self):
        model = FakeDecisions({"PERSONA_ATRAPADA": 0.91, "FUENTES_CONTRADICTORIAS": 0.12})

        result = await EvaluateConditions(model, lambda: NOW).execute("job-1", 7, STATE, [CONFLICT, TRAPPED])

        self.assertEqual([(a.code, a.probability) for a in result.answers],
                         [("FUENTES_CONTRADICTORIAS", 0.12), ("PERSONA_ATRAPADA", 0.91)])
        self.assertEqual((result.job_id, result.incident_id, result.generated_at), ("job-1", 7, NOW))
        self.assertEqual((result.provider, result.model, result.input_tokens, result.cost),
                         ("openrouter", "typesafe/jev-1.13", 120, 0.00003))
        self.assertEqual(model.seen, [(STATE, (CONFLICT, TRAPPED))])

    async def test_an_oversized_state_is_rejected_before_calling_the_model(self):
        model = FakeDecisions()
        with self.assertRaises(AiError) as caught:
            await EvaluateConditions(model).execute("job", 1, {"texto": "x" * MAX_STATE_CHARS}, [TRAPPED])
        self.assertEqual((caught.exception.code, caught.exception.retryable), ("state_too_large", False))
        self.assertEqual(model.seen, [])


class EvaluationApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = FakeDecisions({"PERSONA_ATRAPADA": 0.91})
        app.dependency_overrides[get_settings] = lambda: Settings(service_token="service-secret")
        app.dependency_overrides[get_services] = lambda: Services(
            None, None, EvaluateConditions(self.model, lambda: NOW))
        self.payload = {
            "jobId": str(uuid4()), "incidentId": 3, "state": STATE,
            "questions": [{"code": "PERSONA_ATRAPADA", "statement": "  Hay una persona atrapada  "},
                          {"code": "FUENTES_CONTRADICTORIAS", "statement": "Las fuentes se contradicen"}],
        }

    def tearDown(self):
        app.dependency_overrides.clear()

    async def post(self, payload, token="service-secret"):
        headers = {} if token is None else {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/evaluations", json=payload, headers=headers)

    async def test_success(self):
        response = await self.post(self.payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "jobId": self.payload["jobId"],
            "incidentId": 3,
            "schemaVersion": "condition-evaluation.v1",
            "answers": [{"code": "PERSONA_ATRAPADA", "probability": 0.91},
                        {"code": "FUENTES_CONTRADICTORIAS", "probability": 0.5}],
            "provenance": {"provider": "openrouter", "model": "typesafe/jev-1.13",
                           "generatedAt": NOW.isoformat(), "method": "decision_model"},
            "usage": {"inputTokens": 120, "cost": 0.00003},
        })
        state, questions = self.model.seen[0]
        self.assertEqual(state, STATE)
        self.assertEqual(questions, (TRAPPED, CONFLICT))

    async def test_authentication(self):
        for token in (None, "wrong"):
            response = await self.post(self.payload, token)
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json(), {"errorCode": "unauthorized", "retryable": False})
        self.assertEqual(self.model.seen, [])

    async def test_validation(self):
        repeated = [self.payload["questions"][0], self.payload["questions"][0]]
        for field, value in (("jobId", "bad"), ("incidentId", 0), ("state", "texto suelto"), ("questions", []),
                             ("questions", [{"code": "con espacios", "statement": "Algo"}]),
                             ("questions", [{"code": "CODIGO", "statement": ""}]),
                             ("questions", repeated), ("extra", 1)):
            response = await self.post({**self.payload, field: value})
            self.assertEqual(response.status_code, 422, field)
            self.assertEqual(response.json()["errorCode"], "invalid_request")
        self.assertEqual(self.model.seen, [])

    async def test_error_mapping(self):
        for error, status in ((AiError("provider_unavailable", retryable=True), 503),
                              (AiError("provider_misconfigured"), 503),
                              (AiError("provider_rejected"), 502),
                              (AiError("invalid_model_output", retryable=True), 503)):
            self.model.error = error
            response = await self.post(self.payload)
            self.assertEqual(response.status_code, status, error.code)
            self.assertEqual(response.json(), {"errorCode": error.code, "retryable": error.retryable})

    async def test_state_too_large(self):
        response = await self.post({**self.payload, "state": {"texto": "x" * MAX_STATE_CHARS}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"errorCode": "state_too_large", "retryable": False})

    async def test_without_an_evaluator_the_service_is_not_configured(self):
        app.dependency_overrides[get_services] = lambda: Services(None, None)
        response = await self.post(self.payload)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["errorCode"], "service_not_configured")


if __name__ == "__main__":
    unittest.main()
