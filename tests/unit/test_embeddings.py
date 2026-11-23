import json
import math
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import httpx

from app.adapters.inbound.http.dependencies import get_services, get_settings
from app.adapters.outbound.providers.openrouter_embeddings import ENDPOINT, OpenRouterEmbeddingModel
from app.adapters.outbound.providers.prueba_embeddings import DIMENSIONS, PruebaEmbeddingModel
from app.application.ports.embedding_model import EmbeddingReply
from app.application.use_cases.embed_texts import EmbedTexts
from app.core.composition import Services, build_services
from app.core.config import Settings
from app.domain.errors import AiError
from app.main import app

NOW = datetime(2026, 11, 23, tzinfo=timezone.utc)
SLEEP = "app.adapters.outbound.providers.openrouter_embeddings.asyncio.sleep"
MODEL = "openai/text-embedding-3-small"
TEXTS = ["Hay una persona atrapada en el vehículo", "El paciente no respira"]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


class FakeEmbeddings:
    def __init__(self, error=None):
        self.error = error
        self.seen = []

    async def embed(self, *, texts):
        self.seen.append(tuple(texts))
        if self.error:
            raise self.error
        return EmbeddingReply(tuple((float(index), 1.0, 0.5) for index, _ in enumerate(texts)), "openrouter", MODEL,
                              input_tokens=17, cost=0.0000004)


class OpenRouterEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def embed(self, handler, texts=TEXTS):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch(SLEEP):
                return await OpenRouterEmbeddingModel("secret", MODEL, client).embed(texts=texts)

    async def assert_code(self, handler, code, retryable):
        with self.assertRaises(AiError) as caught:
            await self.embed(handler)
        self.assertEqual((caught.exception.code, caught.exception.retryable), (code, retryable))

    @staticmethod
    def body(*vectors, order=None):
        indexes = order or range(len(vectors))
        return {"object": "list", "model": MODEL,
                "data": [{"object": "embedding", "index": index, "embedding": list(vector)}
                         for index, vector in zip(indexes, vectors)],
                "usage": {"prompt_tokens": 17, "total_tokens": 17, "cost": 0.0000004}}

    async def test_sends_all_the_texts_in_one_request(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=self.body([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]))

        reply = await self.embed(handler)

        self.assertEqual(str(seen[0].url), ENDPOINT)
        self.assertEqual(seen[0].headers["Authorization"], "Bearer secret")
        self.assertEqual(json.loads(seen[0].content), {"model": MODEL, "input": TEXTS, "encoding_format": "float"})
        self.assertEqual(reply.vectors, ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)))
        self.assertEqual((reply.provider, reply.model, reply.input_tokens, reply.cost),
                         ("openrouter", MODEL, 17, 0.0000004))

    async def test_vectors_come_back_in_the_order_of_the_texts(self):
        reply = await self.embed(lambda _: httpx.Response(200, json=self.body([0.4, 0.5], [0.1, 0.2], order=[1, 0])))

        self.assertEqual(reply.vectors, ((0.1, 0.2), (0.4, 0.5)))

    async def test_an_incomplete_or_malformed_answer_is_not_trusted(self):
        for body in (self.body([0.1, 0.2]),  # falta un vector
                     self.body([0.1, 0.2], [0.3]),  # distinto tamaño
                     self.body([0.1, 0.2], []),
                     self.body([0.1, 0.2], ["a", "b"]),
                     self.body([0.1, 0.2], [True, False]),
                     self.body([0.1, 0.2], [0.3, 0.4], order=[0, 0]),
                     {"data": None}, {}):
            await self.assert_code(lambda _, body=body: httpx.Response(200, json=body), "invalid_model_output", True)
        await self.assert_code(lambda _: httpx.Response(200, content=b"no es json"), "invalid_model_output", True)

    async def test_error_mapping(self):
        await self.assert_code(lambda _: httpx.Response(401), "provider_misconfigured", False)
        await self.assert_code(lambda _: httpx.Response(402), "provider_misconfigured", False)
        await self.assert_code(lambda _: httpx.Response(404), "provider_rejected", False)
        await self.assert_code(lambda _: httpx.Response(200, json={"error": {"code": 502, "message": "x"}}),
                               "provider_unavailable", True)

    async def test_retries_transient_failures_once(self):
        calls = []

        def busy_then_ok(_):
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(529)
            return httpx.Response(200, json=self.body([0.1, 0.2], [0.3, 0.4]))

        reply = await self.embed(busy_then_ok)
        self.assertEqual(len(reply.vectors), 2)
        self.assertEqual(len(calls), 2)

        calls.clear()

        def always_busy(_):
            calls.append(1)
            return httpx.Response(429)

        await self.assert_code(always_busy, "provider_unavailable", True)
        self.assertEqual(len(calls), 2)

        def unreachable(_):
            raise httpx.ConnectError("sin red")

        await self.assert_code(unreachable, "provider_unavailable", True)


class PruebaEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def test_texts_that_share_words_end_up_closer(self):
        reply = await PruebaEmbeddingModel().embed(texts=[
            "Hay una persona atrapada en el vehículo",
            "Una persona quedó atrapada en su vehículo",
            "Dolor de pecho con sudor frío",
        ])

        trapped, also_trapped, chest = reply.vectors
        self.assertEqual({len(vector) for vector in reply.vectors}, {DIMENSIONS})
        self.assertGreater(cosine(trapped, also_trapped), cosine(trapped, chest))
        self.assertEqual((reply.provider, reply.model), ("prueba", "embeddings-de-prueba"))

    async def test_the_same_text_always_gives_the_same_unit_vector(self):
        first = await PruebaEmbeddingModel().embed(texts=["No respira", "¿¡!?"])
        second = await PruebaEmbeddingModel().embed(texts=["No respira", "¿¡!?"])

        self.assertEqual(first.vectors, second.vectors)
        for vector in first.vectors:
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0)

    async def test_prueba_provider_needs_no_key(self):
        async with httpx.AsyncClient() as client:
            services = build_services(Settings(provider="prueba", openrouter_api_key=None), client)
        result = await services.embed.execute("job", ["No respira"])
        self.assertEqual((result.dimensions, result.provider), (DIMENSIONS, "prueba"))


class EmbedTextsTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_one_vector_per_text_with_the_model_that_made_them(self):
        model = FakeEmbeddings()

        result = await EmbedTexts(model, lambda: NOW).execute("job-1", TEXTS)

        self.assertEqual(result.vectors, ((0.0, 1.0, 0.5), (1.0, 1.0, 0.5)))
        self.assertEqual((result.job_id, result.dimensions, result.generated_at), ("job-1", 3, NOW))
        self.assertEqual((result.provider, result.model, result.input_tokens, result.cost),
                         ("openrouter", MODEL, 17, 0.0000004))
        self.assertEqual(model.seen, [tuple(TEXTS)])


class EmbeddingApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = FakeEmbeddings()
        app.dependency_overrides[get_settings] = lambda: Settings(service_token="service-secret")
        app.dependency_overrides[get_services] = lambda: Services(
            None, None, embed=EmbedTexts(self.model, lambda: NOW))
        self.payload = {"jobId": str(uuid4()), "texts": ["  Hay una persona atrapada  ", "No respira"]}

    def tearDown(self):
        app.dependency_overrides.clear()

    async def post(self, payload, token="service-secret"):
        headers = {} if token is None else {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/embeddings", json=payload, headers=headers)

    async def test_success(self):
        response = await self.post(self.payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "jobId": self.payload["jobId"],
            "schemaVersion": "text-embedding.v1",
            "dimensions": 3,
            "vectors": [[0.0, 1.0, 0.5], [1.0, 1.0, 0.5]],
            "provenance": {"provider": "openrouter", "model": MODEL, "generatedAt": NOW.isoformat(),
                           "method": "embedding_model"},
            "usage": {"inputTokens": 17, "cost": 0.0000004},
        })
        # Los textos llegan al modelo sin los espacios de los bordes.
        self.assertEqual(self.model.seen, [("Hay una persona atrapada", "No respira")])

    async def test_requires_the_service_token(self):
        self.assertEqual((await self.post(self.payload, token=None)).status_code, 401)
        self.assertEqual((await self.post(self.payload, token="otro")).status_code, 401)
        self.assertEqual(self.model.seen, [])

    async def test_rejects_invalid_requests_without_calling_the_model(self):
        for payload in ({**self.payload, "texts": []},
                        {**self.payload, "texts": ["   "]},
                        {**self.payload, "texts": ["x" * 8001]},
                        {**self.payload, "texts": ["x"] * 65},
                        {**self.payload, "extra": 1},
                        {"texts": ["x"]}):
            response = await self.post(payload)
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["errorCode"], "invalid_request")
        self.assertEqual(self.model.seen, [])

    async def test_provider_errors_keep_their_code_and_retryable(self):
        self.model.error = AiError("provider_unavailable", retryable=True)
        response = await self.post(self.payload)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"errorCode": "provider_unavailable", "retryable": True})

    async def test_without_an_embedding_model_the_service_answers_503(self):
        app.dependency_overrides[get_services] = lambda: Services(None, None)
        response = await self.post(self.payload)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["errorCode"], "service_not_configured")
