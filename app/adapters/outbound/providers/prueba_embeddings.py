"""Modelo de embeddings de prueba: implementa `EmbeddingModel` sin llamar a ningún proveedor.

Cada palabra del texto suma en una posición fija del vector, así que dos textos que comparten palabras quedan
cerca. No entiende sinónimos ni significado: sirve para probar el flujo completo sin clave ni costo, no para
buscar de verdad.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence

from app.application.ports.embedding_model import EmbeddingReply

PROVIDER = "prueba"
MODEL = "embeddings-de-prueba"
DIMENSIONS = 64
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode("ascii")
    return _WORD.findall(plain)


def _vector(text: str) -> tuple[float, ...]:
    counts = [0.0] * DIMENSIONS
    for word in _words(text):
        counts[int.from_bytes(hashlib.sha256(word.encode("ascii")).digest()[:4], "big") % DIMENSIONS] += 1.0
    norm = math.sqrt(sum(value * value for value in counts))
    if norm == 0:
        # Un texto sin palabras no puede quedar en cero: el coseno no estaría definido.
        counts[0] = norm = 1.0
    return tuple(value / norm for value in counts)


class PruebaEmbeddingModel:
    async def embed(self, *, texts: Sequence[str]) -> EmbeddingReply:
        return EmbeddingReply(vectors=tuple(_vector(text) for text in texts), provider=PROVIDER, model=MODEL)
