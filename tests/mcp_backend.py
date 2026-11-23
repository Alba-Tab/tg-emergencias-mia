"""Un servidor MCP mínimo para las pruebas, espejo del que sirve el backend.

Habla el transporte HTTP sin estado y contesta siempre con un JSON, nunca con un flujo de eventos:

- un pedido (trae `id`) recibe un 200 con su respuesta JSON-RPC;
- una notificación (sin `id`) recibe un 202 sin cuerpo;
- un pedido con una versión del protocolo que no habla, como el `server/discover` con que prueba un cliente
  nuevo, recibe un 400 con el error -32022 y la lista de las que sí habla: así el cliente cae al saludo
  `initialize` de siempre;
- un método que no conoce recibe el error -32601;
- GET y DELETE reciben 405, porque no hay flujo ni sesión que abrir o cerrar.

Si el backend cambia cómo responde, este archivo tiene que cambiar igual: es lo que prueba que el cliente del
razonador lo entiende.
"""

from __future__ import annotations

import json
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

SUPPORTED_VERSIONS = ("2025-03-26", "2025-06-18")
TOKEN = "token-del-agente"

TOOLS = [
    {
        "name": "ver_caso",
        "title": "Ver el caso",
        "description": "Todo lo que se sabe de un incidente.",
        "inputSchema": {"type": "object", "properties": {"incidenteId": {"type": "integer"}},
                        "required": ["incidenteId"]},
    },
    {
        "name": "buscar_en_norma",
        "title": "Buscar en la norma",
        "description": "Qué dice el protocolo sobre un tema.",
        "inputSchema": {"type": "object", "properties": {"consulta": {"type": "string"}},
                        "required": ["consulta"]},
    },
]


class FakeBackend:
    """Anota lo que le llega y contesta lo que la prueba le prepara."""

    def __init__(self, case: dict[str, Any] | None = None, failing_tool: str | None = None) -> None:
        self.case = case or {"incidenteId": 7, "evidencias": [{"id": 12, "resumen": "Un conductor sigue adentro."}]}
        self.failing_tool = failing_tool
        self.methods: list[str] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.versions: list[str | None] = []
        self.app = Starlette(routes=[
            Route("/mcp", self._post, methods=["POST"]),
            Route("/mcp", self._not_allowed, methods=["GET", "DELETE"]),
        ])

    async def _not_allowed(self, request: Request) -> Response:
        self.methods.append(request.method)
        return Response(status_code=405)

    async def _post(self, request: Request) -> Response:
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32001, "message": "No autorizado"}},
                                status_code=401)
        body = await request.json()
        method = body.get("method")
        self.methods.append(method)
        self.versions.append(request.headers.get("mcp-protocol-version"))
        if "id" not in body:
            return Response(status_code=202)
        version = request.headers.get("mcp-protocol-version")
        if method != "initialize" and version is not None and version not in SUPPORTED_VERSIONS:
            return JSONResponse({"jsonrpc": "2.0", "id": body["id"], "error": {
                "code": -32022, "message": "Versión del protocolo no soportada.",
                "data": {"supported": list(SUPPORTED_VERSIONS), "requested": version}}}, status_code=400)
        return JSONResponse({"jsonrpc": "2.0", "id": body["id"], **self._answer(method, body.get("params") or {})})

    def _answer(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            asked = params.get("protocolVersion")
            version = asked if asked in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[-1]
            return {"result": {"protocolVersion": version, "capabilities": {"tools": {}},
                               "serverInfo": {"name": "sga-central", "version": "1.0.0"}}}
        if method == "ping":
            return {"result": {}}
        if method == "tools/list":
            return {"result": {"tools": TOOLS}}
        if method == "tools/call":
            return self._call(params.get("name"), params.get("arguments") or {})
        return {"error": {"code": -32601, "message": f"Method not found: {method}"}}

    def _call(self, name: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in {tool["name"] for tool in TOOLS}:
            return {"error": {"code": -32602, "message": f"Unknown tool: {name}"}}
        self.calls.append((name, arguments))
        if name == self.failing_tool:
            return {"result": {"content": [{"type": "text", "text": "No existe el incidente."}], "isError": True}}
        data = self.case if name == "ver_caso" else {"fragmentos": ["Ante la duda, se atiende como el más grave."]}
        return {"result": {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}],
                           "isError": False}}
