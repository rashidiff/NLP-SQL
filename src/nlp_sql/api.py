"""Minimal JSON HTTP API for parse and explain endpoints."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from nlp_sql.engine import NlpSqlEngine


def handle_api_request(
    path: str, payload: dict[str, Any], engine: NlpSqlEngine
) -> tuple[int, dict[str, Any]]:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return HTTPStatus.BAD_REQUEST, {
            "success": False,
            "error": {"code": "INVALID_REQUEST", "message": "Field 'query' is required."},
        }
    if path == "/parse":
        return HTTPStatus.OK, engine.parse(query).to_dict()
    if path == "/explain":
        return HTTPStatus.OK, engine.explain(query).to_dict()
    return HTTPStatus.NOT_FOUND, {
        "success": False,
        "error": {"code": "NOT_FOUND", "message": "Endpoint not found."},
    }


class NlpSqlRequestHandler(BaseHTTPRequestHandler):
    engine = NlpSqlEngine()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            status_code = int(HTTPStatus.BAD_REQUEST)
            body = {
                "success": False,
                "error": {"code": "INVALID_JSON", "message": "Request body must be JSON."},
            }
        else:
            status_code, body = handle_api_request(self.path, payload, self.engine)

        response = json.dumps(body).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        return


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = HTTPServer((host, port), NlpSqlRequestHandler)
    server.serve_forever()
