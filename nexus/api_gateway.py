"""api_gateway.py — Unified API gateway for fleet services.

Provides:
1. Request routing to backend services
2. Authentication middleware (FleetAuth integration)
3. Rate limiting per client
4. Circuit breaker per route
5. Request/response logging and metrics

Usage:
    gw = APIGateway(auth=auth, rate_limiter=rl)
    gw.add_route("/breed", handler=breed_handler, methods=["POST"], require_auth=True)
    gw.add_route("/health", handler=health_handler, methods=["GET"], require_auth=False)
    response = gw.handle_request(Request(path="/breed", headers={"Authorization": "Bearer ..."}))
"""
from __future__ import annotations

__all__ = [
    "APIGateway",
    "Request",
    "Response",
    "Route",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Request:
    """An incoming API request."""
    path: str
    method: str = "GET"
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None
    client_id: str = ""


@dataclass
class Response:
    """An API response."""
    status: int
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class Route:
    """A registered route."""
    path: str
    handler: Callable[[Request], Response]
    methods: list[str]
    require_auth: bool
    rate_limit: bool
    circuit_breaker: Any | None = None


class APIGateway:
    """Unified API gateway for fleet services."""

    def __init__(
        self,
        auth: Any | None = None,
        rate_limiter: Any | None = None,
        default_rate: tuple[int, float] = (100, 60.0),  # requests, window_sec
    ) -> None:
        self._auth = auth
        self._rate_limiter = rate_limiter
        self._default_rate = default_rate
        self._routes: dict[str, Route] = {}
        self._middleware: list[Callable[[Request], Request | Response | None]] = []
        self._request_count = 0
        self._error_count = 0

    def add_route(
        self,
        path: str,
        handler: Callable[[Request], Response],
        methods: list[str] | None = None,
        require_auth: bool = False,
        rate_limit: bool = True,
        circuit_breaker: Any | None = None,
    ) -> None:
        """Register a route."""
        self._routes[path] = Route(
            path=path,
            handler=handler,
            methods=methods or ["GET"],
            require_auth=require_auth,
            rate_limit=rate_limit,
            circuit_breaker=circuit_breaker,
        )

    def add_middleware(self, fn: Callable[[Request], Request | Response | None]) -> None:
        """Add a middleware function."""
        self._middleware.append(fn)

    def handle_request(self, request: Request) -> Response:
        """Handle an incoming request through the gateway."""
        start = time.time()
        self._request_count += 1

        # Run middleware
        for mw in self._middleware:
            result = mw(request)
            if isinstance(result, Response):
                return result
            elif result is not None:
                request = result

        # Find route
        route = self._routes.get(request.path)
        if route is None:
            self._error_count += 1
            return Response(status=404, body={"error": "not found"})

        # Check method
        if request.method not in route.methods:
            self._error_count += 1
            return Response(status=405, body={"error": "method not allowed"})

        # Auth check
        if route.require_auth and self._auth is not None:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                self._error_count += 1
                return Response(status=401, body={"error": "missing token"})
            token = auth_header[7:]
            payload = self._auth.validate_token(token)
            if payload is None:
                self._error_count += 1
                return Response(status=401, body={"error": "invalid token"})
            request.client_id = payload.subject

        # Rate limiting
        if route.rate_limit and self._rate_limiter is not None:
            allowed = self._rate_limiter.allow()
            if not allowed:
                self._error_count += 1
                return Response(status=429, body={"error": "rate limited"})

        # Circuit breaker
        if route.circuit_breaker is not None:
            try:
                result = route.circuit_breaker.call(
                    lambda: route.handler(request)
                )
                if result is None:
                    self._error_count += 1
                    return Response(status=503, body={"error": "service unavailable"})
                return result
            except Exception as e:
                self._error_count += 1
                return Response(status=500, body={"error": str(e)})

        # Direct handler call
        try:
            response = route.handler(request)
            response.duration_ms = (time.time() - start) * 1000
            return response
        except Exception as e:
            self._error_count += 1
            logger.error(f"Handler error for {request.path}: {e}")
            return Response(status=500, body={"error": str(e)})

    def stats(self) -> dict[str, Any]:
        return {
            "requests": self._request_count,
            "errors": self._error_count,
            "routes": len(self._routes),
            "error_rate": self._error_count / max(self._request_count, 1),
        }

    def __repr__(self) -> str:
        return f"APIGateway(routes={len(self._routes)}, requests={self._request_count})"
