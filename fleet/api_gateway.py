"""API gateway with routing, auth, and rate limiting.

Provides a lightweight API gateway for fleet services with request
routing, simple auth, and rate limit integration. Supports path-based
routing and middleware chains. Used for fleet service exposure, load
balancing, and request filtering.

Usage:
    gw = APIGateway()
    gw.route("/users", target="users-service")
    gw.add_middleware("auth", lambda req: req.get("token") == "valid")
    result = gw.process({"path": "/users", "token": "valid"})
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class APIGateway:
    """
    Lightweight API gateway for service routing.
    """

    def __init__(self):
        self._routes: Dict[str, str] = {}  # path -> service_name
        self._middleware: List[Dict[str, Any]] = []  # ordered middleware
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._request_count = 0
        self._error_count = 0

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, path: str, target: str) -> None:
        """
        Register a route.

        :param path: URL path pattern (exact match).
        :param target: Target service name.
        """
        self._routes[path] = target

    def remove_route(self, path: str) -> bool:
        """Remove a route."""
        if path in self._routes:
            del self._routes[path]
            return True
        return False

    def get_target(self, path: str) -> Optional[str]:
        """Get target service for a path."""
        # Exact match first
        if path in self._routes:
            return self._routes[path]
        # Prefix match
        for route_path, target in sorted(self._routes.items(), key=lambda x: -len(x[0])):
            if path.startswith(route_path):
                return target
        return None

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    def add_middleware(self, name: str, fn: Callable[[Dict[str, Any]], bool], priority: int = 0) -> None:
        """
        Add a middleware function.

        :param name: Middleware identifier.
        :param fn: Function that takes request dict, returns True to continue.
        :param priority: Lower number = higher priority.
        """
        self._middleware.append({"name": name, "fn": fn, "priority": priority})
        self._middleware.sort(key=lambda m: m["priority"])

    def remove_middleware(self, name: str) -> bool:
        """Remove middleware by name."""
        before = len(self._middleware)
        self._middleware = [m for m in self._middleware if m["name"] != name]
        return len(self._middleware) < before

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def register_handler(self, service: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        """Register a handler for a service."""
        self._handlers[service] = handler

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request through the gateway.

        :param request: Request dict with at least "path" key.
        :returns: Response dict.
        """
        self._request_count += 1

        # Run middleware
        for mw in self._middleware:
            if not mw["fn"](request):
                self._error_count += 1
                return {
                    "status": "rejected",
                    "reason": f"middleware:{mw['name']}",
                }

        path = request.get("path", "")
        target = self.get_target(path)
        if target is None:
            self._error_count += 1
            return {"status": "not_found", "path": path}

        handler = self._handlers.get(target)
        if handler is None:
            self._error_count += 1
            return {"status": "no_handler", "service": target}

        try:
            result = handler(request)
            return {"status": "ok", "result": result, "service": target}
        except Exception as e:
            self._error_count += 1
            return {"status": "error", "error": str(e), "service": target}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def routes(self) -> List[str]:
        return list(self._routes.keys())

    def middleware_names(self) -> List[str]:
        return [m["name"] for m in self._middleware]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "routes": len(self._routes),
            "middleware": len(self._middleware),
            "requests": self._request_count,
            "errors": self._error_count,
        }

    def __repr__(self) -> str:
        return f"<APIGateway routes={len(self._routes)} middleware={len(self._middleware)}>"
