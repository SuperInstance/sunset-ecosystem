"""dependency_container.py — Simple dependency injection container.

Provides:
1. Register services by name or interface
2. Lazy initialization (factory functions)
3. Singleton and transient lifecycles
4. Constructor injection via auto-wire
5. Circular dependency detection

Usage:
    container = DependencyContainer()
    container.register("db", create_db_pool, lifecycle="singleton")
    container.register("api", create_api_client, deps=["db"])
    db = container.resolve("db")
    api = container.resolve("api")  # Auto-injects db
"""

from __future__ import annotations

__all__ = [
    "DependencyContainer",
    "CircularDependency",
    "ServiceNotFound",
]

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CircularDependency(Exception):
    """Raised when a circular dependency is detected."""


class ServiceNotFound(Exception):
    """Raised when a requested service is not registered."""


class _ServiceEntry:
    """Internal service registration."""

    def __init__(
        self,
        name: str,
        factory: Callable[..., Any],
        lifecycle: str = "transient",
        deps: list[str] | None = None,
    ) -> None:
        self.name = name
        self.factory = factory
        self.lifecycle = lifecycle
        self.deps = deps or []
        self.instance: Any = None
        self._lock = threading.Lock()


class DependencyContainer:
    """Simple DI container with lifecycle management."""

    def __init__(self) -> None:
        self._services: dict[str, _ServiceEntry] = {}
        self._singleton_lock = threading.Lock()
        self._resolution_stack: list[str] = []

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        lifecycle: str = "transient",
        deps: list[str] | None = None,
    ) -> None:
        """Register a service."""
        self._services[name] = _ServiceEntry(
            name=name,
            factory=factory,
            lifecycle=lifecycle,
            deps=deps,
        )

    def resolve(self, name: str, **kwargs: Any) -> Any:
        """Resolve a service by name."""
        if name in self._resolution_stack:
            chain = " -> ".join(self._resolution_stack + [name])
            raise CircularDependency(f"Circular dependency detected: {chain}")

        entry = self._services.get(name)
        if not entry:
            raise ServiceNotFound(f"Service '{name}' not registered")

        if entry.lifecycle == "singleton":
            if entry.instance is not None:
                return entry.instance
            with entry._lock:
                if entry.instance is None:
                    entry.instance = self._create(entry, **kwargs)
            return entry.instance

        return self._create(entry, **kwargs)

    def _create(self, entry: _ServiceEntry, **kwargs: Any) -> Any:
        """Create a service instance, resolving dependencies."""
        self._resolution_stack.append(entry.name)
        try:
            # Resolve declared dependencies
            dep_values: dict[str, Any] = {}
            for dep_name in entry.deps:
                dep_values[dep_name] = self.resolve(dep_name)

            # Override with provided kwargs
            dep_values.update(kwargs)

            return entry.factory(**dep_values)
        finally:
            self._resolution_stack.pop()

    def has(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._services

    def list_services(self) -> list[str]:
        """List all registered service names."""
        return list(self._services.keys())

    def remove(self, name: str) -> bool:
        """Remove a service registration."""
        if name in self._services:
            del self._services[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all registrations."""
        self._services.clear()

    def singletons(self) -> list[str]:
        """List all singleton service names."""
        return [
            name for name, e in self._services.items() if e.lifecycle == "singleton"
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "services": len(self._services),
            "singletons": len(self.singletons()),
        }

    def __repr__(self) -> str:
        return f"DependencyContainer(services={len(self._services)})"
