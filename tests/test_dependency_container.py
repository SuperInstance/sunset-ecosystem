"""Tests for dependency_container.py — Simple DI container.

Run: python3 -m pytest tests/test_dependency_container.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.dependency_container import (
    DependencyContainer,
    CircularDependency,
    ServiceNotFound,
)


class TestDependencyContainer:
    def test_create(self):
        c = DependencyContainer()
        assert c.stats()["services"] == 0

    def test_register_and_resolve(self):
        c = DependencyContainer()
        c.register("greeting", lambda: "hello")
        assert c.resolve("greeting") == "hello"

    def test_resolve_not_found(self):
        c = DependencyContainer()
        with pytest.raises(ServiceNotFound):
            c.resolve("missing")

    def test_singleton(self):
        c = DependencyContainer()
        counter = [0]

        def make():
            counter[0] += 1
            return {"id": counter[0]}

        c.register("obj", make, lifecycle="singleton")
        a = c.resolve("obj")
        b = c.resolve("obj")
        assert a is b
        assert counter[0] == 1

    def test_transient(self):
        c = DependencyContainer()
        c.register("obj", lambda: {"id": 1}, lifecycle="transient")
        a = c.resolve("obj")
        b = c.resolve("obj")
        assert a is not b

    def test_dependency_injection(self):
        c = DependencyContainer()
        c.register("db", lambda: "db_conn")
        c.register("api", lambda db: f"api_with_{db}", deps=["db"])
        assert c.resolve("api") == "api_with_db_conn"

    def test_circular_dependency(self):
        c = DependencyContainer()
        c.register("a", lambda b: f"a+{b}", deps=["b"])
        c.register("b", lambda a: f"b+{a}", deps=["a"])
        with pytest.raises(CircularDependency):
            c.resolve("a")

    def test_has(self):
        c = DependencyContainer()
        c.register("x", lambda: 1)
        assert c.has("x") is True
        assert c.has("y") is False

    def test_list_services(self):
        c = DependencyContainer()
        c.register("a", lambda: 1)
        c.register("b", lambda: 2)
        assert sorted(c.list_services()) == ["a", "b"]

    def test_remove(self):
        c = DependencyContainer()
        c.register("x", lambda: 1)
        assert c.remove("x") is True
        assert c.has("x") is False
        assert c.remove("missing") is False

    def test_clear(self):
        c = DependencyContainer()
        c.register("a", lambda: 1)
        c.clear()
        assert c.list_services() == []

    def test_singletons_list(self):
        c = DependencyContainer()
        c.register("s", lambda: 1, lifecycle="singleton")
        c.register("t", lambda: 2, lifecycle="transient")
        assert c.singletons() == ["s"]

    def test_kwargs_override(self):
        c = DependencyContainer()
        c.register("greet", lambda who: f"hello {who}")
        assert c.resolve("greet", who="world") == "hello world"

    def test_repr(self):
        c = DependencyContainer()
        assert "DependencyContainer" in repr(c)
