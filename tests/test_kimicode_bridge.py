"""Tests for kimicode_bridge.py — Kimi k2p6 coding bridge.

Run: python3 -m pytest tests/test_kimicode_bridge.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.kimicode_bridge import (
    CacheEntry,
    CodingTask,
    KimicodeBridge,
    TaskTemplate,
)


class TestCodingTask:
    def test_cache_key_deterministic(self):
        t1 = CodingTask(task_type="review", target="foo.py", context={"lines": 50})
        t2 = CodingTask(task_type="review", target="foo.py", context={"lines": 50})
        assert t1.cache_key() == t2.cache_key()

    def test_cache_key_differs(self):
        t1 = CodingTask(task_type="review", target="foo.py", context={"lines": 50})
        t2 = CodingTask(task_type="review", target="foo.py", context={"lines": 51})
        assert t1.cache_key() != t2.cache_key()

    def test_cache_key_stable(self):
        t = CodingTask(task_type="refactor", target="bar.py", context={"a": 1, "b": 2})
        k1 = t.cache_key()
        k2 = t.cache_key()
        assert k1 == k2
        assert len(k1) == 32


class TestTaskTemplate:
    def test_review_template(self):
        task = TaskTemplate.review("def foo(): pass")
        assert task.task_type == "review"
        assert "readability" in task.prompt
        assert task.target == "code_block"

    def test_refactor_template(self):
        task = TaskTemplate.refactor("def foo(): pass", "add type hints")
        assert task.task_type == "refactor"
        assert "add type hints" in task.prompt

    def test_test_gen_template(self):
        task = TaskTemplate.test_gen("def bar(x): return x * 2")
        assert task.task_type == "test"
        assert "pytest" in task.prompt
        assert "boundary conditions" in task.prompt

    def test_docs_template(self):
        task = TaskTemplate.docs("def foo(): pass", style="numpy")
        assert task.task_type == "docs"
        assert "numpy" in task.prompt

    def test_debug_template(self):
        task = TaskTemplate.debug("Traceback...", "def foo(): pass")
        assert task.task_type == "debug"
        assert "Traceback" in task.prompt


class TestKimicodeBridge:
    def test_create(self):
        bridge = KimicodeBridge(cache_size=50)
        assert bridge.cache_hit_rate == 0.0

    def test_cache_hit(self):
        bridge = KimicodeBridge()
        code = "def test(): pass"
        result1 = bridge.review(code)
        result2 = bridge.review(code)
        # Same input should hit cache
        assert bridge.cache_hit_rate == pytest.approx(0.5, abs=0.01)
        assert bridge.report()["cache_hits"] == 1
        assert bridge.report()["cache_misses"] == 1

    def test_cache_eviction(self):
        bridge = KimicodeBridge(cache_size=5)
        for i in range(10):
            bridge.review(f"def func_{i}(): pass")
        assert bridge.report()["cache_size"] <= 5

    def test_different_tasks_different_cache(self):
        bridge = KimicodeBridge()
        bridge.review("def a(): pass")
        bridge.refactor("def a(): pass", "add types")
        # Different tasks = different cache keys
        assert bridge.report()["cache_size"] == 2

    def test_batch_execution(self):
        bridge = KimicodeBridge()
        tasks = [
            TaskTemplate.review("def a(): pass"),
            TaskTemplate.docs("def b(): pass"),
        ]
        results = bridge.batch(tasks)
        assert len(results) == 2
        assert bridge.report()["requests"] == 2

    def test_metrics(self):
        bridge = KimicodeBridge()
        bridge.review("def x(): pass")
        r = bridge.report()
        assert "requests" in r
        assert "cache_hit_rate" in r
        assert "avg_latency_sec" in r
        assert r["requests"] == 1

    def test_clear_cache(self):
        bridge = KimicodeBridge()
        bridge.review("def x(): pass")
        assert bridge.report()["cache_size"] == 1
        bridge.clear_cache()
        assert bridge.report()["cache_size"] == 0

    def test_custom_task(self):
        bridge = KimicodeBridge()
        task = bridge.task("optimize", {"target": "loop.py", "goal": "vectorize"})
        assert task.task_type == "optimize"
        assert task.target == "loop.py"
        result = bridge.execute(task)
        assert "dispatched" in result

    def test_dispatch_structure(self):
        bridge = KimicodeBridge()
        task = TaskTemplate.review("code")
        result = bridge.execute(task)
        import json

        data = json.loads(result)
        assert data["status"] == "dispatched"
        assert data["model"] == "k2p6"
        assert "prompt_hash" in data
