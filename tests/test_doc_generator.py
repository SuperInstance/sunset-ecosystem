"""
Tests for Fleet Documentation Generator.

Covers: ModuleDoc, DocGenerator.
"""

import os
import tempfile

import pytest

from fleet.doc_generator import ModuleDoc, DocGenerator


class TestModuleDoc:
    def test_to_dict(self):
        mod = ModuleDoc(name="test.py", path="test.py", docstring="Test module")
        d = mod.to_dict()
        assert d["name"] == "test.py"
        assert d["docstring"] == "Test module"


class TestDocGenerator:
    def test_init(self):
        gen = DocGenerator("test-project")
        assert gen.project_name == "test-project"
        assert gen.modules == []

    def test_scan_file(self):
        gen = DocGenerator()
        # Create a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('"""A test module."""\n')
            f.write("class TestClass:\n")
            f.write('    """A test class."""\n')
            f.write("    def method(self): pass\n")
            f.write("def func(a, b):\n")
            f.write('    """A function."""\n')
            f.write("    pass\n")
            path = f.name

        gen._scan_file(path)
        os.unlink(path)

        assert len(gen.modules) == 1
        mod = gen.modules[0]
        assert mod.docstring == "A test module."
        assert len(mod.classes) == 1
        assert mod.classes[0]["name"] == "TestClass"
        assert len(mod.functions) == 2  # func + method (ast.walk finds all)
        assert mod.functions[0]["name"] == "func"

    def test_scan_file_no_docstring(self):
        gen = DocGenerator()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1\n")
            path = f.name

        gen._scan_file(path)
        os.unlink(path)

        assert len(gen.modules) == 1
        assert gen.modules[0].docstring == ""

    def test_generate_module_index(self):
        gen = DocGenerator()
        gen.modules = [
            ModuleDoc(
                name="a.py",
                path="a.py",
                classes=[{"name": "A"}],
                functions=[{"name": "f"}],
                test_count=5,
            ),
            ModuleDoc(name="b.py", path="b.py", classes=[], functions=[], test_count=0),
        ]
        gen.total_tests = 5
        md = gen.generate_module_index()
        assert "a.py" in md
        assert "b.py" in md
        assert "5" in md

    def test_generate_api_reference(self):
        gen = DocGenerator()
        gen.modules = [
            ModuleDoc(
                name="test.py",
                path="test.py",
                docstring="Test module",
                classes=[
                    {"name": "MyClass", "docstring": "A class", "methods": ["method"]}
                ],
                functions=[
                    {"name": "my_func", "docstring": "A function", "args": ["a", "b"]}
                ],
            ),
        ]
        md = gen.generate_api_reference()
        assert "MyClass" in md
        assert "my_func" in md
        assert "method" in md
        assert "a, b" in md

    def test_generate_readme(self):
        gen = DocGenerator("my-project")
        gen.modules = [
            ModuleDoc(name="main.py", path="main.py", test_count=10),
        ]
        gen.total_tests = 10
        gen.total_lines = 100
        readme = gen.generate_readme()
        assert "my-project" in readme
        assert "Quick Start" in readme
        assert "main.py" in readme

    def test_generate_test_summary(self):
        gen = DocGenerator()
        gen.modules = [
            ModuleDoc(name="tested.py", path="tested.py", test_count=5),
            ModuleDoc(name="untested.py", path="untested.py", test_count=0),
        ]
        gen.total_tests = 5
        md = gen._generate_test_summary()
        assert "Tested Modules" in md
        assert "untested.py" in md

    def test_generate_agent_spec(self):
        gen = DocGenerator("my-project")
        gen.modules = [ModuleDoc(name="main.py", path="main.py", test_count=10)]
        gen.total_tests = 10
        spec = gen.generate_agent_spec()
        assert spec["project"] == "my-project"
        assert spec["total_tests"] == 10
        assert "entry_points" in spec

    def test_export_json(self):
        gen = DocGenerator("my-project")
        gen.modules = [ModuleDoc(name="main.py", path="main.py", test_count=5)]
        gen.total_tests = 5
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "docs.json")
            result = gen.export_json(path)
            assert os.path.exists(result)
            with open(result) as f:
                import json

                data = json.load(f)
                assert data["project"] == "my-project"

    def test_get_stats(self):
        gen = DocGenerator()
        gen.modules = [
            ModuleDoc(
                name="a.py",
                path="a.py",
                classes=[{"name": "A"}],
                functions=[{"name": "f"}],
            ),
            ModuleDoc(
                name="b.py",
                path="b.py",
                classes=[],
                functions=[{"name": "g"}, {"name": "h"}],
            ),
        ]
        stats = gen.get_stats()
        assert stats["modules"] == 2
        assert stats["classes"] == 1
        assert stats["functions"] == 3

    def test_count_tests(self):
        gen = DocGenerator()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test_one(): pass\n")
            f.write("def test_two(): pass\n")
            f.write("def helper(): pass\n")
            path = f.name
        count = gen._count_tests(path)
        os.unlink(path)
        assert count == 2

    def test_scan_modules(self):
        gen = DocGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "fleet"))
            with open(os.path.join(tmpdir, "fleet", "test.py"), "w") as f:
                f.write('"""Test."""\n')
                f.write("class C: pass\n")
            gen.scan_modules(["fleet/"], base_path=tmpdir)
            assert len(gen.modules) >= 1

    def test_empty_project(self):
        gen = DocGenerator("empty")
        readme = gen.generate_readme()
        assert "empty" in readme
        assert "0" in readme
