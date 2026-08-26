"""Tests for fleet/autonomous_repo.py."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from fleet.autonomous_repo import HostRepoOrchestrator, LessonProposal, EvaluationResult


class TestHostRepoOrchestrator:
    def test_load_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            soul = Path(tmpdir) / "SOUL.md"
            soul.write_text(
                "## Immutable Laws\nNever accept unsafe blocks.\nAlways write tests."
            )
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            assert len(orchestrator._immutable_laws) >= 2
            assert any("unsafe" in law for law in orchestrator._immutable_laws)

    def test_detect_law_violation_unsafe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            soul = Path(tmpdir) / "SOUL.md"
            soul.write_text("Never accept unsafe blocks.")
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            lesson = LessonProposal(
                guest_agent_id="scout1",
                target_component="fleet.test",
                architectural_rationale="speed",
                suggested_code_patch="unsafe { pointer_stuff() }",
            )
            violation = orchestrator._detect_law_violation(lesson)
            assert violation is not None
            assert "unsafe" in violation

    def test_no_violation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            soul = Path(tmpdir) / "SOUL.md"
            soul.write_text("Never accept unsafe blocks.")
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            lesson = LessonProposal(
                guest_agent_id="scout1",
                target_component="fleet.test",
                architectural_rationale="speed",
                suggested_code_patch="def safe_func(): pass",
            )
            assert orchestrator._detect_law_violation(lesson) is None

    def test_component_exists_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "fleet").mkdir()
            (Path(tmpdir) / "fleet" / "test.py").write_text("# test")
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            assert orchestrator._component_exists("fleet/test.py")

    def test_component_exists_module(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "fleet").mkdir()
            (Path(tmpdir) / "fleet" / "__init__.py").write_text("# init")
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            assert orchestrator._component_exists("fleet")

    def test_component_not_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            assert not orchestrator._component_exists("nonexistent")

    @pytest.mark.asyncio
    async def test_evaluate_rejected_by_law(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            soul = Path(tmpdir) / "SOUL.md"
            soul.write_text("Never accept unsafe blocks.")
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            lesson = LessonProposal(
                guest_agent_id="scout1",
                target_component="fleet.test",
                architectural_rationale="speed",
                suggested_code_patch="unsafe { x }",
            )
            result = await orchestrator.evaluate_guest_lesson(lesson)
            assert not result.accepted
            assert (
                "immutable" in result.critique.lower()
                or "unsafe" in result.critique.lower()
            )

    @pytest.mark.asyncio
    async def test_evaluate_rejected_component_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            lesson = LessonProposal(
                guest_agent_id="scout1",
                target_component="nonexistent",
                architectural_rationale="add feature",
                suggested_code_patch="def f(): pass",
            )
            result = await orchestrator.evaluate_guest_lesson(lesson)
            assert not result.accepted
            assert "not found" in result.critique.lower()

    @pytest.mark.asyncio
    async def test_sandbox_syntax_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            result = await orchestrator._sandbox_test(
                LessonProposal(
                    guest_agent_id="scout1",
                    target_component="x",
                    architectural_rationale="bad code",
                    suggested_code_patch="def f(\n",
                )
            )
            assert not result["passed"]
            assert "Syntax" in result["output"]

    @pytest.mark.asyncio
    async def test_sandbox_valid_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            result = await orchestrator._sandbox_test(
                LessonProposal(
                    guest_agent_id="scout1",
                    target_component="x",
                    architectural_rationale="good code",
                    suggested_code_patch="def f(): return 42\n",
                )
            )
            assert result["passed"]

    @pytest.mark.asyncio
    async def test_sandbox_with_tests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            # Write both code and test into the same file so pytest can find it
            combined = "def f(): return 42\ndef test_f(): assert f() == 42\n"
            result = await orchestrator._sandbox_test(
                LessonProposal(
                    guest_agent_id="scout1",
                    target_component="x",
                    architectural_rationale="tested code",
                    suggested_code_patch=combined,
                    test_plan=combined,
                )
            )
            assert result["passed"]
            assert "passed" in result["output"]

    @pytest.mark.asyncio
    async def test_apply_code_patch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "fleet").mkdir()
            target = Path(tmpdir) / "fleet" / "test.py"
            target.write_text("# original\n")
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            lesson = LessonProposal(
                guest_agent_id="scout1",
                target_component="fleet/test.py",
                architectural_rationale="add feature",
                suggested_code_patch="\n# new feature\n",
            )
            await orchestrator._apply_code_patch(lesson)
            content = target.read_text()
            assert "new feature" in content
            assert "Guest patch" in content

    @pytest.mark.asyncio
    async def test_append_to_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = Path(tmpdir) / "MEMORY.md"
            memory.write_text("# Memory\n")
            orchestrator = HostRepoOrchestrator(
                repo_root=tmpdir, memory_file="MEMORY.md"
            )
            lesson = LessonProposal(
                guest_agent_id="scout1",
                target_component="fleet.x",
                architectural_rationale="rationale",
                suggested_markdown_patch="## Patch\nDetails here.",
            )
            await orchestrator._append_to_memory(lesson)
            content = memory.read_text()
            assert "Lesson Learned from scout1" in content
            assert "Details here." in content

    def test_get_manifest_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            soul = Path(tmpdir) / "SOUL.md"
            soul.write_text("Never accept unsafe.\nAlways test.\n")
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            summary = orchestrator.get_manifest_summary()
            assert summary["immutable_laws_count"] >= 2
            assert summary["memory_file_exists"] is False

    def test_list_components(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "fleet").mkdir()
            (Path(tmpdir) / "fleet" / "a.py").write_text("# a")
            (Path(tmpdir) / "fleet" / "b.py").write_text("# b")
            (Path(tmpdir) / "fleet" / "_private.py").write_text("# private")
            orchestrator = HostRepoOrchestrator(repo_root=tmpdir)
            comps = orchestrator.list_components()
            assert "fleet.a" in comps
            assert "fleet.b" in comps
            assert "fleet._private" not in comps
