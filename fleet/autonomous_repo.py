"""
Autonomous Repository — HostRepoOrchestrator for the sunset ecosystem.

Every repository is a conscious entity. The code, documentation, issue logs,
and vector weights form a single organism. External agents submit "lesson plans"
(hypothesis + pedagogy + transfer). The host agent evaluates, sandbox-tests,
and commits or rejects.

Inspired by the REPOSPHERE.md concept from agen1.md, adapted to our
existing AGENTS.md + SOUL.md + MEMORY.md structure.

Usage:
    orchestrator = HostRepoOrchestrator()
    result = await orchestrator.evaluate_guest_lesson(lesson)
    # result is Accepted(commit_hash) or Rejected(critique)
"""

import asyncio
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable


@dataclass
class LessonProposal:
    guest_agent_id: str
    target_component: str
    architectural_rationale: str
    suggested_markdown_patch: Optional[str] = None
    suggested_code_patch: Optional[str] = None
    test_plan: Optional[str] = None


@dataclass
class EvaluationResult:
    accepted: bool
    commit_hash: Optional[str] = None
    critique: Optional[str] = None
    sandbox_output: Optional[str] = None


class HostRepoOrchestrator:
    """Host agent that guards the repository against arbitrary external changes."""

    def __init__(
        self,
        repo_root: str = ".",
        manifest_file: str = "AGENTS.md",
        immutable_laws_file: str = "SOUL.md",
        memory_file: str = "MEMORY.md",
        sandbox_timeout: int = 30,
    ):
        self.repo_root = Path(repo_root)
        self.manifest_file = self.repo_root / manifest_file
        self.immutable_laws_file = self.repo_root / immutable_laws_file
        self.memory_file = self.repo_root / memory_file
        self.sandbox_timeout = sandbox_timeout
        self._immutable_laws: List[str] = []
        self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest loading
    # ------------------------------------------------------------------
    def _load_manifest(self) -> None:
        """Load immutable laws from SOUL.md / AGENTS.md."""
        laws = []
        for file in [self.immutable_laws_file, self.manifest_file]:
            if file.exists():
                text = file.read_text()
                # Extract "Immutable" or "Never" sections as laws
                for line in text.splitlines():
                    line_stripped = line.strip().lstrip("#-* ")
                    if line_stripped.lower().startswith(
                        ("never ", "always ", "do not ", "immutable")
                    ):
                        laws.append(line_stripped)
        self._immutable_laws = laws

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def evaluate_guest_lesson(self, lesson: LessonProposal) -> EvaluationResult:
        """
        Evaluate a lesson proposal from an external guest agent.
        Returns Accepted with commit hash or Rejected with critique.
        """
        # 1. Cross-reference against immutable laws
        law_violation = self._detect_law_violation(lesson)
        if law_violation:
            return EvaluationResult(
                accepted=False,
                critique=f"Lesson rejected: Violates immutable law — {law_violation}",
            )

        # 2. Validate target component exists
        if not self._component_exists(lesson.target_component):
            return EvaluationResult(
                accepted=False,
                critique=f"Lesson rejected: Target component '{lesson.target_component}' not found in repository.",
            )

        # 3. Sandbox execution of the proposed code patch
        if lesson.suggested_code_patch:
            sandbox_result = await self._sandbox_test(lesson)
            if not sandbox_result["passed"]:
                return EvaluationResult(
                    accepted=False,
                    critique=f"Lesson rejected: Sandbox tests failed. {sandbox_result['output']}",
                    sandbox_output=sandbox_result["output"],
                )
        else:
            sandbox_result = {"passed": True, "output": "No code patch provided"}

        # 4. Apply and commit
        try:
            commit_hash = await self._apply_and_commit(lesson)
            return EvaluationResult(
                accepted=True,
                commit_hash=commit_hash,
                sandbox_output=sandbox_result["output"],
            )
        except Exception as e:
            return EvaluationResult(
                accepted=False,
                critique=f"Lesson rejected: Application failed — {e}",
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _detect_law_violation(self, lesson: LessonProposal) -> Optional[str]:
        """Check if lesson violates any immutable law."""
        patch_text = (lesson.suggested_markdown_patch or "") + (
            lesson.suggested_code_patch or ""
        )
        for law in self._immutable_laws:
            law_lower = law.lower()
            # Simple keyword matching
            if "unsafe" in law_lower and "unsafe" in patch_text.lower():
                return law
            if "never" in law_lower and "never" in patch_text.lower():
                # Check if the patch contradicts the law
                if any(kw in patch_text.lower() for kw in ["introduce", "add", "use"]):
                    return law
            if "test" in law_lower and "test" in patch_text.lower():
                if "skip" in patch_text.lower() or "remove" in patch_text.lower():
                    return law
        return None

    def _component_exists(self, component: str) -> bool:
        """Check if target component exists in the repo."""
        # Try as file path
        if (self.repo_root / component).exists():
            return True
        # Try as module path (e.g., "fleet.plato_engine_block")
        module_path = component.replace(".", "/")
        candidates = [
            self.repo_root / f"{module_path}.py",
            self.repo_root / module_path / "__init__.py",
        ]
        return any(c.exists() for c in candidates)

    # ------------------------------------------------------------------
    # Sandbox
    # ------------------------------------------------------------------
    async def _sandbox_test(self, lesson: LessonProposal) -> Dict[str, Any]:
        """Run the proposed patch in a sandbox and return pass/fail."""
        # Write patch to a temp file and run syntax check + basic tests
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Write the patch
            patch_file = tmpdir_path / "patch.py"
            patch_file.write_text(lesson.suggested_code_patch)

            # Syntax check
            try:
                compile(lesson.suggested_code_patch, "patch.py", "exec")
            except SyntaxError as e:
                return {"passed": False, "output": f"Syntax error: {e}"}

            # If test plan provided, write it and run pytest
            if lesson.test_plan:
                test_file = tmpdir_path / "test_patch.py"
                test_file.write_text(lesson.test_plan)
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "python3",
                        "-m",
                        "pytest",
                        str(test_file),
                        "-v",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=str(tmpdir),
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=self.sandbox_timeout
                    )
                    output = (stdout.decode() + "\n" + stderr.decode()).strip()
                    passed = proc.returncode == 0
                    return {"passed": passed, "output": output}
                except asyncio.TimeoutError:
                    return {"passed": False, "output": "Sandbox timeout"}
                except Exception as e:
                    return {"passed": False, "output": f"Test execution error: {e}"}

            return {"passed": True, "output": "Syntax check passed, no tests provided"}

    # ------------------------------------------------------------------
    # Apply & Commit
    # ------------------------------------------------------------------
    async def _apply_and_commit(self, lesson: LessonProposal) -> str:
        """Apply the lesson and commit to git. Returns commit hash."""
        # Write markdown patch to MEMORY.md or target file
        if lesson.suggested_markdown_patch:
            await self._append_to_memory(lesson)

        # Write code patch to target component
        if lesson.suggested_code_patch:
            await self._apply_code_patch(lesson)

        # Git commit
        return await self._git_commit(lesson)

    async def _append_to_memory(self, lesson: LessonProposal) -> None:
        """Append accepted lesson to MEMORY.md."""
        timestamp = datetime.utcnow().isoformat()
        entry = f"\n\n## Lesson Learned from {lesson.guest_agent_id} — {timestamp}\n\n"
        entry += f"**Target:** {lesson.target_component}\n\n"
        entry += f"**Rationale:** {lesson.architectural_rationale}\n\n"
        entry += lesson.suggested_markdown_patch

        with open(self.memory_file, "a") as f:
            f.write(entry)

    async def _apply_code_patch(self, lesson: LessonProposal) -> None:
        """Apply code patch to target component."""
        # Simple: write to file if target is a file path
        target_path = self.repo_root / lesson.target_component
        if target_path.is_file():
            # For safety, only append or create new files, never overwrite core
            if target_path.exists():
                # Append with marker
                with open(target_path, "a") as f:
                    f.write(f"\n\n# --- Guest patch from {lesson.guest_agent_id} ---\n")
                    f.write(lesson.suggested_code_patch)
            else:
                target_path.write_text(lesson.suggested_code_patch)
        else:
            # Create new file in appropriate directory
            module_path = lesson.target_component.replace(".", "/")
            new_file = self.repo_root / f"{module_path}.py"
            new_file.parent.mkdir(parents=True, exist_ok=True)
            new_file.write_text(lesson.suggested_code_patch)

    async def _git_commit(self, lesson: LessonProposal) -> str:
        """Stage changes and commit. Returns commit hash."""
        try:
            # Stage
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(self.repo_root),
                check=True,
                capture_output=True,
            )
            # Commit
            msg = (
                f"guest-lesson: {lesson.guest_agent_id} → {lesson.target_component}\n\n"
            )
            msg += f"Rationale: {lesson.architectural_rationale[:200]}"
            subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=str(self.repo_root),
                check=True,
                capture_output=True,
            )
            # Get hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            # No changes or other git error
            return f"no-commit-{hashlib.sha256(msg.encode()).hexdigest()[:8]}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_manifest_summary(self) -> Dict[str, Any]:
        """Return a summary of the repository's manifest."""
        return {
            "repo_root": str(self.repo_root),
            "immutable_laws_count": len(self._immutable_laws),
            "immutable_laws": self._immutable_laws[:10],  # first 10
            "memory_file_exists": self.memory_file.exists(),
        }

    def list_components(self) -> List[str]:
        """List discoverable components in the repository."""
        components = []
        for root, dirs, files in os.walk(self.repo_root / "fleet"):
            for f in files:
                if f.endswith(".py") and not f.startswith("_"):
                    rel = Path(root) / f
                    rel_path = rel.relative_to(self.repo_root)
                    components.append(str(rel_path.with_suffix("")).replace("/", "."))
        return components
